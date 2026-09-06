
from __future__ import annotations
import time
import torch
import torch.nn as nn


def build_plan(model):
    entries, by_name = [], {}

    def add_conv(name, cbr, in_sources, prune_out=True):
        conv, bn = cbr.conv, cbr.bn

        def replace(nc, nb, _c=cbr):
            _c.conv, _c.bn = nc, nb

        e = dict(name=name, module=conv, bn=bn, kind="conv", replace=replace,
                 in_sources=list(in_sources), prune_out=prune_out,
                 orig_out=conv.out_channels, has_bias=conv.bias is not None,
                 kh=conv.kernel_size[0], kw=conv.kernel_size[1])
        entries.append(e); by_name[name] = e

    def add_linear(name, attr, in_sources, prune_out=True):
        lin = getattr(model, attr)

        def replace(nl, _nb, _m=model, _a=attr):
            setattr(_m, _a, nl)

        e = dict(name=name, module=lin, bn=None, kind="linear", replace=replace,
                 in_sources=list(in_sources), prune_out=prune_out,
                 orig_out=lin.out_features, has_bias=lin.bias is not None)
        entries.append(e); by_name[name] = e

    add_conv("conv1", model.conv1, [("INPUT", 1)])
    add_conv("conv2", model.conv2, [("conv1", 1)])
    add_conv("conv3", model.conv3, [("conv2", 1)])
    add_conv("conv4", model.conv4, [("conv3", 1)])
    add_conv("conv5", model.conv5, [("conv4", 1)])
    add_linear("fc1", "fc1", [("conv5", model.flatten_hw)])
    add_linear("fc2", "fc2", [("fc1", 1)])
    add_linear("fc3", "fc3", [("fc2", 1)], prune_out=False)
    return entries, by_name


def _src_orig_out(src, by_name):
    return 3 if src == "INPUT" else by_name[src]["orig_out"]


def params_from_counts(entries, by_name, counts):
    total = 0
    for e in entries:
        out_eff = counts[e["name"]] if e["prune_out"] else e["orig_out"]
        in_eff = 0
        for src, block in e["in_sources"]:
            in_eff += (3 if src == "INPUT" else counts[src]) * block
        if e["kind"] == "conv":
            total += out_eff * in_eff * e["kh"] * e["kw"]
            if e["bn"] is not None:
                total += 2 * out_eff
        else:
            total += out_eff * in_eff
        if e["has_bias"]:
            total += out_eff
    return total


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def compute_importance(model, entries, criterion, loss_fn, calib_batches, device):
    prunable = [e for e in entries if e["prune_out"]]
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    importance, n_images = {}, 0

    if criterion == "magnitude":
        for e in prunable:
            importance[e["name"]] = e["module"].weight.detach().abs().flatten(1).sum(1).cpu()
        needs_backward = False
    else:
        model.eval()
        gsum = {e["name"]: torch.zeros_like(e["module"].weight) for e in prunable}
        gsq = {e["name"]: torch.zeros_like(e["module"].weight) for e in prunable}
        nb = 0
        for batch in calib_batches:
            for p in model.parameters():
                p.grad = None
            loss, bs = loss_fn(model, batch)
            loss.backward()
            for e in prunable:
                g = e["module"].weight.grad
                if g is not None:
                    gsum[e["name"]] += g.detach()
                    gsq[e["name"]] += g.detach() ** 2
            nb += 1; n_images += bs
        for e in prunable:
            W = e["module"].weight.detach()
            g = gsum[e["name"]] / max(nb, 1)
            g2 = gsq[e["name"]] / max(nb, 1)
            if criterion == "gradient":
                imp = g.abs().flatten(1).sum(1)
            elif criterion == "taylor":
                imp = (g * W).flatten(1).sum(1) ** 2
            elif criterion == "hessian":
                imp = 0.5 * (g2 * W ** 2).flatten(1).sum(1)
            else:
                raise ValueError(criterion)
            importance[e["name"]] = imp.detach().cpu()
        for p in model.parameters():
            p.grad = None
        needs_backward = True

    if is_cuda:
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    overhead = {
        "time_s": dt, "n_images": n_images,
        "time_ms_per_image": (dt * 1e3 / n_images) if n_images else 0.0,
        "peak_mem_mb": (torch.cuda.max_memory_allocated(device) / 1e6) if is_cuda else float("nan"),
        "needs_backward": needs_backward,
    }
    return importance, overhead


def select_kept_indices(entries, by_name, importance, keep_param_frac,
                        min_keep=2, alloc="global"):
    if alloc != "global":
        raise NotImplementedError("samo alloc='global'")
    prunable = [e for e in entries if e["prune_out"]]
    counts = {e["name"]: e["orig_out"] for e in prunable}
    baseline = params_from_counts(entries, by_name, counts)
    target = keep_param_frac * baseline

    ranked = []
    for e in prunable:
        imp = importance[e["name"]].float()
        norm = imp / (imp.norm() + 1e-12)
        for ch in range(e["orig_out"]):
            ranked.append((float(norm[ch]), e["name"], ch))
    ranked.sort(key=lambda t: t[0])

    removed = {e["name"]: set() for e in prunable}
    cur = params_from_counts(entries, by_name, counts)
    for _v, name, ch in ranked:
        if cur <= target:
            break
        if counts[name] <= min_keep:
            continue
        removed[name].add(ch); counts[name] -= 1
        cur = params_from_counts(entries, by_name, counts)

    kept_idx = {e["name"]: torch.tensor([i for i in range(e["orig_out"])
                                         if i not in removed[e["name"]]], dtype=torch.long)
                for e in prunable}
    info = {
        "baseline_params": baseline, "target_params": int(target),
        "achieved_params": cur, "achieved_keep_frac": cur / baseline,
        "per_layer_kept": {n: len(kept_idx[n]) for n in kept_idx},
        "per_layer_orig": {e["name"]: e["orig_out"] for e in prunable},
    }
    return kept_idx, info


def _gather_in_idx(entry, by_name, kept_idx):
    parts, offset = [], 0
    for src, block in entry["in_sources"]:
        if src == "INPUT":
            keep_local = torch.arange(3); orig = 3
        else:
            keep_local = kept_idx[src]; orig = by_name[src]["orig_out"]
        base = (keep_local.view(-1, 1) * block + torch.arange(block).view(1, -1)).reshape(-1)
        parts.append(base + offset)
        offset += orig * block
    return torch.cat(parts)


def apply_pruning(model, entries, by_name, kept_idx):
    for e in entries:
        m = e["module"]
        out_idx = kept_idx[e["name"]] if e["prune_out"] else torch.arange(e["orig_out"])
        in_idx = _gather_in_idx(e, by_name, kept_idx)

        if e["kind"] == "conv":
            new = nn.Conv2d(len(in_idx), len(out_idx), m.kernel_size, m.stride,
                            m.padding, m.dilation, groups=1, bias=e["has_bias"]
                            ).to(m.weight.device, m.weight.dtype)
            with torch.no_grad():
                new.weight.copy_(m.weight.detach()[out_idx][:, in_idx])
                if e["has_bias"]:
                    new.bias.copy_(m.bias.detach()[out_idx])
            new_bn = None
            if e["bn"] is not None:
                bn = e["bn"]
                new_bn = nn.BatchNorm2d(len(out_idx), eps=bn.eps, momentum=bn.momentum,
                                        affine=bn.affine,
                                        track_running_stats=bn.track_running_stats
                                        ).to(bn.weight.device, bn.weight.dtype)
                with torch.no_grad():
                    new_bn.weight.copy_(bn.weight.detach()[out_idx])
                    new_bn.bias.copy_(bn.bias.detach()[out_idx])
                    if bn.track_running_stats:
                        new_bn.running_mean.copy_(bn.running_mean.detach()[out_idx])
                        new_bn.running_var.copy_(bn.running_var.detach()[out_idx])
                        new_bn.num_batches_tracked.copy_(bn.num_batches_tracked.detach())
            e["replace"](new, new_bn)
        else:
            new = nn.Linear(len(in_idx), len(out_idx), bias=e["has_bias"]
                            ).to(m.weight.device, m.weight.dtype)
            with torch.no_grad():
                new.weight.copy_(m.weight.detach()[out_idx][:, in_idx])
                if e["has_bias"]:
                    new.bias.copy_(m.bias.detach()[out_idx])
            e["replace"](new, None)
    return model


@torch.no_grad()
def count_flops(model, device, imgsz):
    total = [0]
    handles = []

    def conv_hook(m, i, o):
        total[0] += 2 * o.shape[1] * m.in_channels * m.kernel_size[0] * m.kernel_size[1] \
            * o.shape[2] * o.shape[3] // m.groups

    def lin_hook(m, i, o):
        total[0] += 2 * m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(lin_hook))
    was = model.training
    model.eval()
    model(torch.randn(1, 3, imgsz, imgsz, device=device))
    for h in handles:
        h.remove()
    if was:
        model.train()
    return total[0]


@torch.no_grad()
def recalibrate_bn(model, calib_batches, get_imgs, device, reset=True):
    if reset:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None
    model.train()
    for batch in calib_batches:
        model(get_imgs(batch).to(device, non_blocking=True))
    model.eval()
    return model


def _selfcheck():
    from model_cnn import SchoolCNN, INPUT_SIZE
    device = torch.device("cpu")
    model = SchoolCNN().to(device).eval()
    entries, by_name = build_plan(model)

    counts = {e["name"]: e["orig_out"] for e in entries if e["prune_out"]}
    calc, real = params_from_counts(entries, by_name, counts), count_params(model)
    print(f"params_from_counts={calc:,}  real={real:,}  match={calc == real}")
    assert calc == real

    imp = {e["name"]: e["module"].weight.detach().abs().flatten(1).sum(1)
           for e in entries if e["prune_out"]}
    kept, info = select_kept_indices(entries, by_name, imp, keep_param_frac=0.70)
    apply_pruning(model, entries, by_name, kept)
    pruned = count_params(model)
    print(f"baseline={info['baseline_params']:,}  target(70%)={info['target_params']:,}  "
          f"achieved={pruned:,}  keep={pruned/info['baseline_params']*100:.1f}%")
    print("per-layer kept:", info["per_layer_kept"])

    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        y = model(x)
    print(f"output {tuple(y.shape)} (ocek. (2,6))  | GFLOPs={count_flops(model, device, INPUT_SIZE)/1e9:.3f}")
    assert y.shape == (2, 6)
    print("SELF-CHECK OK")


if __name__ == "__main__":
    _selfcheck()
