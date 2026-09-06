
import copy
import json
import os
import sys

import torch
import torch.nn as nn

_MORPH = "/home/tomi/code/dipl/morphology"
if _MORPH not in sys.path:
    sys.path.insert(0, _MORPH)

REGISTER_PATH = "/home/tomi/code/dipl/arch_agnostic/LAYER_REGISTER.json"

SIZE_LADDER = (96, 128, 224, 320, 640)
SIZE_LADDER_1D = (1024, 4096, 8000, 16000, 48000)
SEQ_LADDER_TOK = (8, 16, 32, 64)
MAX_REPRESENTATIVES = 3


class ProbeAdapter:

    kind = "probe"

    def __init__(self, call, size, in_ch, mode):
        self._call = call
        self.imgsz = size
        self._in_ch = in_ch
        self._mode = mode
        self.flexible = None

    def _one(self, device):
        if self._mode == "vector":
            return torch.rand(self._in_ch, device=device)
        if self._mode == "seq":
            return torch.rand(self._in_ch, self.imgsz, device=device)
        if self._mode == "token":
            return torch.randint(0, self._in_ch, (self.imgsz,), dtype=torch.long, device=device)
        return torch.rand(self._in_ch, self.imgsz, self.imgsz, device=device)

    def forward_example(self, device):
        return [self._one(device)]

    def tp_example(self, device):
        x = self._one(device).unsqueeze(0)
        return [x[0]] if self._call == "list" else x

    def forward(self, model, imgs):
        if self._call == "kwargs":
            return _unwrap(model(input_ids=torch.stack([im for im in imgs])))
        if self._call == "list":
            return _unwrap(model(list(imgs)))
        return _unwrap(model(torch.stack([im for im in imgs])))

    @torch.no_grad()
    def teacher_outputs(self, model, imgs):
        return _detach(self.forward(model, imgs))


def _unwrap(o):
    if hasattr(o, "logits"):
        return o.logits
    if hasattr(o, "to_tuple"):
        return o.to_tuple()
    return o


def _detach(o):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu()
    if isinstance(o, dict):
        return {k: _detach(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_detach(v) for v in o]
    return o


def _finite(o):
    if isinstance(o, torch.Tensor):
        return bool(torch.isfinite(o).all().item()) if o.numel() else True
    if isinstance(o, dict):
        return all(_finite(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return all(_finite(v) for v in o)
    return True


def _input_spec(model):
    for _, m in model.named_modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() in (3, 4):
            return int(w.shape[1]) * int(getattr(m, "groups", 1)), ("seq" if w.dim() == 3 else "image")
    for _, m in model.named_modules():
        if isinstance(m, nn.Embedding):
            return int(m.num_embeddings), "token"
    for _, m in model.named_modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            return int(w.shape[1]), "vector"
    return 3, "image"


def probe_adapter(model, device, verbose=True):
    in_ch, mode = _input_spec(model)
    ladder = {"vector": (in_ch,), "seq": SIZE_LADDER_1D, "image": SIZE_LADDER, "token": SEQ_LADDER_TOK}[mode]
    calls = ("kwargs",) if mode == "token" else ("list", "batch")
    model.eval()
    ok = []
    for call in calls:
        for sz in sorted(ladder, reverse=True):
            a = ProbeAdapter(call, sz, in_ch, mode)
            try:
                with torch.no_grad():
                    out = a.forward(model, a.forward_example(device))
                if _finite(out):
                    ok.append((call, sz))
            except BaseException:
                pass
        if ok:
            break
    if not ok:
        return None
    call, sz = ok[0]
    a = ProbeAdapter(call, sz, in_ch, mode)
    a.flexible = len({s for _, s in ok}) > 1
    if verbose:
        unit = "vocab" if mode == "token" else "ch"
        print(f"[probe] poziv={call} · ulaz={in_ch}{unit} @ {sz} ({mode}) · "
              f"fleksibilan={a.flexible} · radne velicine={sorted({s for _, s in ok})}")
    return a


def load_register(path=REGISTER_PATH):
    if not os.path.exists(path):
        return {"defaults": {"unlisted": "rules_decide"}, "types": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fqn(m):
    t = type(m)
    return f"{t.__module__}.{t.__name__}"


def weighted_leaves(model):
    out = []
    for name, m in model.named_modules():
        if list(m.children()):
            continue
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() in (2, 3, 4):
            out.append((name, m, type(m).__name__, w))
    return out


def _reg_entry(reg, m):
    types = reg.get("types", {})
    return types.get(fqn(m)) or types.get(type(m).__name__) or {}


def _shapes(model, adapter, device):
    rec, handles = {}, []

    def mk(name):
        def hook(mod, inp, out):
            ish = tuple(inp[0].shape) if inp and isinstance(inp[0], torch.Tensor) else None
            osh = tuple(out.shape) if isinstance(out, torch.Tensor) else None
            rec[name] = (ish, osh)
        return hook

    for name, m in model.named_modules():
        if not list(m.children()):
            handles.append(m.register_forward_hook(mk(name)))
    model.eval()
    try:
        with torch.no_grad():
            adapter.forward(model, adapter.forward_example(device))
    except BaseException:
        pass
    for h in handles:
        h.remove()
    return rec


def _is_activation(m, device):
    try:
        x = torch.randn(1, 4, 8, 8, device=device)
        with torch.no_grad():
            y0 = m(x.clone())
            if not isinstance(y0, torch.Tensor) or y0.shape != x.shape:
                return False
            if torch.allclose(y0, x):
                return False
            x2 = x.clone()
            x2[0, 0, 3, 3] += 5.0
            y1 = m(x2.clone())
        moved = int(((y0 - y1).abs() > 1e-6).sum().item())
        return moved <= 1
    except BaseException:
        return False


def classify_leaf(name, m, shapes, reg, device):
    ent = _reg_entry(reg, m)
    if ent.get("status") == "hazard":
        return dict(morph=False, is_activation=False, is_unknown=False,
                    why=f"hazard: {ent.get('reason', 'poznato ogranicenje')} — zasticeno, poznat slucaj")

    w = getattr(m, "weight", None)
    has_w = isinstance(w, torch.Tensor)
    nparams = sum(p.numel() for p in m.parameters(recurse=False))
    fired = name in shapes
    ish, osh = shapes.get(name, (None, None))

    if has_w and w.dim() >= 2:
        axis = int(ent.get("out_axis", 0))
        n_out = int(w.shape[axis])
        declared = getattr(m, "out_channels", None)
        if declared is None:
            declared = getattr(m, "out_features", None)

        if osh is not None:
            measured = osh[-1] if w.dim() == 2 else osh[1]
            if measured != n_out:
                return dict(morph=False, is_activation=False, is_unknown=True,
                            why=f"shape[{axis}]={n_out} != izmjerena izlazna sirina {measured}")
            src = "forward"
        elif declared is not None:
            if declared != n_out:
                return dict(morph=False, is_activation=False, is_unknown=True,
                            why=f"shape[{axis}]={n_out} != deklarirano {declared}")
            src = "deklaracija"
        else:
            return dict(morph=False, is_activation=False, is_unknown=True,
                        why="nema dokaza o izlaznoj sirini (ni forward ni out_channels/out_features)")

        g = int(getattr(m, "groups", 1))
        if g != 1:
            dw = g == getattr(m, "in_channels", -1) == getattr(m, "out_channels", -2)
            return dict(morph=False, is_activation=False, is_unknown=False,
                        why=f"{'depthwise' if dw else 'grouped'} (groups={g}) — sirina vezana uz producenta")
        return dict(morph=True, is_activation=False, is_unknown=False,
                    why=f"out={n_out} potvrdjen ({src})")

    if nparams and (hasattr(m, "running_mean") or (has_w and w.dim() == 1)):
        return dict(morph=False, is_activation=False, is_unknown=False,
                    why="norm — sirina vezana uz producenta")

    if nparams == 0:
        if not fired:
            return dict(morph=False, is_activation=False, is_unknown=False,
                        why="0 param i izvan compute-grafa (nije se izvrsio) — nebitno za kompresiju")
        if ish is not None and osh is not None:
            if ish == osh:
                if _is_activation(m, device):
                    return dict(morph=False, is_activation=True, is_unknown=False,
                                why="aktivacija (elementwise) — census hook tocka")
                return dict(morph=False, is_activation=False, is_unknown=False,
                            why="prolaz/pooling bez promjene sirine — nije census tocka")
            return dict(morph=False, is_activation=False, is_unknown=False,
                        why="topolosko — mijenja oblik")
        return dict(morph=False, is_activation=False, is_unknown=False,
                    why="task mehanika — nije tensor->tensor")

    return dict(morph=False, is_activation=False, is_unknown=True,
                why=f"neprepoznato ({fqn(m)}, params={nparams})")


def classify(model, adapter, device, reg=None):
    reg = reg if reg is not None else load_register()
    shapes = _shapes(model, adapter, device)
    out = {}
    for name, m in model.named_modules():
        if list(m.children()):
            continue
        out[name] = classify_leaf(name, m, shapes, reg, device)
    return out


def capabilities_by_type(model, adapter, device, cls):
    import compress as C

    by_type = {}
    for name, m in model.named_modules():
        if list(m.children()):
            continue
        by_type.setdefault(fqn(m), []).append((name, m))

    caps = {}
    for ft, items in by_type.items():
        nparams = sum(p.numel() for p in items[0][1].parameters(recurse=False))
        cap = {"status": "verified", "trainable": nparams > 0, "frozen": False,
               "prunable": False, "growable": False}
        cand = [n for n, _ in items if cls.get(n, {}).get("morph")][:MAX_REPRESENTATIVES]
        for n in cand:
            if not cap["prunable"]:
                try:
                    _, n_rem, _, _, _ = C._apply_prune_plan(copy.deepcopy(model), adapter, device, {n: [0]})
                    cap["prunable"] = n_rem > 0
                except BaseException:
                    pass
            if not cap["growable"]:
                try:
                    cap["growable"] = C._try_grow_layer(model, adapter, device, n, 1) is not None
                except BaseException:
                    pass
            if cap["prunable"] and cap["growable"]:
                break
        caps[ft] = cap
    return caps


def merge_register(caps, path=REGISTER_PATH):
    reg = load_register(path)
    types = reg.setdefault("types", {})
    reg.setdefault("defaults", {"unlisted": "rules_decide"})
    added, updated, skipped = [], [], []
    for ft, cap in caps.items():
        cur = types.get(ft)
        if cur is None:
            types[ft] = dict(cap)
            added.append(ft)
        elif cur.get("status") == "hazard":
            skipped.append(ft)
        else:
            for k in ("prunable", "growable", "trainable", "frozen"):
                cur[k] = bool(cur.get(k, False)) or bool(cap[k])
            cur["status"] = "verified"
            updated.append(ft)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return added, updated, skipped


def run(spec, tag, device):
    import analysis as A

    print(f"\n{'=' * 88}\n### {tag}\n{'=' * 88}")
    model = A.load_any(spec, device)
    adapter = probe_adapter(model, device)
    if adapter is None:
        print("  [probe] nijedna kombinacija poziva/velicine nije prosla — preskacem")
        return

    cls = classify(model, adapter, device)
    n_morph = sum(1 for v in cls.values() if v["morph"])
    n_act = sum(1 for v in cls.values() if v["is_activation"])
    n_unk = sum(1 for v in cls.values() if v["is_unknown"])
    print(f"  leafova={len(cls)}  morph={n_morph}  aktivacija={n_act}  unknown={n_unk}")

    groups = {}
    for name, v in cls.items():
        groups.setdefault((v["morph"], v["why"]), []).append(name)
    print(f"\n  {'morph':>6}  {'n':>4}  why")
    for (mo, why), names in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0][1])):
        print(f"  {str(mo):>6}  {len(names):>4}  {why}")
        print(f"          npr: {', '.join(names[:2])}")

    caps = capabilities_by_type(model, adapter, device, cls)
    print(f"\n  --- sposobnosti po tipu (stvarni pokusaj kroz pipeline) ---")
    for ft, c in sorted(caps.items()):
        print(f"    {ft:52s} prune={str(c['prunable']):5s} grow={str(c['growable']):5s} "
              f"train={str(c['trainable']):5s} frozen={str(c['frozen']):5s}")

    a, u, s = merge_register(caps)
    print(f"\n  [registar] dodano={len(a)} azurirano={len(u)} preskoceno(hazard)={len(s)}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run("/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt", "yolo26n", dev)
    run("fasterrcnn", "fasterrcnn", dev)
    print(f"\nregistar: {REGISTER_PATH}")
