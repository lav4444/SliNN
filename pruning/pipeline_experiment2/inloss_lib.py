"""
inloss_lib.py — jezgra za PRUNING UGRADEN U LOSS (train-time, risk-reward) na
production detektoru (fasterrcnn_mobilenet_v3_large_320_fpn).

Ideja (risk-reward u lossu):
    L = det_loss(student na GT) + alpha * KD_feature(FPN MSE, teacher=nepruned self)
        + lambda * SUM_c  cost_c * penalty(gate_c)
  - gate_c: ucljiva 'rucica' po izlaznom kanalu. Za conv slojeve gate je IZA BN-a
    (pa maskiran kanal -> aktivacija(0)=0, prava nula; bez BN train/eval mismatcha).
    Za fc6/fc7 gate je na izlazu Linear-a (relu(0)=0, vec cisto).
  - cost_c = 0.5 * FLOPs_c/FLOPs_tot + 0.5 * params_c/params_tot   (REWARD, 50/50)
  - RISK (Taylor) ulazi prirodno: Taylor vaznost gatea = gate_c * dL/dgate_c.
  - Gradualno (cubic) hard-maskiramo po RISK/REWARD (score/cost) dok efektivni
    params ne padnu na KEEP_PARAM_FRAC; na kraju tp materijalizira pravi rez.

L1 mod:  gate realni parametar (init 1); penal = cost*|gate|; rank = |gate|.
L0 mod:  hard-concrete gate (Louizos); penal = cost*P(otvoren); rank = P(otvoren).

Napomena: gate se primjenjuje IZA BN-a, a fold_into_weights ufolda gate u afine
parametre tog modula (BN gamma/beta, ili conv/linear weight/bias).
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Hard-concrete (L0)
# --------------------------------------------------------------------------- #
HC_BETA = 2.0 / 3.0
HC_GAMMA = -0.1
HC_ZETA = 1.1


def _hardconcrete_sample(loga):
    u = torch.rand_like(loga).clamp_(1e-6, 1 - 1e-6)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + loga) / HC_BETA)
    s_bar = s * (HC_ZETA - HC_GAMMA) + HC_GAMMA
    return s_bar.clamp(0, 1)


def _hardconcrete_eval(loga):
    s = torch.sigmoid(loga)
    s_bar = s * (HC_ZETA - HC_GAMMA) + HC_GAMMA
    return s_bar.clamp(0, 1)


def _hardconcrete_open_prob(loga):
    return torch.sigmoid(loga - HC_BETA * math.log(-HC_GAMMA / HC_ZETA))


# --------------------------------------------------------------------------- #
# GateController: gate IZA BN-a (conv) / na izlazu Linear-a (fc)
# --------------------------------------------------------------------------- #
class GateController(nn.Module):
    def __init__(self, model, mode="l1"):
        super().__init__()
        assert mode in ("l1", "l0")
        self.mode = mode
        self.apply_mod = OrderedDict()     # key -> modul gdje hook mnozi izlaz (BN ili Linear)
        self.cost_mod = {}                 # key -> conv/linear za racun cost-a
        self.dims = {}                     # key -> 'spatial'(4D) | 'flat'(2D)
        self.gate = nn.ParameterDict()
        self._handles = []

        # mapiraj conv -> BN koji ga slijedi (unutar Conv2dNormActivation / Sequential)
        conv_to_bn = {}
        for parent in model.backbone.body.modules():
            ch = list(parent.children())
            for a, b in zip(ch, ch[1:]):
                if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                    conv_to_bn[a] = b

        # gate-amo SAMO conv slojeve koji imaju BN iza sebe (glavni filteri) + fc6/fc7.
        # (SE conv-ovi nemaju BN; preskacemo ih kao gate jedinice — tp ih svejedno
        #  moze rezati pri materijalizaciji.)
        for name, m in model.backbone.body.named_modules():
            if isinstance(m, nn.Conv2d) and not name.startswith("0") and m in conv_to_bn:
                key = f"body__{name.replace('.', '__')}"
                self._register(key, conv_to_bn[m], m, "spatial", m.out_channels)
        bh = model.roi_heads.box_head
        self._register("fc6", bh.fc6, bh.fc6, "flat", bh.fc6.out_features)
        self._register("fc7", bh.fc7, bh.fc7, "flat", bh.fc7.out_features)

        for key in self.apply_mod:
            C = self.gate[key].numel()
            self.register_buffer(f"mask__{key}", torch.ones(C))
            self.register_buffer(f"cost__{key}", torch.zeros(C))
            self.register_buffer(f"pcount__{key}", torch.zeros(C))
            self.register_buffer(f"macs__{key}", torch.zeros(C))

    def _register(self, key, apply_mod, cost_mod, dims, C):
        self.apply_mod[key] = apply_mod
        self.cost_mod[key] = cost_mod
        self.dims[key] = dims
        if self.mode == "l1":
            self.gate[key] = nn.Parameter(torch.ones(C))
        else:
            self.gate[key] = nn.Parameter(torch.full((C,), 2.5))

    # ---- multiplikator po kanalu ----
    def _factor(self, key, training):
        mask = getattr(self, f"mask__{key}")
        if self.mode == "l1":
            g = self.gate[key]
        else:
            g = _hardconcrete_sample(self.gate[key]) if training else _hardconcrete_eval(self.gate[key])
        return mask * g

    def attach(self):
        self.remove()
        for key, module in self.apply_mod.items():
            dims = self.dims[key]

            def hook(m, inp, out, key=key, dims=dims):
                f = self._factor(key, self.training)
                if dims == "spatial":
                    return out * f.view(1, -1, 1, 1)
                return out * f.view(1, -1)
            self._handles.append(module.register_forward_hook(hook))
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # ---- score za rangiranje (manji = prvi za rez) ----
    @torch.no_grad()
    def scores(self):
        out = {}
        for key in self.apply_mod:
            if self.mode == "l1":
                out[key] = self.gate[key].detach().abs()
            else:
                out[key] = _hardconcrete_open_prob(self.gate[key].detach())
        return out

    # ---- penal u lossu (samo zivi kanali) ----
    def penalty(self):
        total = 0.0
        for key in self.apply_mod:
            mask = getattr(self, f"mask__{key}")
            cost = getattr(self, f"cost__{key}")
            if self.mode == "l1":
                p = self.gate[key].abs()
            else:
                p = _hardconcrete_open_prob(self.gate[key])
            total = total + (mask * cost * p).sum()
        return total

    @torch.no_grad()
    def effective_params(self, total_params):
        return total_params - self._current_removed_params()

    @torch.no_grad()
    def alive_fraction(self):
        a = t = 0
        for key in self.apply_mod:
            mask = getattr(self, f"mask__{key}")
            a += int((mask > 0.5).sum()); t += mask.numel()
        return a, t

    # ---- gradualno maskiranje do ciljanog uklonjenog params (RISK/REWARD) ----
    @torch.no_grad()
    def prune_to_removed_params(self, target_removed, min_alive_frac=0.10, min_alive=8):
        """Maskiraj ZIVE kanale po RISK/REWARD = score/cost (uzlazno) dok uklonjeni
        params ne dosegnu target_removed. Per-layer floor sprjecava gutanje sloja."""
        sc = self.scores()
        cand = []
        floor, alive_now = {}, {}
        for key in self.apply_mod:
            mask = getattr(self, f"mask__{key}")
            cost = getattr(self, f"cost__{key}")
            pcount = getattr(self, f"pcount__{key}")
            s = sc[key]
            C = mask.numel()
            floor[key] = max(min_alive, int(math.ceil(min_alive_frac * C)))
            alive_now[key] = int((mask > 0.5).sum())
            for i in (mask > 0.5).nonzero(as_tuple=True)[0].tolist():
                rr = float(s[i]) / (float(cost[i]) + 1e-12)
                cand.append((rr, key, i, float(pcount[i])))
        cand.sort(key=lambda x: x[0])
        removed = self._current_removed_params()
        for rr, key, i, pc in cand:
            if removed >= target_removed:
                break
            if alive_now[key] <= floor[key]:
                continue
            getattr(self, f"mask__{key}")[i] = 0.0
            alive_now[key] -= 1
            removed += pc
            if self.mode == "l1":
                self.gate[key].data[i] = 0.0
        return removed

    @torch.no_grad()
    def _current_removed(self, buf="pcount"):
        removed = 0.0
        for key in self.apply_mod:
            mask = getattr(self, f"mask__{key}")
            c = getattr(self, f"{buf}__{key}")
            removed += float((c * (mask < 0.5)).sum())
        return removed

    @torch.no_grad()
    def prune_by_external_to_removed(self, score, target_removed, min_alive_frac=0.10, min_alive=8,
                                     metric="params"):
        """Kao prune_to_removed_params, ali rangira po EKSTERNOM score-u (npr. gradijent
        |dL/dgate|) umjesto self.scores(). RISK/REWARD = score/cost (uzlazno).
        metric='params' -> budzet u pcount; metric='macs' -> budzet u FLOPs (macs po kanalu).
        Maskira (mask=0, gate=0 za l1) najmanje vazne ZIVE kanale do target_removed."""
        buf = "pcount" if metric == "params" else "macs"
        cand = []
        floor, alive_now = {}, {}
        for key in self.apply_mod:
            mask = getattr(self, f"mask__{key}")
            cost = getattr(self, f"cost__{key}")
            cnt = getattr(self, f"{buf}__{key}")
            s = score[key]
            C = mask.numel()
            floor[key] = max(min_alive, int(math.ceil(min_alive_frac * C)))
            alive_now[key] = int((mask > 0.5).sum())
            for i in (mask > 0.5).nonzero(as_tuple=True)[0].tolist():
                rr = float(s[i]) / (float(cost[i]) + 1e-12)
                cand.append((rr, key, i, float(cnt[i])))
        cand.sort(key=lambda x: x[0])
        removed = self._current_removed(buf)
        for rr, key, i, c in cand:
            if removed >= target_removed:
                break
            if alive_now[key] <= floor[key]:
                continue
            getattr(self, f"mask__{key}")[i] = 0.0
            alive_now[key] -= 1
            removed += c
            if self.mode == "l1":
                self.gate[key].data[i] = 0.0
        return removed

    @torch.no_grad()
    def reset_survivor_gates(self):
        """Gate ZIVIH kanala -> 1 (da fold_into_weights ne skalira preživjele; odbacuje L1 drift)."""
        for key in self.apply_mod:
            mask = getattr(self, f"mask__{key}")
            if self.mode == "l1":
                self.gate[key].data[mask > 0.5] = 1.0

    @torch.no_grad()
    def _current_removed_params(self):
        removed = 0.0
        for key in self.apply_mod:
            mask = getattr(self, f"mask__{key}")
            pcount = getattr(self, f"pcount__{key}")
            removed += float((pcount * (mask < 0.5)).sum())
        return removed

    @torch.no_grad()
    def fold_into_weights(self):
        """Ufolda (mask*gate, eval) u afine parametre modula na kojem visi gate
        (BN gamma/beta za conv put, ili conv/linear weight/bias). Maskirani kanal ->
        izlaz 0 i nakon micanja hookova. tp ga (preko 'prod' importancea) rezi prvi."""
        for key, module in self.apply_mod.items():
            f = self._factor(key, training=False)
            w = module.weight
            if w.dim() == 4:        # conv
                module.weight.data.mul_(f.view(-1, 1, 1, 1))
            elif w.dim() == 2:      # linear
                module.weight.data.mul_(f.view(-1, 1))
            else:                   # BN gamma (1D)
                module.weight.data.mul_(f)
            if module.bias is not None:
                module.bias.data.mul_(f)


# --------------------------------------------------------------------------- #
# Cost (reward): params + FLOPs po kanalu, normalizirano (50/50)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_costs(gc: GateController, model, device, imgsz=320):
    out_hw = {}
    handles = []
    for key, conv in gc.cost_mod.items():
        if isinstance(conv, nn.Conv2d):
            handles.append(conv.register_forward_hook(
                lambda m, i, o, key=key: out_hw.__setitem__(key, (o.shape[-2], o.shape[-1]))))
    model.eval()
    model([torch.randn(3, imgsz, imgsz, device=device)])
    for h in handles:
        h.remove()

    pcount_all, macs_all, raw = [], [], {}
    for key, mod in gc.cost_mod.items():
        C = gc.gate[key].numel()
        if isinstance(mod, nn.Conv2d):
            cin = mod.in_channels // mod.groups
            k = mod.kernel_size[0] * mod.kernel_size[1]
            params_per = cin * k + (1 if mod.bias is not None else 0)
            Ho, Wo = out_hw.get(key, (1, 1))
            macs_per = cin * k * Ho * Wo
        else:  # Linear
            inf = mod.in_features
            params_per = inf + (1 if mod.bias is not None else 0)
            macs_per = inf
        pcount = torch.full((C,), float(params_per))
        macs = torch.full((C,), float(macs_per))
        raw[key] = (pcount, macs)
        pcount_all.append(pcount); macs_all.append(macs)

    tot_p = float(torch.cat(pcount_all).sum()) + 1e-9
    tot_m = float(torch.cat(macs_all).sum()) + 1e-9
    for key in gc.cost_mod:
        pcount, macs = raw[key]
        cost = 0.5 * (macs / tot_m) + 0.5 * (pcount / tot_p)
        getattr(gc, f"cost__{key}").copy_(cost.to(device))
        getattr(gc, f"pcount__{key}").copy_(pcount.to(device))
        getattr(gc, f"macs__{key}").copy_(macs.to(device))
    return {"total_params_per_filter": tot_p, "total_macs_per_filter": tot_m}


# --------------------------------------------------------------------------- #
# Feature-KD: MSE na FPN izlazima (teacher=nepruned self)
# --------------------------------------------------------------------------- #
class FPNFeatureKD:
    def __init__(self, teacher, student):
        self.t_feat = None
        self.s_feat = None
        self._h = []
        self._h.append(teacher.backbone.fpn.register_forward_hook(
            lambda m, i, o: setattr(self, "t_feat", o)))
        self._h.append(student.backbone.fpn.register_forward_hook(
            lambda m, i, o: setattr(self, "s_feat", o)))

    def loss(self):
        return self.loss_with(self.t_feat)

    def loss_with(self, t_feat):
        """MSE student FPN (hook) vs zadani teacher FPN dict (live ili iz cachea)."""
        if t_feat is None or self.s_feat is None:
            return torch.tensor(0.0)
        keys = self.s_feat.keys() if hasattr(self.s_feat, "keys") else range(len(self.s_feat))
        tot, n = 0.0, 0
        for k in keys:
            s = self.s_feat[k]
            t = t_feat[k].to(s.dtype)
            tot = tot + F.mse_loss(s, t.detach())
            n += 1
        return tot / max(n, 1)

    def remove(self):
        for h in self._h:
            h.remove()
        self._h = []


# --------------------------------------------------------------------------- #
# Cubic sparsity schedule (Zhu & Gupta)
# --------------------------------------------------------------------------- #
def cubic_removed_target(epoch, prune_start, prune_end, final_removed):
    if epoch < prune_start:
        return 0.0
    if epoch >= prune_end:
        return final_removed
    t = (epoch - prune_start) / max(1, (prune_end - prune_start))
    return final_removed * (1 - (1 - t) ** 3)


def linear_removed_target(epoch, prune_start, prune_end, final_removed):
    """Linearni raspored: jednaki rezovi (bez front-loada) -> svaki rez podjednako velik."""
    if epoch < prune_start:
        return 0.0
    if epoch >= prune_end:
        return final_removed
    return final_removed * (epoch - prune_start) / max(1, (prune_end - prune_start))
