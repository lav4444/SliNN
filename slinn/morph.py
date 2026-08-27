"""slinn/morph.py — dokazane prune/grow/dead mehanike (preseljeno iz morphology/compress.py).

TASK-AGNOSTICNO: operira nad (model, adapter), ne zna nista o tasku ni obitelji modela.
Preseljeno BEZ IZMJENA (semanticki neutralno) -- svaka izmjena ide u zaseban korak nakon pariteta.
Sadrzi: coupled tp-group cost, GradMax grow, dead/near-dead rez, align faktore,
i depthwise guard u _widen_* (trial + forward-validate + commit/rollback).
"""

import copy
import math

import torch
import torch.nn as nn

import settings as config               # ALIGN_M / ALIGN_BETA / ALIGN_POW citaju se DINAMICKI (worker ih override-a po kvantizaciji)
import introspect as A        # bivsi morphology.analysis; ovdje samo genericki dio
from settings import PHASE2_COST_FLOPS_W, PHASE2_GROW_DOM, PHASE2_GROW_MAX_LAYERS


GB = 1024 ** 3


def gpu_status():
    """Stanje GPU-a za semafor: postoji li + koliko VRAM-a je zauzeto. ok = zauzeto < 50% ukupnog."""
    if not torch.cuda.is_available():
        return {"available": False, "ok": False, "msg": "nema CUDA GPU-a (radit ce na CPU, sporo)"}
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return {"available": True, "name": torch.cuda.get_device_name(0), "total_gb": total / GB,
            "used_gb": used / GB, "free_gb": free / GB, "used_pct": used / total * 100.0,
            "ok": used < total * 0.5}


def _forward_ok(model, adapter, device):
    """Provjeri da model jos radi (rez koji slomi npr. depthwise groups -> forward pukne)."""
    was = model.training
    try:
        model.eval()
        with torch.no_grad():
            adapter.forward(model, [torch.rand(3, 320, 320, device=device)])
        return True
    except BaseException:
        return False
    finally:
        model.train(was)


def remove_dead_neardead(model, adapter, device, loader, struct, census_max=None, eps=1e-6, weak=0.01):
    """FAZA 1, korak 1: one-shot tp rez SVIH dead+near-dead izlaznih kanala (samo prunabilni slojevi; BEZ growa).
    Svaki rez se radi na TRIAL kopiji i validira forwardom; commita se SAMO ako model i dalje radi (tp zna slomiti
    depthwise groups -> takav rez se odbaci). Vrati (model, n_removed_kanala, n_layers_touched)."""
    import torch_pruning as tp
    for p in model.parameters():
        p.requires_grad_(True)                            # tp tracer treba grad-graf
    act, _ = A.activation_stats(model, adapter, device, loader, census_max, eps, weak)
    pconv = getattr(tp, "prune_conv_out_channels", None) or tp.function.prune_conv_out_channels
    plin = getattr(tp, "prune_linear_out_channels", None) or tp.function.prune_linear_out_channels
    info = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(model)}
    # plan iz ORIGINALNOG censusa (idx u vlastite out-kanale ostaju valjani dok sloj nije rezan)
    plan = []
    for leaf, st in act.items():
        if (struct is not None and not struct.get(leaf)) or leaf not in info:
            continue                                      # samo prunabilni slojevi
        m = info[leaf][0]
        if getattr(m, "groups", 1) > 1:                   # depthwise: ne rezi kao ROOT (tp slomi groups);
            continue                                      # propagira se kroz regularni producer
        idx = sorted(set(st["dead_idx"]) | set(st["near_idx"]))
        if idx and len(idx) >= st["C"]:
            idx = idx[:st["C"] - 1]                        # ostavi bar 1 kanal
        if idx:
            plan.append((leaf, idx))
    cur = model
    n_rem = n_lay = 0
    n_bad = 0
    bad = set()                                           # slojevi koje dead-rez NIJE mogao maknuti (forward/tp) -> frozen za align_best
    for leaf, idx in plan:
        trial = copy.deepcopy(cur)                        # sandbox: rezi pa validiraj, commit tek ako radi
        try:
            info_now = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(trial)}
            if leaf not in info_now:
                continue                                  # sloj nestao (propagiran prijasnjim rezom)
            m, dim = info_now[leaf]
            fn = pconv if dim >= 3 else plin
            DG = tp.DependencyGraph().build_dependency(trial, example_inputs=adapter.tp_example(device))
            g = DG.get_pruning_group(m, fn, idxs=idx)
            if not DG.check_pruning_group(g):
                bad.add(leaf); continue                   # tp sam kaze da grupa nije cista -> preskoci
            g.prune()
            if not _forward_ok(trial, adapter, device):   # rez slomio model (depthwise groups i sl.) -> odbaci
                n_bad += 1; bad.add(leaf)
                continue
            cur = trial; n_rem += len(idx); n_lay += 1     # commit
        except BaseException:
            n_bad += 1; bad.add(leaf)                      # sloj koji tp ne moze cisto rezati -> preskoci
    if n_bad:
        print(f"[dead/near-dead] preskoceno {n_bad} slojeva (rez bi slomio model)")
    return cur, n_rem, n_lay, bad


def tap_coupled(model, adapter, device, taps, candidates):
    """Slojevi iz `candidates` koji DIJELE tp-grupu s nekim feature-KD tapom -> rez im mijenja
    SIRINU TAPA, pa feature-KD pukne (student 758 vs teacher 768).

    Zasto ovo postoji: `position.py` tap vec oznaci `morph=False`, ali to ga stiti samo kao KORIJEN
    reza. Sirina mu i dalje visi o spregnutim slojevima. Morphology je isto rjesavao preko
    `adapter.protect_prefixes` — RUCNIM popisom podstabala po obitelji (`backbone.fpn`, `rpn`,
    `roi_heads`). Ovdje je isto nacelo, ali IZMJERENO iz tp-grafa, pa vrijedi za bilo koji model.

    Kod yola je to 8 od 93 slojeva (tapovi u vratu, rezivo u okosnici — uglavnom odvojeno).
    Kod transformera je to SVE (rezidualni tok veze cijeli model na istu sirinu)."""
    import torch_pruning as tp
    if not taps:
        return set()
    mods = {nm: mm for nm, mm, _, _ in A.weighted_leaves(model)}
    for p in model.parameters():
        p.requires_grad_(True)                                # tp traceru treba grad-graf
    try:
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=adapter.tp_example(device))
    except BaseException:
        return set()                                          # bez grafa ne mozemo tvrditi nista
    pconv = getattr(tp, "prune_conv_out_channels", None) or tp.function.prune_conv_out_channels
    plin = getattr(tp, "prune_linear_out_channels", None) or tp.function.prune_linear_out_channels
    hit = set()
    for t in taps:
        tm = mods.get(t)
        if tm is None:
            continue
        try:
            g = DG.get_pruning_group(tm, pconv if tm.weight.dim() >= 3 else plin, idxs=[0])
        except BaseException:
            continue
        members = {id(getattr(getattr(d, "target", None), "module", None)) for d, _ in g}
        hit |= {nm for nm in candidates if nm in mods and id(mods[nm]) in members}
    return hit


def coupled_unit_cost(model, adapter, device, prunable):
    """STVARNI spregnuti trosak reza JEDNOG izlaznog kanala sloja, iz tp-grafa (NE samo vlastiti FLOPs):
    rez kanala makne i ULAZNE kanale potrosaca + spregnute siblinge, pa per-kanal cijena = ZBROJ po clanovima
    tp-grupe (out-clan: gflops/units, in-clan: gflops/in_ch). Racuna SAMO za 'prunable' (sigurno za tp;
    attention QKV bi tvrdo oborio proces -> off-limits se ovdje NE salju). Fallback na vlastiti trosak ako
    grupa zakaze. Vrati (flops_per[L] GFLOPs, params_per[L], units[L]) — sve u FLOPs (MACs se NE koristi)."""
    import torch_pruning as tp
    rec = A.layer_table(model, adapter, device)
    name2mod_all = {nm: mm for nm, mm, _, _ in A.weighted_leaves(model)}
    gpf, ppf, units, in_ch = {}, {}, {}, {}
    for r in rec:
        gpf[r["name"]] = r["gflops"]; ppf[r["name"]] = r["params"]
        units[r["name"]] = max(int(r["units"]), 1)
        ish = r.get("in")
        # BROJ ULAZNIH KANALA — iz SAMOG MODULA, ne iz izmjerenog oblika ulaza.
        # (BUG do 6.7: uzimalo se ish[1], sto vrijedi samo za NCHW conv i 2D linear. Kod sekvencijskog
        #  ulaza (B, S, C) to je DULJINA SEKVENCE, ne kanali -> DistilBERT ffn.lin1 dijelio s 64
        #  umjesto sa 768 (i lin2 s 64 umjesto 3072) -> cijena kanala precijenjena 49x/134x, pa je
        #  planer rezao ~20 kanala umjesto tisuca i isporucivao 0.07% umjesto ciljanih 1.5%.)
        mod_i = name2mod_all.get(r["name"])
        real_in = getattr(mod_i, "in_features", None) or getattr(mod_i, "in_channels", None)
        if not real_in and ish and len(ish) >= 2:
            real_in = ish[-1] if len(ish) == 3 else ish[1]     # (B,S,C) -> C; (B,C,H,W) -> C
        in_ch[r["name"]] = max(int(real_in or units[r["name"]]), 1)
    id2name = {id(m): nm for nm, m, _, _ in A.weighted_leaves(model)}
    name2mod = {nm: m for nm, m, _, _ in A.weighted_leaves(model)}
    for p in model.parameters():
        p.requires_grad_(True)                                 # tp tracer treba grad-graf (frozen ckpt -> 0 ovisnosti)
    try:
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=adapter.tp_example(device))
    except Exception:
        DG = None
    pconv = getattr(tp, "prune_conv_out_channels", None) or tp.function.prune_conv_out_channels
    plin = getattr(tp, "prune_linear_out_channels", None) or tp.function.prune_linear_out_channels
    flops_per, params_per = {}, {}
    for nm in prunable:
        m = name2mod.get(nm)
        if m is None or nm not in gpf:
            continue
        own_f, own_p = gpf[nm] / units[nm], ppf[nm] / units[nm]   # fallback = samo vlastiti per-kanal
        if DG is None:
            flops_per[nm], params_per[nm] = own_f, own_p; continue
        fn = pconv if m.weight.dim() >= 3 else plin
        try:
            g = DG.get_pruning_group(m, fn, idxs=[0])
        except BaseException:
            flops_per[nm], params_per[nm] = own_f, own_p; continue
        fl = pr = 0.0
        for item in g:
            dep = getattr(item, "dep", None) or (item[0] if isinstance(item, (tuple, list)) else None)
            mm = getattr(getattr(dep, "target", None), "module", None)
            mnm = id2name.get(id(mm))
            if mnm is None or mnm not in gpf:                  # ne-weighted clan (BN/norm) -> ~0 flops, preskoci
                continue
            hn = ""                                            # handler ime -> out-dim (/units) vs in-dim (/in_ch)
            h = getattr(dep, "handler", None)
            if h is not None:
                hn = getattr(h, "__name__", "") or getattr(getattr(h, "__func__", None), "__name__", "") or str(h)
            if "in_channel" in hn:
                denom = in_ch[mnm]
            elif "out_channel" in hn:
                denom = units[mnm]
            else:
                denom = units[mnm] if mm is m else in_ch[mnm]  # nepoznato: root=out, ostalo=potrosac(in)
            fl += gpf[mnm] / max(denom, 1); pr += ppf[mnm] / max(denom, 1)
        flops_per[nm] = fl if fl > 0 else own_f
        params_per[nm] = pr if pr > 0 else own_p
    return flops_per, params_per, units


def prune_costs(model, adapter, device, prunable, cost_flops_w=PHASE2_COST_FLOPS_W):
    """Blended REWARD-cost po sloju iz SPREGNUTE per-kanal cijene (coupled_unit_cost): cost[L] = w·(flops udio) +
    (1-w)·(params udio), udjeli normirani totalom preko prunabilnih (skale FLOPs/params nestaju).
    Vrati (cost, flops_per[GFLOPs], units). MACs se NE koristi."""
    flops_per, params_per, units = coupled_unit_cost(model, adapter, device, prunable)
    tot_f = sum(flops_per.values()) + 1e-12
    tot_p = sum(params_per.values()) + 1e-12
    cost = {nm: cost_flops_w * (flops_per[nm] / tot_f) + (1.0 - cost_flops_w) * (params_per[nm] / tot_p)
            for nm in flops_per}
    return cost, flops_per, units


def align_factors(width, m=None, beta=None, p=None):
    """Direkcioni HW-alignment faktori za poravnanje sirine na visekratnik m (int8/CHW32=32; fp16=8).
    m/beta/p se citaju DINAMICKI iz config (worker ih override-a po GUI kvantizaciji) ako nisu zadani.
      r=w%m (do donjeg xm, prune smjer), g=(-w)%m (do gornjeg xm, grow smjer)
      prune_factor = 1 - beta·(1-r/m)^p  in [1-beta,1]  -> <1 kad je sloj TIK IZNAD xm (jeftin tile-drop) -> reze se ranije
      grow_factor  = 1 + beta·(1-g/m)^p  in [1,1+beta]  -> >1 kad je sloj TIK ISPOD xm (jeftin tile-fill) -> raste ranije
    Poravnati (w%m==0) -> oba 1.0 (neutralno). beta<=0 -> iskljuceno. SOFT (ne forsira korake od m)."""
    m = config.ALIGN_M if m is None else m
    beta = config.ALIGN_BETA if beta is None else beta
    p = config.ALIGN_POW if p is None else p
    if beta <= 0 or m <= 1 or width <= 0:
        return 1.0, 1.0
    r = width % m
    g = (-width) % m
    ps = (1.0 - r / m) ** p if r else 0.0
    gs = (1.0 - g / m) ** p if g else 0.0
    return 1.0 - beta * ps, 1.0 + beta * gs


def grow_potential(gavg):
    """Simplified GradMax: σ po sloju = svdvals(signed grad matrice [O, in*k]). σ_i (padajuce) = korist i-tog
    NOVOG neurona (smjer najveceg neispunjenog gradijenta). Aproksimacija (grad sloja, ne sljedeceg) — pojednostavljeni
    GradMax. Vrati {name: tensor σ}. (Isti racun za KD-grad u kompresiji i GT-grad u Overviewu.)"""
    out = {}
    for nm, G in gavg.items():
        M = G.reshape(G.shape[0], -1).float()
        try:
            out[nm] = torch.linalg.svdvals(M)
        except Exception:
            out[nm] = torch.zeros(min(M.shape))
    return out


def _align_prune_score(imp_val, cost_nm, alive):
    """Dinamicni prune score 1 kanala: importance × align_PRUNE(TRENUTNA sirina) / spregnuti cost. Manji = prije rezan.
    align_PRUNE = align_factors(alive)[0] ∈ [1-beta,1]: <1 kad je sloj tik IZNAD ×M (snap dolje), =1.0 na ×M (r=0, neutralno)."""
    pf, _ = align_factors(alive)
    return imp_val * pf / (cost_nm + 1e-12)


def probe_min_gflops(model, adapter, device, prunable, banned=(), min_alive=None, min_alive_frac=None,
                     gflops_fn=None, on_progress=None):
    """STRUKTURNI MINIMUM GFLOPs-a: rezi SVAKI prunabilan sloj na `floor` sirinu i IZMJERI sto ostane.
    Radi na kopiji koja se na kraju baca. Vrati (g_min, banned) — `banned` su slojevi cija tp-grupa
    razbije forward, pa ih Faza 2 dobije vec ocisceni (nuspojava koja se isplati).

    ZASTO MJERENO, A NE PROCIJENJENO: analiticki zbroj doprinosa po sloju je tocno onaj racun koji je
    u 6.11 precjenjivao za 46% — spregnutost se ne da zbrojiti linearno (rez producenta vec suzava
    potrosaca). Prenizak `g_min` bi dao ljestvicu koju model nikad ne dosegne.

    Sloj po sloj, jer rez jednog spregnuto mijenja sirine drugih -> ciljna sirina se racuna na
    TRENUTNOM stanju kopije, ne na pocetnom. Depthwise se preskace (rezu se kroz producenta).
    REUSE `_apply_prune_plan` -> ista trial+forward+rollback disciplina kao u pravoj petlji."""
    import math as _math
    import settings as _CFG
    min_alive = _CFG.PHASE2_MIN_ALIVE if min_alive is None else min_alive
    min_alive_frac = _CFG.PHASE2_MIN_ALIVE_FRAC if min_alive_frac is None else min_alive_frac
    gflops_fn = gflops_fn or (lambda m: A.gflops_total(m, adapter, device))

    cur = copy.deepcopy(model)
    bad = set(banned)
    names = [nm for nm in prunable if nm not in bad]
    for k, nm in enumerate(sorted(names)):
        info_now = {n2: (m2, w2.dim()) for n2, m2, _, w2 in A.weighted_leaves(cur)}
        if nm not in info_now:                               # nestao (spregnuto propagiran rez)
            continue
        mod = info_now[nm][0]
        if getattr(mod, "groups", 1) > 1:                    # depthwise: ne kao root
            continue
        w = int(mod.weight.shape[0])
        floor = max(int(min_alive), int(_math.ceil(min_alive_frac * w)))
        if w <= floor:
            continue
        idx = list(range(floor, w))                          # zadrzi prvih `floor`; KOJI kanali je svejedno
        cur, _n_rem, _n_lay, _n_bad, b = _apply_prune_plan(cur, adapter, device, {nm: idx})
        bad |= b
        if on_progress is not None:
            on_progress(k + 1, len(names), nm)
    return float(gflops_fn(cur)), bad


def _select_prune_plan(struct, imp, cost, flops_per, units, info, target_gflops,
                       layer_cap, min_alive_frac, min_alive, exclude=()):
    """Biraj kanale dok SPREGNUTI oslobodeni GFLOPs ne dosegnu target_gflops, uz per-layer cap (max udio kanala/korak)
    i floor (min ostalih kanala). align_PRUNE je DINAMICAN preko min-heapa: nakon SVAKOG reza preracuna se na trenutnoj
    sirini (alive) tog sloja -> cim sloj snapne na ×M (r=0 -> faktor 1.0, izgubi popust) greedy prijede na druge slojeve
    (anti over-cut, ne razbija svjeze poravnanje; budzet netaknut - mijenja se KOJI kanali, ne KOLIKO). Importance je
    fiksan pa su kanali sloja rangirani jednom (argsort), a samo align faktor reagira na alive. flops_per[L] = STVARNI
    spregnuti GFLOPs po kanalu -> budzet pogada cilj. (prune_candidates = staticna lista za Overview viz, NE koristi se ovdje.)
    Vrati (plan, removed_gflops)."""
    import heapq
    order, ptr, alive, floor, quota, taken = {}, {}, {}, {}, {}, {}
    heap = []
    for nm, v in imp.items():                                  # eligible slojevi (isti filtri kao prune_candidates)
        if struct is not None and not struct.get(nm):
            continue
        if nm not in cost or nm not in info or nm in exclude:  # bez off-limits / banananih (forward-nesigurnih)
            continue
        if getattr(info[nm][0], "groups", 1) > 1:              # depthwise: rez kroz producera, ne kao root
            continue
        vf = v.float()
        C = units.get(nm, vf.numel())
        order[nm] = torch.argsort(vf).tolist()                 # kanali rastuce po importance (najnebitniji prvi); fiksno
        ptr[nm] = 0; alive[nm] = C; taken[nm] = 0
        floor[nm] = max(min_alive, math.ceil(min_alive_frac * C))
        quota[nm] = max(0, math.floor(layer_cap * C))
        if alive[nm] > floor[nm] and quota[nm] > 0:
            heapq.heappush(heap, (_align_prune_score(float(vf[order[nm][0]]), cost[nm], alive[nm]), nm))
    removed = 0.0; sel = {}
    while heap and removed < target_gflops:
        _s, nm = heapq.heappop(heap)
        ch = order[nm][ptr[nm]]
        sel.setdefault(nm, []).append(ch)
        alive[nm] -= 1; taken[nm] += 1; ptr[nm] += 1; removed += flops_per[nm]
        if alive[nm] > floor[nm] and taken[nm] < quota[nm] and ptr[nm] < len(order[nm]):   # jos rezivo -> re-score na NOVOJ alive i vrati u heap
            heapq.heappush(heap, (_align_prune_score(float(imp[nm][order[nm][ptr[nm]]]), cost[nm], alive[nm]), nm))
    return {nm: sorted(idx) for nm, idx in sel.items() if idx}, removed


def _apply_prune_plan(model, adapter, device, plan):
    """Materijaliziraj plan {leaf: idx} preko tp grupa, svaki leaf na TRIAL kopiji + forward-validacija +
    commit/rollback (isto kao remove_dead_neardead — tp zna slomiti npr. C2f split/concat). Vrati (model, n_rem, n_lay, n_bad, bad)
    gdje je `bad` skup leafova koji su pali (check/forward/iznimka) -> pozivatelj ih banana i bira sljedece kandidate."""
    import torch_pruning as tp
    for p in model.parameters():
        p.requires_grad_(True)
    pconv = getattr(tp, "prune_conv_out_channels", None) or tp.function.prune_conv_out_channels
    plin = getattr(tp, "prune_linear_out_channels", None) or tp.function.prune_linear_out_channels
    cur = model; n_rem = n_lay = n_bad = 0; bad = set()
    for leaf, idx in plan.items():
        trial = copy.deepcopy(cur)
        try:
            info_now = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(trial)}
            if leaf not in info_now:
                continue
            m, dim = info_now[leaf]
            C = m.weight.shape[0]
            idx2 = [j for j in idx if j < C]
            if not idx2 or len(idx2) >= C:
                continue
            fn = pconv if dim >= 3 else plin
            DG = tp.DependencyGraph().build_dependency(trial, example_inputs=adapter.tp_example(device))
            g = DG.get_pruning_group(m, fn, idxs=idx2)
            if not DG.check_pruning_group(g):
                bad.add(leaf); continue
            g.prune()
            if not _forward_ok(trial, adapter, device):
                n_bad += 1; bad.add(leaf); continue           # tp-rez slomio forward (split/concat) -> rollback + banaj
            cur = trial; n_rem += len(idx2); n_lay += 1
        except BaseException:
            n_bad += 1; bad.add(leaf)
    return cur, n_rem, n_lay, n_bad, bad


def _max_abs_diff(a, b):
    """Rekurzivni max|a-b| preko dict/list/tensor (isti oblik/struktura inace inf). Float -> max apsolutne razlike;
    INT (npr. roi_cidx klase) -> 0 ako identicni, inace inf (ne smije se kastati u float pa lazno javljati inf)."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape:
            return float("inf")
        if a.numel() == 0:
            return 0.0
        if not a.is_floating_point():                      # cijelobrojni (indeksi/klase): egzaktna jednakost
            return 0.0 if torch.equal(a, b.to(a.dtype)) else float("inf")
        return float((a.float() - b.float()).abs().max())
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return float("inf")
        return max((_max_abs_diff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return float("inf")
        return max((_max_abs_diff(x, y) for x, y in zip(a, b)), default=0.0)
    return 0.0


def _widen_out(mod, k, init_w):
    """Conv/Linear: dodaj k IZLAZNIH jedinica (weight=init_w [k, in, ...], bias=0). In-place (kao tp)."""
    w = mod.weight.data
    mod.weight = nn.Parameter(torch.cat([w, init_w.to(w.device, w.dtype)], dim=0))
    if mod.bias is not None:
        mod.bias = nn.Parameter(torch.cat([mod.bias.data, torch.zeros(k, device=w.device, dtype=w.dtype)]))
    if hasattr(mod, "out_channels"):
        mod.out_channels += k
    if hasattr(mod, "out_features"):
        mod.out_features += k


def _widen_bn(bn, k):
    """BatchNorm: dodaj k kanala (gamma=1, beta=0, running_mean=0, running_var=1) -> ne mijenja izlaz."""
    dev, dt = bn.weight.device, bn.weight.dtype
    bn.weight = nn.Parameter(torch.cat([bn.weight.data, torch.ones(k, device=dev, dtype=dt)]))
    bn.bias = nn.Parameter(torch.cat([bn.bias.data, torch.zeros(k, device=dev, dtype=dt)]))
    if bn.running_mean is not None:
        bn.running_mean = torch.cat([bn.running_mean, torch.zeros(k, device=dev, dtype=bn.running_mean.dtype)])
        bn.running_var = torch.cat([bn.running_var, torch.ones(k, device=dev, dtype=bn.running_var.dtype)])
    bn.num_features += k


def _widen_frozen_bn(fbn, k):
    """FrozenBatchNorm2d (torchvision): produzi BUFFERE za k (scale=1, bias=0, rm=0, rv=1). Identitet ionako
    cuvaju NULE kod potrosaca; ovdje samo da se oblici poklope i ne srusi forward."""
    for nm_, val in (("weight", 1.0), ("bias", 0.0), ("running_mean", 0.0), ("running_var", 1.0)):
        buf = getattr(fbn, nm_, None)
        if isinstance(buf, torch.Tensor):
            ext = torch.full((k,), val, device=buf.device, dtype=buf.dtype)
            fbn.register_buffer(nm_, torch.cat([buf, ext]))


def _widen_depthwise(dw, k):
    """Depthwise conv (groups==in==out): kanali rastu za k (groups, in, out svi +k; weight [C+k,1,kh,kw]).
    Novi depthwise filteri = 0 -> novi kanal prolazi ~0; identitet ionako cuva potrosac (project conv) NULAMA."""
    w = dw.weight.data                                     # [C, 1, kh, kw]
    dw.weight = nn.Parameter(torch.cat([w, w.new_zeros(k, *w.shape[1:])], dim=0))
    if dw.bias is not None:
        dw.bias = nn.Parameter(torch.cat([dw.bias.data, torch.zeros(k, device=w.device, dtype=w.dtype)]))
    dw.groups += k; dw.in_channels += k; dw.out_channels += k


def _is_depthwise(m):
    return isinstance(m, nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels == m.out_channels


def _insert_in_zeros(mod, k, positions):
    """Conv/Linear potrosac: umetni k NUL ulaznih kanala na svaku poziciju iz `positions`
    (silazno da raniji umetci ne pomaknu kasnije). In-place."""
    for pos in sorted(positions, reverse=True):
        w = mod.weight.data
        zshape = list(w.shape); zshape[1] = k
        z = torch.zeros(*zshape, device=w.device, dtype=w.dtype)
        mod.weight = nn.Parameter(torch.cat([w[:, :pos], z, w[:, pos:]], dim=1))
    tot = k * len(positions)
    if hasattr(mod, "in_channels"):
        mod.in_channels += tot
    if hasattr(mod, "in_features"):
        mod.in_features += tot


def _try_grow_layer(model, adapter, device, name, k, init_filters=None):
    """Trial function-preserving rast IZLAZA sloja `name` za k, RASTUCI CIJELU tp-grupu zajedno (coupled):
    svi izlazno-spregnuti conv/linear (L + residual/SE siblinzi) +k, sve BN/FrozenBN +k, DEPTHWISE (groups==in==out)
    +k (kanali+groups), svi potrosaci +k NUL-stupaca na ispravnom concat-offsetu. Commit SAMO ako forward ostane
    identican (diff<1e-3) i radi; inace None (rollback) — tp-internals fragilnost je samoispravljiva.
    init_filters: opcionalni init L-ovih novih filtera (GradMax clone iz _grow_decide); siblinzi kloniraju vlastite.
    NIJE @no_grad: tp.build_dependency treba autograd-graf za trace (kao remove_dead_neardead)."""
    import torch_pruning as tp
    sz = getattr(adapter, "imgsz", 320)
    ref_imgs = [torch.rand(3, sz, sz, device=device)]
    try:
        with torch.no_grad():
            ref_out = adapter.teacher_outputs(model, ref_imgs)
    except BaseException:
        return None
    trial = copy.deepcopy(model)
    leaves = {nm: (mm, w.dim()) for nm, mm, _, w in A.weighted_leaves(trial)}
    if name not in leaves:
        return None
    L, dim = leaves[name]
    if getattr(L, "groups", 1) > 1:                       # depthwise kao ROOT: rast ide kroz producera, ne ovdje
        return None
    old_L = L.weight.shape[0]                              # sirina bloka PRIJE rasta (offset potrosaca = o + old_L)
    try:
        for p in trial.parameters():
            p.requires_grad_(True)
        fn = (tp.function.prune_conv_out_channels if dim >= 3 else tp.function.prune_linear_out_channels)
        DG = tp.DependencyGraph().build_dependency(trial, example_inputs=adapter.tp_example(device))
        group = DG.get_pruning_group(L, fn, idxs=[0])      # idxs=[0] -> mapiranje L-ovog kanala 0 na cijelu spregnutu grupu
        out_mods, bns, fbns, dws, cons = [], [], [], [], {}
        for dep, idxs in group:
            tgt = getattr(getattr(dep, "target", None), "module", None)
            if tgt is None:
                continue
            hn = getattr(dep.handler, "__name__", type(dep.handler).__name__).lower()
            if isinstance(tgt, nn.modules.batchnorm._BatchNorm):
                bns.append(tgt)
            elif type(tgt).__name__ == "FrozenBatchNorm2d" or (hasattr(tgt, "running_mean") and hasattr(tgt, "weight")
                                                               and not isinstance(tgt, (nn.Conv2d, nn.Linear))):
                fbns.append(tgt)                           # frozen-BN: izlazni norm, siri se bufferima
            elif _is_depthwise(tgt):
                dws.append(tgt)                            # depthwise u lancu (mobilenet) -> kanali rastu (groups+in+out)
            elif isinstance(tgt, nn.Conv2d) and tgt.groups > 1:
                return None                                # ne-depthwise grouped conv -> nesigurno, odustani
            elif "in_channel" in hn or "in_feature" in hn:
                if isinstance(tgt, (nn.Conv2d, nn.Linear)):
                    cons.setdefault(tgt, []).append(int(min(idxs)))   # offset L-ovog bloka u ulazu potrosaca
            elif "out_channel" in hn or "out_feature" in hn:
                if isinstance(tgt, (nn.Conv2d, nn.Linear)):
                    out_mods.append(tgt)                   # L + izlazno-spregnuti siblinzi (residual/SE) -> svi rastu zajedno
        if L not in out_mods:
            out_mods.append(L)
        seen = set()
        for mod in out_mods:                               # svaki izlazno-spregnuti +k (L: dani init; ostali: clone vlastitih)
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            if mod is L and init_filters is not None:
                init = init_filters
            else:
                wabs = mod.weight.detach().abs().flatten(1).mean(1)
                order = torch.argsort(wabs, descending=True)
                idx = [int(order[i % len(order)]) for i in range(k)]
                cl = mod.weight.detach()[idx].clone()
                init = cl + torch.randn_like(cl) * 0.02 * cl.abs().mean().clamp(min=1e-6)
            _widen_out(mod, k, init)
        for dw in {id(d): d for d in dws}.values():
            _widen_depthwise(dw, k)
        for bn in {id(b): b for b in bns}.values():
            _widen_bn(bn, k)
        for fb in {id(f): f for f in fbns}.values():
            _widen_frozen_bn(fb, k)
        for cmod, offs in cons.items():
            _insert_in_zeros(cmod, k, [o + old_L for o in offs])   # novi kanali na KRAJ L-ovog bloka
        if not _forward_ok(trial, adapter, device):
            return None
        with torch.no_grad():
            after = adapter.teacher_outputs(trial, ref_imgs)
        if _max_abs_diff(ref_out, after) >= 1e-3:          # NIJE function-preserving (kriva kirurgija) -> rollback
            return None
        return trial
    except BaseException:
        return None


def _select_grow_plan(sigma, flops_per, units, struct, pool_gflops):
    """Odluci {sloj: k} za grow per-kanal preko MAX-HEAPA s DINAMICNIM align_GROW (simetrija prune _select_prune_plan).
    Svaki sloj nudi sljedeci kanal: score = σ[ptr]/coupled-flops × align_GROW(TRENUTNA sirina). Pop najjaceg; dodaj 1 ako
    σ[ptr] >= PHASE2_GROW_DOM×med (dominance na MARGINALNOM kanalu — σ pada kroz GradMax niz, isti prag kao za prvi) I ima
    budzeta I nije otvoren PHASE2_GROW_MAX_LAYERS-ti razliciti sloj; pa rekalkuliraj (σ[ptr+1] + nova sirina) i vrati u heap.
    align_GROW je BOOST koji NESTANE na ×M (g=0 -> faktor 1.0) -> sloj tad izgubi prednost i drugi preteknu (SOFT snap),
    ali dovoljno jak σ moze svejedno probiti ×M (overshoot kad signal vrijedi). k <= LAYER_CAP×sirina. NE radi operaciju
    (samo broji); materijalizacija je u _grow_decide. Vrati (plan, est_spent, med)."""
    import heapq, statistics
    cands = [nm for nm in sigma if (struct is None or struct.get(nm)) and flops_per.get(nm, 0.0) > 0
             and len(sigma[nm]) > 0 and float(sigma[nm][0]) > 0]
    if not cands:
        return {}, 0.0, 1e-12
    def _gscore(nm, w, p):                                             # score MARGINALNOG (p-tog) novog kanala na sirini w
        _, gf = align_factors(w)                                      # align_GROW (boost tik ispod ×M, 1.0 na ×M)
        return float(sigma[nm][p]) / flops_per[nm] * gf
    med = statistics.median(sorted(_gscore(nm, units.get(nm, 1), 0) for nm in cands)) + 1e-12   # tipicna prilika (kao stari grow_candidates med)
    thr = PHASE2_GROW_DOM * med                                       # dominance prag NA SCOREU (ne na sirovom σ) — isti kao prije
    ptr, width, cap, heap = {}, {}, {}, []
    for nm in cands:
        ptr[nm] = 0; width[nm] = units.get(nm, 1)
        cap[nm] = config.ALIGN_M                            # fiksni per-event cap = velicina pocice (dinamicki; g<M -> ne blokira snap na sljedeci ×M)
        s = _gscore(nm, width[nm], 0)
        if cap[nm] >= 1 and s >= thr:                                 # samo dominantni (score >= thr) ulaze u heap
            heapq.heappush(heap, (-s, nm))
    remaining = pool_gflops; plan = {}; grown = set()
    while heap:
        neg, nm = heapq.heappop(heap)
        if -neg < thr:                                                # najjaci preostali ispod praga -> svi su -> stani
            break
        if nm not in grown and len(grown) >= PHASE2_GROW_MAX_LAYERS:  # ne otvaramo (MAX+1)-ti razliciti sloj
            continue
        if flops_per[nm] > remaining:                                 # najbolji DOPUSTENI nepriustiv -> stani (ne rastemo slabijeg)
            break
        plan[nm] = plan.get(nm, 0) + 1; grown.add(nm)
        ptr[nm] += 1; width[nm] += 1; remaining -= flops_per[nm]      # +1 kanal -> napreduj σ niz + sirina (align se mijenja)
        if plan[nm] < cap[nm] and ptr[nm] < len(sigma[nm]):
            s2 = _gscore(nm, width[nm], ptr[nm])                      # re-score na NOVOJ sirini + σ[ptr]
            if s2 >= thr:                                            # vrati u heap samo ako jos dominira (σ pao / align pao na ×M -> ispada)
                heapq.heappush(heap, (-s2, nm))
    return plan, pool_gflops - remaining, med


def _grow_decide(student, adapter, device, struct, gavg, flops_per, units, info, pool_gflops):
    """Simplified-GradMax grow, MULTI-SLOJ per-kanal (<= PHASE2_GROW_MAX_LAYERS po koraku). σ = svdvals(signed KD-grad)
    za RANGIRANJE svih growabilnih (jeftino). _select_grow_plan odluci {sloj: k} (per-kanal heap + dinamicki align_GROW +
    dominance na σ[ptr]). Pa MATERIJALIZIRA: PUNI SVD samo za ODABRANE slojeve (<=3) -> init = gornji desni sing. vektori
    (GradMax smjer) skalirani na sr. normu filtera; potrosaci=0 (function-preserving osigurava _try_grow_layer). Sekvencijalno;
    _fresh preskace sloj kojem je prethodni rast promijenio ulaznu dim. Vrati (grown_or_None, [info-po-sloju], total_spent)."""
    if pool_gflops <= 0:
        return None, [], 0.0
    def _fresh(nm, info_now):                              # gavg (prije reza) mora odgovarati TRENUTNOJ arhitekturi tog sloja:
        if nm not in info_now:                            # potrosac prethodnog reza/rasta promijenio dim -> stari gavg krivog oblika
            return False
        W = info_now[nm][0].weight
        return (gavg[nm].shape[0] == W.shape[0]
                and gavg[nm].reshape(gavg[nm].shape[0], -1).shape[1] == W.reshape(W.shape[0], -1).shape[1])
    growable = {nm: gavg[nm] for nm in gavg                 # σ SAMO za growabilne, svjezeg signala (off-limits/attention preskoceni)
                if (struct is None or struct.get(nm)) and flops_per.get(nm, 0.0) > 0 and _fresh(nm, info)}
    sigma = grow_potential(growable)                       # svdvals (jeftino) za rangiranje
    plan, _est, med = _select_grow_plan(sigma, flops_per, units, struct, pool_gflops)
    if not plan:
        return None, [], 0.0
    infos = []; total_spent = 0.0
    for nm, k in plan.items():
        info_now = {nm2: (mod, w.dim()) for nm2, mod, _, w in A.weighted_leaves(student)}
        if not _fresh(nm, info_now):                       # prethodni rast promijenio arhitekturu ovog (potrosac) -> preskoci (svjez iduci korak)
            continue
        C = info_now[nm][0].weight.shape[0]
        G = gavg[nm].reshape(gavg[nm].shape[0], -1).float()           # [O, in*k]
        try:
            _, _, Vh = torch.linalg.svd(G, full_matrices=False)       # PUNI svd SAMO za odabrane -> desni sing. vektori (GradMax smjerovi)
        except Exception:
            continue
        kk = min(k, Vh.shape[0])
        if kk < 1:
            continue
        W = info_now[nm][0].weight.detach()
        fnorm = float(W.reshape(W.shape[0], -1).norm(dim=1).mean().clamp(min=1e-6))   # skala ~ sr. norma postojeceg filtera (CPU skalar)
        init = (Vh[:kk] * fnorm).reshape((kk,) + tuple(W.shape[1:])).to(W.device, W.dtype)   # Vh je CPU -> skaliraj pa na uredjaj
        g_before = A.gflops_total(student, adapter, device)
        grown = _try_grow_layer(student, adapter, device, nm, kk, init)
        if grown is None:
            continue                                       # numericki nije function-preserving growable -> preskoci
        spent = A.gflops_total(grown, adapter, device) - g_before
        student = grown; total_spent += spent
        _, gf0 = align_factors(units.get(nm, C))                      # dom = pocetni score sloja / med (score-based, kao stari log)
        infos.append({"layer": nm, "k": kk, "had": C, "dom": (float(sigma[nm][0]) / flops_per[nm] * gf0) / med})
    if not infos:
        return None, [], 0.0
    return student, infos, total_spent


# =========================== HW-PORAVNANJE (align score) ===========================
# Preseljeno iz morphology/compress.py (6.4 korak 7) BEZ IZMJENA. Mjeri iskoristivost
# ×M plocice po sloju -> koliko je model spreman za int8/fp16 inference.


def model_align_score(model, m=None):
    """Model-level HW-poravnanje = prosjecna ISKORISTIVOST pocice po sloju: u = w / (m·ceil(w/m)).
    1.0 = svi (weighted) slojevi su visekratnik m (savrseno poravnato); manje = padding-otpad pocice.
    ASIMETRICNO (w=33 -> 0.52, PUNO gore od w=31 -> 0.97, jer 33 otvara polupraznu 2. pocicu). Preko SVIH
    conv/linear slojeva (stvarna HW iskoristivost, ne samo prunabilni). m iz config dinamicki. Azurira se svaki morph korak."""
    m = config.ALIGN_M if m is None else m
    if m <= 1:
        return 1.0
    us = []
    for _, _, _, w in A.weighted_leaves(model):
        c = int(w.shape[0])
        if c > 0:
            us.append(c / (m * math.ceil(c / m)))
    return sum(us) / len(us) if us else 1.0


def best_align_score(model, struct, frozen, m=None):
    """Teoretski MAX dostizni align_score: slojeve koje SMIJEMO dirati (touchable = struct True I nije u `frozen`)
    racunamo kao savrseno poravnate (u=1.0), a UNTOUCHABLE (struct False = off-limits ILI u `frozen` = forward-odbijeni)
    ostaju na trenutnoj sirini -> fiksni u = c/(m·ceil(c/m)). Prosjek po SVIM weighted leafovima (kao model_align_score).
    Spusta se kako `frozen` raste (vise odbijenih -> nizi strop). Coupling touchable↔untouchable se ZANEMARUJE (blagi over-estimate)."""
    m = config.ALIGN_M if m is None else m
    if m <= 1:
        return 1.0
    us = []
    for nm, _, _, w in A.weighted_leaves(model):
        c = int(w.shape[0])
        if c <= 0:
            continue
        touchable = (struct is None or struct.get(nm)) and nm not in frozen
        us.append(1.0 if touchable else c / (m * math.ceil(c / m)))
    return sum(us) / len(us) if us else 1.0
