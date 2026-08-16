"""
position.py — POZICIJSKI prolaz: suzava `morph` iz classify.py po tome GDJE sloj stoji.

classify.py odgovara na "je li sloj strukturno rezljiv" (tip + instanca, stabilno).
Ovdje se odgovara na "smijemo li ga rezati bas tu" — i to se mora PRERACUNAVATI svaki morph
korak, jer se ovisnosti i sirine mijenjaju.

Cetiri filtera:
  terminalnost  SIGURNOST — sloj bez weighted potrosaca izbacuje konacan broj klasa/okvira
  attention     SIGURNOST — tp.get_pruning_group na multi-head QKV-u TVRDO obori proces
  se_1x1        politika  — conv nad globalno poolanim deskriptorom: cost~0 vara score
  tap           GRAF-SIDRO  — feature-KD tap mora zadrzati sirinu (kanal-po-kanal MSE)

Tapovi se SIDRE na izlazne glave: od svake glave penjemo se uz graf ovisnosti i stajemo na PRVOM
morphabilnom sloju koji je "izlaz bloka/neck-a" — tj. ciji izlaz ide na ne-samo-izlaznog potrosaca
(fan-out) ILI mijenja rezoluciju prema ne-terminalnom potrosacu (kraj stagea, conv->fc). BFS bira
NAJBLIZU granicu glavi. Time se preskacu interni slojevi glave (redundantni s logit-KD-om) i cls/box
grananje (oba potrosaca terminalna). Ravan lanac bez granice -> fallback = najblizi morphabilan iznad
glave. Bez ijednog magicnog broja; cijena je mjerljiva: svaki tap kosta tocno 1 sloj prunabilnosti.
"""

import sys

import torch

_MORPH = "/home/tomi/code/dipl/morphology"
if _MORPH not in sys.path:
    sys.path.insert(0, _MORPH)

import analysis as A                                        # noqa: E402
from classify import _shapes, classify, probe_adapter, weighted_leaves  # noqa: E402

# =========================== TAP (feature-KD) =========================== #
# Nema magicnog broja: tap = prvi "block-boundary" morphabilan sloj iznad glave (v. docstring).
TAP_CAP_ABS = 5          # guard: max apsolutno tapova prije nego proglasimo tap-set nepouzdanim
TAP_CAP_FRAC = 0.10      # guard: ...ili max udio morphabilnih slojeva (sto je vece)
# Prekoraci li se guard -> pure logit KD (feature-KD off), da ne ostanemo bez slojeva za rezanje.


# =========================== GRAF =========================== #
def attention_leaves(model):
    """Duck-typing: modul s .num_heads ili imenom tipa koje sadrzi attn/attention -> svi leafovi ispod."""
    pref = [nm + "." for nm, mod in model.named_modules()
            if hasattr(mod, "num_heads") or "attention" in type(mod).__name__.lower()
            or "attn" in type(mod).__name__.lower()]
    return {n for n, _, _, _ in weighted_leaves(model) if any(n.startswith(p) for p in pref)}


def consumer_map(model, adapter, device, skip=()):
    """{L: set(weighted potrosaca)} iz tp grafa. `skip` se NE trasira (attention obori proces)."""
    import torch_pruning as tp
    for p in model.parameters():
        p.requires_grad_(True)
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=adapter.tp_example(device))
    id2n = {id(m): n for n, m, _, _ in weighted_leaves(model)}
    pconv = tp.function.prune_conv_out_channels
    plin = tp.function.prune_linear_out_channels
    out = {}
    for n, m, _, w in weighted_leaves(model):
        cons = set()
        if n not in skip:
            try:
                g = DG.get_pruning_group(m, pconv if w.dim() >= 3 else plin, idxs=[0])
                for dep, _ in g:
                    tgt = getattr(getattr(dep, "target", None), "module", None)
                    hn = getattr(dep.handler, "__name__", type(dep.handler).__name__).lower()
                    nm = id2n.get(id(tgt))
                    if nm and nm != n and ("in_channel" in hn or "in_feature" in hn):
                        cons.add(nm)
            except BaseException:
                pass
        out[n] = cons
    return out


def terminal_leaves(cons, skip=()):
    """Bez weighted potrosaca -> izlazna glava. `skip` (attention) se ne racuna: njih smo namjerno
    ostavili bez potrosaca, pa bi lazno ispali terminalni."""
    return {n for n, c in cons.items() if not c and n not in skip}


def se_leaves(shapes):
    """Conv/linear ciji je ULAZ prostorno 1x1 -> radi nad globalno poolanim deskriptorom."""
    out = set()
    for n, (ish, _) in shapes.items():
        if ish is not None and len(ish) >= 3 and all(int(d) == 1 for d in ish[2:]):
            out.add(n)
    return out


# =========================== TAPOVI (block-boundary) =========================== #
def producer_map(cons):
    """Obrni consumer_map: {L: set(tezinskih PRODUCENATA)}."""
    prod = {n: set() for n in cons}
    for n, cs in cons.items():
        for c in cs:
            prod.setdefault(c, set()).add(n)
    return prod


def _res_of(n, shapes):
    """Izlazna rezolucija sloja: (H,W) za 4D, inace ('flat',)."""
    osh = shapes.get(n, (None, None))[1]
    return tuple(osh[2:]) if osh and len(osh) == 4 else ("flat",)


def _is_boundary(n, cons, term, shapes):
    """Je li izlaz sloja `n` kraj bloka/neck-a: ide na ne-samo-izlaznog potrosaca (fan-out) ILI
    mijenja rezoluciju prema ne-terminalnom potrosacu (kraj stagea / conv->fc).
    Grananje SAMO u izlazne glave (cls/box) se NE racuna."""
    c = cons.get(n, set())
    nonterm = [x for x in c if x not in term]
    if not nonterm:
        return False                                          # hrani samo glave -> nije neck
    rn = _res_of(n, shapes)
    if any(_res_of(x, shapes) != rn for x in nonterm):
        return True                                           # promjena rezolucije (stage kraj, conv->fc)
    return len(c) > 1                                         # fan-out uz bar 1 ne-terminalni potrosac


def _tap_for_head(head, prod, cons, term, cls, shapes):
    """BFS uz graf od glave. Vrati (boundary_taps, nearest_morphable):
      boundary_taps  NAJBLIZI morphabilni block-boundary slojevi (moze biti prazno ako put
                     udari u slijepi zid prije granice, npr. neispracena grana glave).
      nearest        najblizi morphabilan iznad glave (kandidat za NET-fallback, ne per-glava)."""
    import collections
    morph_ok = lambda n: cls.get(n, {}).get("morph", False)   # noqa: E731
    seen, taps, nearest = set(), set(), None
    dq = collections.deque(prod.get(head, ()))
    while dq:
        n = dq.popleft()
        if n in seen:
            continue
        seen.add(n)
        if morph_ok(n):
            if nearest is None:
                nearest = n
            if _is_boundary(n, cons, term, shapes):
                taps.add(n)
                continue                                      # nasli izlaz bloka -> ne penji dalje ovim putem
        dq.extend(prod.get(n, ()))
    return taps, nearest


def pick_taps(cons, term, cls, shapes):
    """Za svaku glavu nadji izlaz bloka (block-boundary) iznad nje. Vrati (taps, opis, kd_mode) gdje
    taps[name] = {"head": glava, "res": izlazna rezolucija}. Zajednicki neck izlaz na koji pada vise
    glava se prirodno spaja u JEDAN tap (kljuc = ime sloja). Ako tapova ima previse (> cap) -> TRIM na cap
    RAVNOMJERNO PO DUBINI (zadrzi feature-KD s multi-skalnom pokrivenoscu, bez magicnog "prvih N"), a ne
    odbacivanje. Ako CIJELA mreza nema nijednu granicu (ravan MLP), net-fallback = jedan najblizi morphabilan.

    (Namjerno BEZ dedupa po rezoluciji: probe koristi najmanju radnu velicinu pa razlicite skale
    znaju koincidirati u rezoluciji -> lazno spajanje. Multi-skalu cuvamo, blow-up hvata trim.)"""
    prod = producer_map(cons)
    taps, fallbacks = {}, []
    for head in term:
        bt, nearest = _tap_for_head(head, prod, cons, term, cls, shapes)
        for t in bt:
            taps.setdefault(t, {"head": head, "res": _res_of(t, shapes)})
        if nearest:
            fallbacks.append(nearest)
    if not taps and fallbacks:
        t = fallbacks[0]
        taps[t] = {"head": "?", "res": _res_of(t, shapes), "fallback": True}

    n_morph = sum(1 for v in cls.values() if v.get("morph"))
    cap = max(TAP_CAP_ABS, int(TAP_CAP_FRAC * n_morph))
    if len(taps) > cap:                                       # TRIM na cap, ravnomjerno po dubini (ne odbacuj)
        order = {n: i for i, n in enumerate(cls)}             # dubinski redoslijed = redoslijed u modelu
        names = sorted(taps, key=lambda n: order.get(n, 0))
        keep_idx = {round(i * (len(names) - 1) / (cap - 1)) for i in range(cap)} if cap > 1 else {len(names) // 2}
        taps = {n: taps[n] for j, n in enumerate(names) if j in keep_idx}
        return taps, f"trim: {len(names)} -> {len(taps)} tapova ravnomjerno po dubini (cap {cap})", "feature+logit"

    mode = "feature+logit" if taps else "logit"
    return taps, f"block-boundary -> {len(taps)} tapova (cap {cap})", mode


# =========================== PROLAZ =========================== #
def positional(model, adapter, device, cls=None, shapes=None):
    """Suzi morph iz classify.py. Vrati (pos, meta) gdje pos[name] = {morph, why}."""
    shapes = shapes if shapes is not None else _shapes(model, adapter, device)
    cls = cls if cls is not None else classify(model, adapter, device)

    att = attention_leaves(model)
    cons = consumer_map(model, adapter, device, skip=att)
    term = terminal_leaves(cons, skip=att)
    se = se_leaves(shapes) & set(cls)
    taps, tap_desc, kd_mode = pick_taps(cons, term, cls, shapes)

    pos = {}
    for n, v in cls.items():
        if not v["morph"]:
            pos[n] = {"morph": False, "why": v["why"]}
        elif n in att:
            pos[n] = {"morph": False, "why": "attention — tp se rusi na QKV-u"}
        elif n in term:
            pos[n] = {"morph": False, "why": "terminalno — izbacuje konacan broj klasa/okvira"}
        elif n in taps:
            pos[n] = {"morph": False, "why": f"feature-KD tap (glava '{taps[n]['head']}') — sirina mora ostati poravnata"}
        elif n in se:
            pos[n] = {"morph": False, "why": "SE/1x1 deskriptor — cost~0 vara score"}
        else:
            pos[n] = {"morph": True, "why": v["why"]}

    meta = {"attention": len(att), "terminal": len(term), "se": len(se), "taps": sorted(taps),
            "tap_desc": tap_desc, "kd_mode": kd_mode,
            "morph_classify": sum(1 for v in cls.values() if v["morph"]),
            "morph_final": sum(1 for v in pos.values() if v["morph"])}
    return pos, meta


# =========================== RUN =========================== #
def run(spec, tag, device, code_dirs=None):
    print(f"\n{'=' * 88}\n### {tag}\n{'=' * 88}")
    model = A.load_any(spec, device, code_dirs)
    probe = probe_adapter(model, device, verbose=False)
    shapes = _shapes(model, probe, device)
    cls = classify(model, probe, device)

    pos, meta = positional(model, probe, device, cls=cls, shapes=shapes)
    drop = meta["morph_classify"] - meta["morph_final"]
    print(f"\n  KD: {meta['kd_mode']}  |  {meta['tap_desc']}")
    print(f"    attention={meta['attention']}  terminal={meta['terminal']}  se={meta['se']}  "
          f"tapova={len(meta['taps'])}")
    print(f"    morph {meta['morph_classify']} -> {meta['morph_final']}  (-{drop})")
    for t in meta["taps"]:
        print(f"      tap  {t}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run("/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt", "yolo26n", dev)
    run("fasterrcnn", "fasterrcnn", dev)
    run("/home/tomi/code/dipl/pareto_sweep/schoolcnn_pareto_final.pt", "SchoolCNN", dev,
        code_dirs=["/home/tomi/code/dipl/pruning/critereum_experiment2"])
