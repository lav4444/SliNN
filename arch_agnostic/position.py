"""
position.py — POZICIJSKI prolaz: suzava `morph` iz classify.py po tome GDJE sloj stoji.

classify.py odgovara na "je li sloj strukturno rezljiv" (tip + instanca, stabilno).
Ovdje se odgovara na "smijemo li ga rezati bas tu" — i to se mora PRERACUNAVATI svaki morph
korak, jer se ovisnosti i sirine mijenjaju.

Cetiri filtera:
  terminalnost  SIGURNOST — sloj bez weighted potrosaca izbacuje konacan broj klasa/okvira
  attention     SIGURNOST — tp.get_pruning_group na multi-head QKV-u TVRDO obori proces
  se_1x1        politika  — conv nad globalno poolanim deskriptorom: cost~0 vara score
  tap           POLITIKA S KOTACICEM — feature-KD tap mora zadrzati sirinu (kanal-po-kanal MSE)

Tap se NE izvodi iz topologije. Testirano i oboreno dva puta (rezolucijske granice, povratno
zatvorenje glave): neck i glava su topoloski nerazlucivi, oboje "hrane samo izlaz". Granica je
modelarska odluka autora. Zato: politika s kotacicem + mjerljiva cijena.
"""

import sys

import torch

_MORPH = "/home/tomi/code/dipl/morphology"
if _MORPH not in sys.path:
    sys.path.insert(0, _MORPH)

import analysis as A                                        # noqa: E402
from classify import _shapes, classify, probe_adapter       # noqa: E402

# =========================== KOTACIC =========================== #
TAP_POLICY = "depth_fraction"    # "protect_prefixes" | "depth_fraction" | "none"
TAP_COUNT = 3                    # koliko tapova (svaki kosta tocno 1 sloj prunabilnosti)
TAP_DEPTH = 0.80                 # od koje relativne dubine ih traziti (0..1, forward redoslijed)


# =========================== GRAF =========================== #
def attention_leaves(model):
    """Duck-typing: modul s .num_heads ili imenom tipa koje sadrzi attn/attention -> svi leafovi ispod."""
    pref = [nm + "." for nm, mod in model.named_modules()
            if hasattr(mod, "num_heads") or "attention" in type(mod).__name__.lower()
            or "attn" in type(mod).__name__.lower()]
    return {n for n, _, _, _ in A.weighted_leaves(model) if any(n.startswith(p) for p in pref)}


def consumer_map(model, adapter, device, skip=()):
    """{L: set(weighted potrosaca)} iz tp grafa. `skip` se NE trasira (attention obori proces)."""
    import torch_pruning as tp
    for p in model.parameters():
        p.requires_grad_(True)
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=adapter.tp_example(device))
    id2n = {id(m): n for n, m, _, _ in A.weighted_leaves(model)}
    pconv = tp.function.prune_conv_out_channels
    plin = tp.function.prune_linear_out_channels
    out = {}
    for n, m, _, w in A.weighted_leaves(model):
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


# =========================== TAPOVI =========================== #
def pick_taps(model, adapter, shapes, cls, policy=None, count=None, depth=None):
    """Vrati (set tapova, opis politike). Svaki tap kosta TOCNO 1 sloj prunabilnosti."""
    policy = policy or TAP_POLICY
    count = TAP_COUNT if count is None else count
    depth = TAP_DEPTH if depth is None else depth

    if policy == "none":
        return set(), "none (samo logit/output KD — sve morphabilno)"

    if policy == "protect_prefixes":
        gp = getattr(adapter, "protect_prefixes", None)
        if not gp:
            return set(), "protect_prefixes (adapter ih ne nudi -> prazno)"
        pref = list(gp(model))
        taps = {n for n in cls if any(n.startswith(p) for p in pref)}
        return taps, f"protect_prefixes {pref}"

    # depth_fraction: najdublji morph-kandidati na RAZLICITIM izlaznim rezolucijama
    order = [n for n in shapes if n in cls]                  # hookovi su pucali u forward redoslijedu
    cut = int(depth * len(order))
    taps, seen = set(), set()
    for n in reversed(order[cut:]):                          # od najdubljeg prema plicem
        if not cls[n]["morph"]:
            continue
        osh = shapes[n][1]
        res = tuple(osh[2:]) if osh and len(osh) == 4 else ("flat",)
        if res in seen:
            continue
        seen.add(res)
        taps.add(n)
        if len(taps) >= count:
            break
    return taps, f"depth_fraction(depth={depth}, count={count}) -> {len(taps)} na razlicitim rezolucijama"


# =========================== PROLAZ =========================== #
def positional(model, adapter, device, cls=None, shapes=None, **tap_kw):
    """Suzi morph iz classify.py. Vrati (pos, meta) gdje pos[name] = {morph, why}."""
    shapes = shapes if shapes is not None else _shapes(model, adapter, device)
    cls = cls if cls is not None else classify(model, adapter, device)

    att = attention_leaves(model)
    cons = consumer_map(model, adapter, device, skip=att)
    term = terminal_leaves(cons, skip=att)
    se = se_leaves(shapes) & set(cls)
    taps, tap_desc = pick_taps(model, adapter, shapes, cls, **tap_kw)

    pos = {}
    for n, v in cls.items():
        if not v["morph"]:
            pos[n] = {"morph": False, "why": v["why"]}
        elif n in att:
            pos[n] = {"morph": False, "why": "attention — tp se rusi na QKV-u"}
        elif n in term:
            pos[n] = {"morph": False, "why": "terminalno — izbacuje konacan broj klasa/okvira"}
        elif n in taps:
            pos[n] = {"morph": False, "why": "feature-KD tap — sirina mora ostati poravnata"}
        elif n in se:
            pos[n] = {"morph": False, "why": "SE/1x1 deskriptor — cost~0 vara score"}
        else:
            pos[n] = {"morph": True, "why": v["why"]}

    meta = {"attention": len(att), "terminal": len(term), "se": len(se), "taps": sorted(taps),
            "tap_policy": tap_desc,
            "morph_classify": sum(1 for v in cls.values() if v["morph"]),
            "morph_final": sum(1 for v in pos.values() if v["morph"])}
    return pos, meta


# =========================== RUN =========================== #
def run(spec, tag, device):
    print(f"\n{'=' * 88}\n### {tag}\n{'=' * 88}")
    model = A.load_any(spec, device)
    probe = probe_adapter(model, device, verbose=False)
    shapes = _shapes(model, probe, device)
    cls = classify(model, probe, device)

    for policy in ("protect_prefixes", "depth_fraction", "none"):
        ad = A.pick_adapter(model) if policy == "protect_prefixes" else probe   # rucne liste su na pravom adapteru
        pos, meta = positional(model, probe if policy != "protect_prefixes" else ad, device,
                               cls=cls, shapes=shapes, policy=policy)
        drop = meta["morph_classify"] - meta["morph_final"]
        print(f"\n  [{policy}]")
        print(f"    {meta['tap_policy']}")
        print(f"    attention={meta['attention']}  terminal={meta['terminal']}  se={meta['se']}  "
              f"tapova={len(meta['taps'])}")
        print(f"    morph {meta['morph_classify']} -> {meta['morph_final']}  (-{drop})")
        if meta["taps"]:
            print(f"    tapovi: {meta['taps'][:6]}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run("/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt", "yolo26n", dev)
    run("fasterrcnn", "fasterrcnn", dev)
