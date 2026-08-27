"""
classify.py — klasifikacija leaf slojeva, model-agnosticno.

Izlaz po sloju (ugovor):
    morph          bool    smije li biti prune/grow ROOT (strukturna sigurnost)
    is_activation  bool    census hook tocka (zamjena za hardkodiranu ACT_TYPES listu)
    is_unknown     bool    alarm pokrivenosti
    why            string  dijagnostika za covjeka

NE radi poziciju (protect / terminalnost / SE-policy) — to je zaseban prolaz koji se mora
preracunavati svaki morph korak. Ovdje su samo tip i instanca, koji su stabilni.

Uz to gradi LAYER_REGISTER.json ADITIVNO: po TIPU sloja upisuje sposobnosti
(prunable/growable/trainable/frozen) dobivene STVARNIM pokusajem operacije kroz
morphology pipeline (tp rez + _try_grow_layer, s trial/rollback), a ne citanjem koda.
Rucno pisani "hazard" unosi se NIKAD ne prepisuju.
"""

import copy
import json
import os
import sys

import torch
import torch.nn as nn


REGISTER_PATH = "/home/tomi/code/dipl/slinn/LAYER_REGISTER.json"

# ljestvica ulaznih velicina: NAJMANJA koja radi (validacija se vrti nakon svakog reza)
SIZE_LADDER = (96, 128, 224, 320, 640)          # 2D slika (stranica)
SIZE_LADDER_1D = (1024, 4096, 8000, 16000, 48000)  # 1D sekvenca/val (duljina) — audio je duzi od slike
SEQ_LADDER_TOK = (8, 16, 32, 64)                # token: broj tokena (<= 512 zbog position_embeddings)
MAX_REPRESENTATIVES = 3          # koliko instanci po tipu probati prije nego zakljucimo da tip ne moze


# =========================== PROBE ADAPTER =========================== #
class ProbeAdapter:
    """Minimalni adapter izveden PROBINGOM — bez ijednog retka model-specificnog koda.

    Pokriva tocno ono sto klasifikacija treba: forward, forward_example, tp_example, teacher_outputs.
    (predict / gt_loss / kd_loss NISU potrebni — oni sluze evaluaciji i KD treningu.)
    """

    kind = "probe"

    def __init__(self, call, size, in_ch, mode):
        self._call = call            # "list" (lista CHW) · "batch" (NCHW tenzor) · "kwargs" (input_ids=…)
        self.imgsz = size            # slika: stranica · seq: duljina · token: broj tokena
        self._in_ch = in_ch          # slika/seq: kanali · vektor: F · token: vocab
        self._mode = mode            # "image" | "seq" | "vector" | "token"
        self.flexible = None         # radi li model na vise velicina (postavlja probe)

    # --- kanonski ulaz (jedan uzorak, bez batch dim) --- #
    def _one(self, device):
        if self._mode == "vector":
            return torch.rand(self._in_ch, device=device)                      # [F]
        if self._mode == "seq":
            return torch.rand(self._in_ch, self.imgsz, device=device)          # [C, L]
        if self._mode == "token":
            return torch.randint(0, self._in_ch, (self.imgsz,), dtype=torch.long, device=device)  # [L] long
        return torch.rand(self._in_ch, self.imgsz, self.imgsz, device=device)  # [C, H, W]

    def forward_example(self, device):
        """Ulaz u formatu koji prima self.forward (za _forward_ok validaciju)."""
        return [self._one(device)]

    def tp_example(self, device):
        """SIROVI argument za model(...) — sto tp DependencyGraph treba."""
        x = self._one(device).unsqueeze(0)
        return [x[0]] if self._call == "list" else x       # kwargs/batch -> [B, …] tenzor (input_ids za token)

    def forward(self, model, imgs):
        if self._call == "kwargs":
            return _unwrap(model(input_ids=torch.stack([im for im in imgs])))  # token: keyword forward
        if self._call == "list":
            return _unwrap(model(list(imgs)))
        return _unwrap(model(torch.stack([im for im in imgs])))

    @torch.no_grad()
    def teacher_outputs(self, model, imgs):
        """Referenca za function-preserving provjeru: sirovi izlaz, detached na cpu."""
        return _detach(self.forward(model, imgs))


def _unwrap(o):
    """HF ModelOutput (SequenceClassifierOutput...) -> tensor/tuple, da _finite/_detach vide tenzore.
    Ostali izlazi (tensor, lista, dict) prolaze netaknuti."""
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
    """Nema NaN/Inf. PRAZAN tenzor je uredu — detektor na sumu legitimno vraca 0 okvira."""
    if isinstance(o, torch.Tensor):
        return bool(torch.isfinite(o).all().item()) if o.numel() else True
    if isinstance(o, dict):
        return all(_finite(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return all(_finite(v) for v in o)
    return True


def _input_spec(model):
    """(in_ch, mode) iz PRVOG relevantnog sloja. mode odredjuje kako probe gradi ulaz:
      "image"  Conv2d (weight dim-4)  -> [C, H, W] float
      "seq"    Conv1d (weight dim-3)  -> [C, L] float
      "token"  Embedding             -> [L] long indeksi u [0, vocab); in_ch = vocab (num_embeddings)
      "vector" Linear (weight dim-2)  -> [F] float
    Prioritet: conv (prostorni ulaz) > Embedding (token ulaz) > Linear (vektor). Embedding se provjerava
    PRIJE generickog dim-2 jer mu je weight isto dim-2 [vocab, dim] pa bi inace ispao kao vektor sirine dim."""
    for _, m in model.named_modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() in (3, 4):
            return int(w.shape[1]) * int(getattr(m, "groups", 1)), ("seq" if w.dim() == 3 else "image")
    for _, m in model.named_modules():
        if isinstance(m, nn.Embedding):
            return int(m.num_embeddings), "token"          # in_ch = vocab (shape[0]), NE dim (shape[1])
    for _, m in model.named_modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            return int(w.shape[1]), "vector"
    return 3, "image"


def probe_adapter(model, device, verbose=True):
    """Odredi ulazni ugovor JEDNOM: konvencija poziva + NAJVECA radna velicina + je li fleksibilan.

    Pretraga ide SILAZNO (640 -> nize): uzima se NAJVECA velicina koja daje valjani forward. Razlog: KD-trening
    i mjerenje (mAP/mIoU) trebaju REPREZENTATIVNU rezoluciju (detekcija je osjetljiva na male objekte); najmanja
    radna (96) je dovoljna za STRUKTURU ali presitna za trening. Pravilo je isto za SVE modele (agnosticno) —
    nema per-model velicine. Fiksno-ulazni modeli (npr. FC glava) svejedno padnu na svoju jedinu radnu velicinu.

    Kaskada je OVDJE, a ne u _forward_ok — inace se ne moze razlikovati 'rez je slomio model'
    od 'pogodio sam krivu velicinu', i pretrazivalo bi se nakon svakog reza.
    """
    in_ch, mode = _input_spec(model)
    ladder = {"vector": (in_ch,), "seq": SIZE_LADDER_1D, "image": SIZE_LADDER, "token": SEQ_LADDER_TOK}[mode]
    calls = ("kwargs",) if mode == "token" else ("list", "batch")
    model.eval()
    ok = []
    for call in calls:
        for sz in sorted(ladder, reverse=True):    # SILAZNO: 640 -> nize; prva (=najveca) radna pobjeduje
            a = ProbeAdapter(call, sz, in_ch, mode)
            try:
                with torch.no_grad():
                    out = a.forward(model, a.forward_example(device))
                if _finite(out):
                    ok.append((call, sz))
            except BaseException:
                pass
        if ok:
            break                                  # prva konvencija koja radi pobjeduje
    if not ok:
        return None
    call, sz = ok[0]                               # NAJVECA radna velicina (ok je silazno po velicini)
    a = ProbeAdapter(call, sz, in_ch, mode)
    a.flexible = len({s for _, s in ok}) > 1
    if verbose:
        unit = "vocab" if mode == "token" else "ch"
        print(f"[probe] poziv={call} · ulaz={in_ch}{unit} @ {sz} ({mode}) · "
              f"fleksibilan={a.flexible} · radne velicine={sorted({s for _, s in ok})}")
    return a


# =========================== REGISTAR =========================== #
def load_register(path=REGISTER_PATH):
    if not os.path.exists(path):
        return {"defaults": {"unlisted": "rules_decide"}, "types": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fqn(m):
    t = type(m)
    return f"{t.__module__}.{t.__name__}"


def weighted_leaves(model):
    """[(name, module, typename, weight)] za leaf s conv/linear tezinom — kao
    morphology.analysis.weighted_leaves, ALI ukljucuje i Conv1d (weight dim-3).
    (morphology filtrira dim in (2,4); pozicijskom prolazu 1D convovi moraju biti vidljivi,
    inace je citav 1D lanac nevidljiv grafu ovisnosti -> nema tapova/terminala.)"""
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


# =========================== PRAVILA =========================== #
def _shapes(model, adapter, device):
    """Jedan forward s hookovima -> {name: (ishape|None, oshape|None)} za leafove KOJI SU SE IZVRSILI."""
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
    """Elementwise nelinearnost? Dva uvjeta, oba mjerena:
      1) mijenja vrijednosti  -> Identity/Dropout(eval) otpadaju
      2) promjena JEDNOG ulaznog elementa utjece SAMO na taj element -> MaxPool/pooling otpadaju
    Ulaz se KLONIRA pri svakom pozivu: inplace aktivacije (nn.SiLU(inplace=True)) inace vrate
    isti objekt pa bi se usporedjivao tenzor sam sa sobom."""
    try:
        x = torch.randn(1, 4, 8, 8, device=device)
        with torch.no_grad():
            y0 = m(x.clone())
            if not isinstance(y0, torch.Tensor) or y0.shape != x.shape:
                return False
            if torch.allclose(y0, x):
                return False                                  # identitet -> nije aktivacija
            x2 = x.clone()
            x2[0, 0, 3, 3] += 5.0
            y1 = m(x2.clone())
        moved = int(((y0 - y1).abs() > 1e-6).sum().item())     # koliko izlaznih elemenata je reagiralo
        return moved <= 1                                      # 1 = elementwise, 0 = zasicenje, >1 = mijesa susjede
    except BaseException:
        return False


def classify_leaf(name, m, shapes, reg, device):
    """4-poljni ugovor za jedan leaf. Registar samo VETIRA (hazard) i daje out_axis; pravila odlucuju."""
    ent = _reg_entry(reg, m)
    if ent.get("status") == "hazard":
        # hazard je POZNAT i obradjen (zasticen morph=False) — nije coverage gap, pa NIJE unknown.
        return dict(morph=False, is_activation=False, is_unknown=False,
                    why=f"hazard: {ent.get('reason', 'poznato ogranicenje')} — zasticeno, poznat slucaj")

    w = getattr(m, "weight", None)
    has_w = isinstance(w, torch.Tensor)
    nparams = sum(p.numel() for p in m.parameters(recurse=False))
    fired = name in shapes
    ish, osh = shapes.get(name, (None, None))

    # --- nosi tezinu s izlaznom sirinom? --- #
    if has_w and w.dim() >= 2:
        axis = int(ent.get("out_axis", 0))
        n_out = int(w.shape[axis])
        declared = getattr(m, "out_channels", None)
        if declared is None:
            declared = getattr(m, "out_features", None)

        if osh is not None:                                  # DOKAZ 1: izmjereno forwardom
            measured = osh[-1] if w.dim() == 2 else osh[1]    # Linear: kanal=zadnja os; conv 1D/2D: os 1
            if measured != n_out:
                return dict(morph=False, is_activation=False, is_unknown=True,
                            why=f"shape[{axis}]={n_out} != izmjerena izlazna sirina {measured}")
            src = "forward"
        elif declared is not None:                           # DOKAZ 2: vlastita deklaracija
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

    # --- normalizacija: parametri po kanalu, sirina je posljedica --- #
    if nparams and (hasattr(m, "running_mean") or (has_w and w.dim() == 1)):
        return dict(morph=False, is_activation=False, is_unknown=False,
                    why="norm — sirina vezana uz producenta")

    # --- bez parametara --- #
    if nparams == 0:
        if not fired:
            # 0 param + izvan compute-grafa: dokazivo nebitan za strukturnu kompresiju (ne moze biti
            # prune-root — nema tezine; ni census-tocka ni topoloski op — nije u grafu). probe je vec
            # validirao da pun forward daje konacan izlaz, pa "nije se izvrsio" = stvarno neaktivan
            # (npr. SDPA-inline Dropout, quant-stub Identity), a NE slomljen probe.
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


# =========================== SPOSOBNOSTI PO TIPU (iz PRAVOG pipelinea) =========================== #
def capabilities_by_type(model, adapter, device, cls):
    """Za svaki TIP: pokusaj STVARNU operaciju na do MAX_REPRESENTATIVES instanci i OR-aj ishod.

    prunable — tp rez 1 kanala preko compress._apply_prune_plan (trial + forward + rollback)
    growable — compress._try_grow_layer(k=1) (function-preserving provjera + rollback)
    trainable/frozen — ima li ucljive parametre; frozen je danas UVIJEK False (nema freeze liste)

    Tvrdnja je o TIPU ("pipeline zna ovo rezati"), ne o svakoj instanci — konkretan sloj svejedno
    moze otpasti zbog groups ili pozicije.
    """
    import morph as C

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
        # kandidati: instance koje su prosle strukturna pravila (inace garantirani promasaj)
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
    """ADITIVNO: novi tipovi se dodaju, postojeci OR-aju sposobnosti. 'hazard' unosi se NE DIRAJU."""
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
            skipped.append(ft)                        # rucno pisano — netaknuto
        else:
            for k in ("prunable", "growable", "trainable", "frozen"):
                cur[k] = bool(cur.get(k, False)) or bool(cap[k])
            cur["status"] = "verified"
            updated.append(ft)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return added, updated, skipped


# =========================== RUN =========================== #
def run(spec, tag, device):
    import introspect as A

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
