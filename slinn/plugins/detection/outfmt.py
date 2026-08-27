"""slinn/plugins/detection/outfmt.py — prepoznavanje FORMATA DETEKCIJSKOG IZLAZA mjerenjem.

SVRHA: model koji nikad nismo vidjeli mora i dalje proci. Umjesto da pitamo "tko je napisao ovaj
model", pitamo "kakvog je oblika ono sto vraca" — pa po tome biramo decode.

REDOSLIJED PREPOZNAVANJA (adapters.pick_adapter):
  1. poznata obitelj (ultralytics paket / `roi_heads` atribut)   <- brzo i sigurno za nase modele
  2. FALLBACK: klasifikacija formata izlaza (ovaj modul)          <- za nepoznat model
  3. neprepoznato -> None -> jezgra degradira na KD-only (glasno upozorenje, NE prekid)

STO OBLIK KAZE, A STO NE: dimenzije otkrivaju OBITELJ glave, ali NE i konvenciju okvira
([N,4] moze biti xyxy / xywh / cxcywh, u pikselima ili normalizirano). Zato konvencija ide
kroz `probe_box_layout` — dekodira po svim pretpostavkama i bira onu koja daje VALJANE okvire
(pozitivna sirina/visina, unutar granica, razumna povrsina). Isto nacelo kao LAYER_REGISTER:
provjeri pokusajem, ne citanjem koda.
"""

import torch

# Prepoznate obitelji formata. "decoded" = okviri su vec u upotrebljivom obliku (bez sidara/DFL).
# SVI osim dense_cn/dense_nc potvrdjeni na STVARNIM modelima (v. _outfmt64.py).
FORMATS = {
    "boxes_dicts":  "lista dict{boxes,scores,labels} po slici (torchvision: frcnn/retinanet/ssd)",
    "nms_out":      "tenzor [B, D, 6] — xyxy + conf + CJELOBROJAN id razreda (yolo26 end2end eval)",
    "dense_split":  "dict{boxes:[B,4,N], scores:[B,K,N]} — gusta glava, razdvojeni okviri i skorovi",
    "set_pred":     "dict{logits|pred_logits, pred_boxes} (DETR/YOLOS set-prediction, cxcywh norm)",
    "dense_cn":     "tenzor [B, N, 4+K] — gusta glava, sidra u 2. dimenziji (yolov5-stil)",
    "dense_nc":     "tenzor [B, 4+K, N] — gusta glava, kanali u 2. dimenziji (yolov8-stil)",
    "feat_pyramid": "lista 4D tenzora RAZLICITIH kanala po razini — piramida ZNACAJKI, NIJE izlaz detekcije",
    "multilevel":   "lista 4D tenzora ISTIH kanala po razini — sirova glava prije spajanja",
    "unknown":      "neprepoznato",
}

# Format -> koji postojeci decode ga zna obraditi. None = nemamo decode (degradacija na KD-only).
FORMAT_ADAPTER = {
    "boxes_dicts":  "FrcnnAdapter",    # vec dekodirano; torchvision-stil citanja
    "nms_out":      "builtin",         # vec dekodirano; decode() ga cita izravno
    "dense_split":  "builtin",
    "set_pred":     "builtin",
    "dense_cn":     "YoloAdapter",     # gusti decode + NMS
    "dense_nc":     "YoloAdapter",
    "feat_pyramid": None,              # to NISU predikcije nego znacajke
    "multilevel":   None,              # trazi podjelu kanala tog modela — nemamo je cime provjeriti
    "unknown":      None,
}

_BOX_KEYS = ("boxes", "bbox", "bboxes")
_LOGIT_KEYS = ("logits", "pred_logits", "class_logits", "scores")


def _first_tensor(o):
    """Prvi tenzor u ugnijezdenoj strukturi (izlaz zna biti tuple(pred, aux))."""
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, dict):
        for v in o.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(o, (list, tuple)):
        for v in o:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def classify_output(out):
    """Klasificiraj SIROVI izlaz modela. Vrati dict:
       {format, why, decoded, n_anchors?, n_channels?}  — format iz FORMATS."""
    # --- A. lista dict-ova s okvirima (torchvision) ---
    if isinstance(out, (list, tuple)) and len(out) and isinstance(out[0], dict):
        keys = set(out[0].keys())
        if keys & set(_BOX_KEYS):
            return {"format": "boxes_dicts", "decoded": True,
                    "why": "lista dict-ova, kljucevi {}".format(sorted(keys)[:4])}

    # --- B. dict-oliki izlazi (DETR/YOLOS set-prediction, razdvojeni box/score) ---
    if hasattr(out, "keys"):
        keys = set(out.keys())
        if "pred_boxes" in keys:                          # DETR/YOLOS; HF koristi 'logits', DETR-orig 'pred_logits'
            lk = next((k for k in _LOGIT_KEYS if k in keys), None)
            pl = out[lk] if lk else None
            return {"format": "set_pred", "decoded": False, "logit_key": lk,
                    "why": "dict pred_boxes {}{}".format(
                        tuple(out["pred_boxes"].shape),
                        " + {} {}".format(lk, tuple(pl.shape)) if torch.is_tensor(pl) else "")}
        if "boxes" in keys and "scores" in keys and torch.is_tensor(out["boxes"]) \
                and out["boxes"].dim() == 3:              # gusta glava s razdvojenim izlazima (yolo26 train)
            return {"format": "dense_split", "decoded": False,
                    "why": "dict boxes {} + scores {}".format(
                        tuple(out["boxes"].shape), tuple(out["scores"].shape))}

    # --- C. lista/tuple 4D tenzora po razini ---
    if isinstance(out, (list, tuple)) and len(out) and all(
            torch.is_tensor(t) and t.dim() == 4 for t in out):
        chans = [t.shape[1] for t in out]
        shapes = [tuple(t.shape) for t in out[:3]]
        if len(set(chans)) == 1 and chans[0] >= 5:        # ISTI kanali po razini -> glava
            return {"format": "multilevel", "decoded": False, "n_levels": len(out),
                    "why": "{} razina, {} kanala na svakoj, oblici {}".format(len(out), chans[0], shapes)}
        return {"format": "feat_pyramid", "decoded": False, "n_levels": len(out),
                "why": "{} razina RAZLICITIH kanala {} -> znacajke, ne predikcije".format(len(out), chans)}

    # --- D. gusti tenzor [B, ?, ?] ---
    t = out if torch.is_tensor(out) else _first_tensor(out)
    if torch.is_tensor(t) and t.dim() == 3:
        b, d1, d2 = t.shape
        # D0. VEC DEKODIRAN NMS izlaz [B, D, 6] = xyxy + conf + ID RAZREDA (cjelobrojan).
        # Bez ove provjere bi ispalo "gusta glava s ~2 razreda" — yolo26 end2end eval upravo tako izgleda.
        if d2 == 6 and d1 != 6:
            cls_col, conf_col = t[..., 5].float(), t[..., 4].float()
            if torch.allclose(cls_col, cls_col.round()) and float(conf_col.min()) >= -1e-6 \
                    and float(conf_col.max()) <= 1.0 + 1e-6:
                return {"format": "nms_out", "decoded": True, "n_det": int(d1),
                        "why": "[B,{},6]: ch5 cjelobrojan (id razreda 0..{}), ch4 u [0,1] (conf)".format(
                            d1, int(cls_col.max()))}
        # sidara je RED VELICINE vise nego kanala (tisuce naspram ~4+K) -> manja dimenzija = kanali
        if d1 == d2:
            return {"format": "unknown", "decoded": False,
                    "why": "kvadratni [B,{},{}] — ne razaznajem sidra od kanala".format(d1, d2)}
        ch, n = (d1, d2) if d1 < d2 else (d2, d1)
        if ch < 5:
            return {"format": "unknown", "decoded": False,
                    "why": "kanala {} < 5 (premalo za 4 koordinate + bar 1 razred)".format(ch)}
        fmt = "dense_nc" if d1 < d2 else "dense_cn"
        return {"format": fmt, "decoded": False, "n_anchors": int(n), "n_channels": int(ch),
                "why": "gusti {} — {} sidara x {} kanala (4 koord + ~{} razreda)".format(
                    tuple(t.shape), n, ch, ch - 4)}

    shape = tuple(t.shape) if torch.is_tensor(t) else type(out).__name__
    return {"format": "unknown", "decoded": False, "why": "neprepoznata struktura: {}".format(shape)}


# =========================== KONVENCIJA OKVIRA (mjerenjem) =========================== #
_LAYOUTS = ("xyxy", "xywh", "cxcywh")


def _to_xyxy(b, layout):
    if layout == "xyxy":
        return b
    x, y, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    if layout == "xywh":                                   # gornji-lijevi + sirina/visina
        return torch.stack([x, y, x + w, y + h], -1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], -1)   # cxcywh


def probe_box_layout(boxes, img_hw, sample=512, tol=1e-3):
    """IZMJERI konvenciju okvira umjesto da je pretpostavis.

    boxes: [...,4] bilo kojeg rasporeda/skale. img_hw: (H, W) ulazne slike.
    Za svaku kombinaciju (raspored x normalizirano?) pretvori u xyxy i ocijeni UDIO VALJANIH okvira:
    x2>x1, y2>y1, unutar granica (uz malu toleranciju), povrsina u [0.01%, 100%] slike.

    OGRANICENJE KOJE SE PRIJAVLJUJE, NE PRESUCUJE: `xywh` (gornji-lijevi) i `cxcywh` (srediste) se
    razlikuju SAMO po tome probije li okvir granicu. Ako su svi okviri udobno unutar slike, oba
    tumacenja daju 100% valjanih i razlika je NEMJERLJIVA. Tada vracamo `confident=False` i popis
    `ambiguous` — pozivatelj mora naci drugi signal, a ne dobiti tiho pogodjen odgovor.
    Kod pravih gustih izlaza (tisuce sidara preko cijele slike) okviri DOTICU rub pa se razlucuje.

    Vrati {'layout','normalized','score','confident','ambiguous','table'}."""
    H, W = img_hw
    b = boxes.detach().float().reshape(-1, 4)
    if b.numel() == 0:
        return {"layout": None, "normalized": None, "score": 0.0,
                "confident": False, "ambiguous": [], "table": {}}
    if b.shape[0] > sample:
        b = b[torch.randperm(b.shape[0])[:sample]]

    table = {}
    for norm in (False, True):
        scale = torch.tensor([W, H, W, H], dtype=b.dtype, device=b.device) if norm else 1.0
        for lay in _LAYOUTS:
            xy = _to_xyxy(b * scale if norm else b, lay)
            x1, y1, x2, y2 = xy.unbind(-1)
            wpos, hpos = (x2 - x1), (y2 - y1)
            area = (wpos.clamp(min=0) * hpos.clamp(min=0)) / float(W * H)
            ok = ((wpos > 0) & (hpos > 0)
                  & (x1 >= -0.02 * W) & (y1 >= -0.02 * H)
                  & (x2 <= 1.02 * W) & (y2 <= 1.02 * H)
                  & (area > 1e-4) & (area <= 1.0))
            table["{}{}".format(lay, "/norm" if norm else "/px")] = float(ok.float().mean())

    best = max(table, key=table.get)
    top = table[best]
    tied = sorted(k for k, v in table.items() if abs(v - top) <= tol)   # nerazlucivi kandidati
    lay, sc = best.split("/")
    return {"layout": lay, "normalized": sc == "norm", "score": top,
            "confident": len(tied) == 1 and top > 0.5,
            "ambiguous": [] if len(tied) == 1 else tied, "table": table}


# =========================== JAVNI ULAZ =========================== #
def describe(model, sample_input, device=None):
    """Pokreni model na uzorku i klasificiraj izlaz. NIKAD ne baca — kvar = unknown (s razlogom).
    sample_input: ono sto model prima (npr. `adapter.forward_example(device)`)."""
    was = model.training
    try:
        model.eval()
        with torch.no_grad():
            out = model(sample_input)
    except BaseException as e:
        return {"format": "unknown", "decoded": False,
                "why": "forward pukao: {}: {}".format(type(e).__name__, str(e)[:120])}
    finally:
        model.train(was)
    rep = classify_output(out)
    rep["adapter"] = FORMAT_ADAPTER.get(rep["format"])
    rep["doc"] = FORMATS.get(rep["format"], "")
    return rep


# =========================== DECODE -> jedinstven oblik =========================== #
# Svaki prepoznat format svodimo na ISTO: lista po slici {boxes:[N,4] xyxy PIKSELI, scores:[N], labels:[N]}.
# To je oblik koji torchmetrics mAP jede, pa jezgra dalje ne mora znati odakle je dosao.

def _nms(boxes, scores, labels, iou, max_det):
    from torchvision.ops import batched_nms
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep = batched_nms(boxes, scores, labels, iou)[:max_det]
    return boxes[keep], scores[keep], labels[keep]


def anchor_grid(n, img_hw, strides=(8, 16, 32)):
    """Standardna FPN resetka za N sidara: centri celija + korak po sidru.
    Celija po razini je (H/s)*(W/s); ako zbroj ne pogodi N, ovo NIJE ta resetka -> None.
    (640x640 sa 8/16/32 -> 6400+1600+400 = 8400 = yolov8+ konvencija.)"""
    H, W = img_hw
    if sum((H // s) * (W // s) for s in strides) != n:
        return None
    cx, st = [], []
    for s in strides:
        gh, gw = H // s, W // s
        ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        cx.append(torch.stack([(xs.reshape(-1) + 0.5) * s, (ys.reshape(-1) + 0.5) * s], -1).float())
        st.append(torch.full((gh * gw,), float(s)))
    return torch.cat(cx), torch.cat(st)


def _ltrb_to_xyxy(d, centers, strides):
    """Sidro-relativne udaljenosti (l,t,r,b) u jedinicama koraka -> apsolutni xyxy."""
    c, s = centers.to(d.device), strides.to(d.device).unsqueeze(-1)
    return torch.cat([c - d[:, :2] * s, c + d[:, 2:] * s], -1)


def _dense_to_dets(box_nk, score_nk, img_hw, conf, iou, max_det, layout=None, min_score=0.2):
    """box_nk [N,4] (NEPOZNATA konvencija), score_nk [N,K] -> dets nakon praga i NMS-a.

    Konvencija se MJERI, dvije skupine hipoteza:
      (a) izravne koordinate — xyxy/xywh/cxcywh, px ili normalizirano (`probe_box_layout`)
      (b) SIDRO-RELATIVNE ltrb udaljenosti u jedinicama koraka (yolov8+ DFL izlaz)
    Ako nijedna ne da razuman udio valjanih okvira (< `min_score`) -> None, umjesto smeca.
    (Nalaz: yolo26 train `boxes` su upravo (b) — raspon [-0.6, 13.7], nikakve koordinate.)"""
    lay = layout or probe_box_layout(box_nk, img_hw)

    if lay.get("kind") != "anchor_ltrb" and lay["score"] < min_score:   # (a) pala -> probaj (b)
        g = anchor_grid(box_nk.shape[0], img_hw)
        if g is not None:
            probe = probe_box_layout(_ltrb_to_xyxy(box_nk.float(), *g), img_hw)
            if probe["score"] >= min_score:
                lay = {"kind": "anchor_ltrb", "layout": "xyxy", "normalized": False,
                       "score": probe["score"], "confident": True, "ambiguous": []}

    if lay.get("kind") == "anchor_ltrb":
        g = anchor_grid(box_nk.shape[0], img_hw)
        if g is None:
            return None
        xyxy = _ltrb_to_xyxy(box_nk.float(), *g)
    elif lay["score"] < min_score:
        return None                                                     # ne znamo procitati -> posteno None
    else:
        scale = torch.tensor([img_hw[1], img_hw[0]] * 2, device=box_nk.device) if lay["normalized"] else 1.0
        xyxy = _to_xyxy(box_nk * scale, lay["layout"])

    sc, lb = score_nk.max(-1)
    m = sc >= conf
    b, s, l = _nms(xyxy[m], sc[m], lb[m], iou, max_det)
    return {"boxes": b, "scores": s, "labels": l, "layout": lay}


def decode(out, img_hw, conf=0.25, iou=0.7, max_det=300, rep=None):
    """Dekodiraj SIROVI izlaz u [{boxes,scores,labels}] po slici. Vrati None ako format nema decode.

    img_hw = (H, W) ulazne slike (za skaliranje normaliziranih okvira). Konvencija okvira se MJERI
    (`probe_box_layout`), ne pretpostavlja. NIKAD ne baca — nepoznato = None."""
    rep = rep or classify_output(out)
    f = rep["format"]
    try:
        if f == "boxes_dicts":
            return [{"boxes": o["boxes"], "scores": o.get("scores"), "labels": o.get("labels")}
                    for o in out]

        if f == "nms_out":                                 # vec dekodirano: xyxy + conf + id
            t = out if torch.is_tensor(out) else _first_tensor(out)
            res = []
            for i in range(t.shape[0]):
                d = t[i]
                m = d[:, 4] >= conf
                res.append({"boxes": d[m, :4], "scores": d[m, 4], "labels": d[m, 5].long()})
            return res

        if f == "set_pred":                                # DETR/YOLOS: cxcywh NORMALIZIRANO, bez NMS-a
            lg = out[rep.get("logit_key") or "logits"]
            pb = out["pred_boxes"]
            prob = lg.softmax(-1)[..., :-1]                # zadnji razred = "no object" -> izbaci
            H, W = img_hw
            res = []
            for i in range(pb.shape[0]):
                sc, lb = prob[i].max(-1)
                xyxy = _to_xyxy(pb[i] * torch.tensor([W, H, W, H], device=pb.device), "cxcywh")
                m = sc >= conf
                res.append({"boxes": xyxy[m], "scores": sc[m], "labels": lb[m]})
            return res

        if f == "dense_split":                             # {boxes:[B,4,N], scores:[B,K,N]}
            bx, sc = out["boxes"], out["scores"]
            res, lay = [], None
            for i in range(bx.shape[0]):
                d = _dense_to_dets(bx[i].transpose(0, 1), sc[i].transpose(0, 1).sigmoid(),
                                   img_hw, conf, iou, max_det, lay)
                if d is None:
                    return None                            # konvencija okvira neprepoznata -> posteno None
                lay = d.pop("layout")
                res.append(d)
            return res

        if f in ("dense_nc", "dense_cn"):                  # [B,4+K,N] / [B,N,4+K]
            t = out if torch.is_tensor(out) else _first_tensor(out)
            res, lay = [], None
            for i in range(t.shape[0]):
                m = t[i] if f == "dense_cn" else t[i].transpose(0, 1)      # -> [N, 4+K]
                d = _dense_to_dets(m[:, :4], m[:, 4:].sigmoid(), img_hw, conf, iou, max_det, lay)
                if d is None:
                    return None
                lay = d.pop("layout")
                res.append(d)
            return res
    except BaseException as e:
        print("[outfmt] decode('{}') pukao: {}: {}".format(f, type(e).__name__, str(e)[:120]))
        return None
    return None                                            # feat_pyramid / multilevel / unknown


def explain(rep):
    """Jednoredni ljudski sazetak (za log/GUI karticu)."""
    if rep["format"] == "unknown":
        return "[outfmt] NEPREPOZNAT detekcijski izlaz — {}. Nastavljam BEZ mAP-a (KD-only gate).".format(rep["why"])
    ad = rep.get("adapter")
    if ad is None:
        return "[outfmt] format '{}' prepoznat ({}), ali decode NIJE implementiran -> KD-only gate.".format(
            rep["format"], rep["why"])
    return "[outfmt] format '{}' -> decode preko {} · {}".format(rep["format"], ad, rep["why"])
