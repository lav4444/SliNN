
import torch

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

FORMAT_ADAPTER = {
    "boxes_dicts":  "FrcnnAdapter",
    "nms_out":      "builtin",
    "dense_split":  "builtin",
    "set_pred":     "builtin",
    "dense_cn":     "YoloAdapter",
    "dense_nc":     "YoloAdapter",
    "feat_pyramid": None,
    "multilevel":   None,
    "unknown":      None,
}

_BOX_KEYS = ("boxes", "bbox", "bboxes")
_LOGIT_KEYS = ("logits", "pred_logits", "class_logits", "scores")


def _first_tensor(o):
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
    if isinstance(out, (list, tuple)) and len(out) and isinstance(out[0], dict):
        keys = set(out[0].keys())
        if keys & set(_BOX_KEYS):
            return {"format": "boxes_dicts", "decoded": True,
                    "why": "lista dict-ova, kljucevi {}".format(sorted(keys)[:4])}

    if hasattr(out, "keys"):
        keys = set(out.keys())
        if "pred_boxes" in keys:
            lk = next((k for k in _LOGIT_KEYS if k in keys), None)
            pl = out[lk] if lk else None
            return {"format": "set_pred", "decoded": False, "logit_key": lk,
                    "why": "dict pred_boxes {}{}".format(
                        tuple(out["pred_boxes"].shape),
                        " + {} {}".format(lk, tuple(pl.shape)) if torch.is_tensor(pl) else "")}
        if "boxes" in keys and "scores" in keys and torch.is_tensor(out["boxes"]) \
                and out["boxes"].dim() == 3:
            return {"format": "dense_split", "decoded": False,
                    "why": "dict boxes {} + scores {}".format(
                        tuple(out["boxes"].shape), tuple(out["scores"].shape))}

    if isinstance(out, (list, tuple)) and len(out) and all(
            torch.is_tensor(t) and t.dim() == 4 for t in out):
        chans = [t.shape[1] for t in out]
        shapes = [tuple(t.shape) for t in out[:3]]
        if len(set(chans)) == 1 and chans[0] >= 5:
            return {"format": "multilevel", "decoded": False, "n_levels": len(out),
                    "why": "{} razina, {} kanala na svakoj, oblici {}".format(len(out), chans[0], shapes)}
        return {"format": "feat_pyramid", "decoded": False, "n_levels": len(out),
                "why": "{} razina RAZLICITIH kanala {} -> znacajke, ne predikcije".format(len(out), chans)}

    t = out if torch.is_tensor(out) else _first_tensor(out)
    if torch.is_tensor(t) and t.dim() == 3:
        b, d1, d2 = t.shape
        if d2 == 6 and d1 != 6:
            cls_col, conf_col = t[..., 5].float(), t[..., 4].float()
            if torch.allclose(cls_col, cls_col.round()) and float(conf_col.min()) >= -1e-6 \
                    and float(conf_col.max()) <= 1.0 + 1e-6:
                return {"format": "nms_out", "decoded": True, "n_det": int(d1),
                        "why": "[B,{},6]: ch5 cjelobrojan (id razreda 0..{}), ch4 u [0,1] (conf)".format(
                            d1, int(cls_col.max()))}
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


_LAYOUTS = ("xyxy", "xywh", "cxcywh")


def _to_xyxy(b, layout):
    if layout == "xyxy":
        return b
    x, y, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    if layout == "xywh":
        return torch.stack([x, y, x + w, y + h], -1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], -1)


def probe_box_layout(boxes, img_hw, sample=512, tol=1e-3):
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
    tied = sorted(k for k, v in table.items() if abs(v - top) <= tol)
    lay, sc = best.split("/")
    return {"layout": lay, "normalized": sc == "norm", "score": top,
            "confident": len(tied) == 1 and top > 0.5,
            "ambiguous": [] if len(tied) == 1 else tied, "table": table}


def describe(model, sample_input, device=None):
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


def _nms(boxes, scores, labels, iou, max_det):
    from torchvision.ops import batched_nms
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep = batched_nms(boxes, scores, labels, iou)[:max_det]
    return boxes[keep], scores[keep], labels[keep]


def anchor_grid(n, img_hw, strides=(8, 16, 32)):
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
    c, s = centers.to(d.device), strides.to(d.device).unsqueeze(-1)
    return torch.cat([c - d[:, :2] * s, c + d[:, 2:] * s], -1)


def _dense_to_dets(box_nk, score_nk, img_hw, conf, iou, max_det, layout=None, min_score=0.2):
    lay = layout or probe_box_layout(box_nk, img_hw)

    if lay.get("kind") != "anchor_ltrb" and lay["score"] < min_score:
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
        return None
    else:
        scale = torch.tensor([img_hw[1], img_hw[0]] * 2, device=box_nk.device) if lay["normalized"] else 1.0
        xyxy = _to_xyxy(box_nk * scale, lay["layout"])

    sc, lb = score_nk.max(-1)
    m = sc >= conf
    b, s, l = _nms(xyxy[m], sc[m], lb[m], iou, max_det)
    return {"boxes": b, "scores": s, "labels": l, "layout": lay}


def decode(out, img_hw, conf=0.25, iou=0.7, max_det=300, rep=None):
    rep = rep or classify_output(out)
    f = rep["format"]
    try:
        if f == "boxes_dicts":
            return [{"boxes": o["boxes"], "scores": o.get("scores"), "labels": o.get("labels")}
                    for o in out]

        if f == "nms_out":
            t = out if torch.is_tensor(out) else _first_tensor(out)
            res = []
            for i in range(t.shape[0]):
                d = t[i]
                m = d[:, 4] >= conf
                res.append({"boxes": d[m, :4], "scores": d[m, 4], "labels": d[m, 5].long()})
            return res

        if f == "set_pred":
            lg = out[rep.get("logit_key") or "logits"]
            pb = out["pred_boxes"]
            prob = lg.softmax(-1)[..., :-1]
            H, W = img_hw
            res = []
            for i in range(pb.shape[0]):
                sc, lb = prob[i].max(-1)
                xyxy = _to_xyxy(pb[i] * torch.tensor([W, H, W, H], device=pb.device), "cxcywh")
                m = sc >= conf
                res.append({"boxes": xyxy[m], "scores": sc[m], "labels": lb[m]})
            return res

        if f == "dense_split":
            bx, sc = out["boxes"], out["scores"]
            res, lay = [], None
            for i in range(bx.shape[0]):
                d = _dense_to_dets(bx[i].transpose(0, 1), sc[i].transpose(0, 1).sigmoid(),
                                   img_hw, conf, iou, max_det, lay)
                if d is None:
                    return None
                lay = d.pop("layout")
                res.append(d)
            return res

        if f in ("dense_nc", "dense_cn"):
            t = out if torch.is_tensor(out) else _first_tensor(out)
            res, lay = [], None
            for i in range(t.shape[0]):
                m = t[i] if f == "dense_cn" else t[i].transpose(0, 1)
                d = _dense_to_dets(m[:, :4], m[:, 4:].sigmoid(), img_hw, conf, iou, max_det, lay)
                if d is None:
                    return None
                lay = d.pop("layout")
                res.append(d)
            return res
    except BaseException as e:
        print("[outfmt] decode('{}') pukao: {}: {}".format(f, type(e).__name__, str(e)[:120]))
        return None
    return None


def explain(rep):
    if rep["format"] == "unknown":
        return "[outfmt] NEPREPOZNAT detekcijski izlaz — {}. Nastavljam BEZ mAP-a (KD-only gate).".format(rep["why"])
    ad = rep.get("adapter")
    if ad is None:
        return "[outfmt] format '{}' prepoznat ({}), ali decode NIJE implementiran -> KD-only gate.".format(
            rep["format"], rep["why"])
    return "[outfmt] format '{}' -> decode preko {} · {}".format(rep["format"], ad, rep["why"])
