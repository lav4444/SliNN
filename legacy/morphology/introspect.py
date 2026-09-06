
import torch
import torch.nn as nn

_CONV = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _is_attention(m):
    n = type(m).__name__.lower()
    return ("attn" in n) or ("attention" in n) or hasattr(m, "num_heads")


def _last_weighted(model):
    last = None
    for m in model.modules():
        if isinstance(m, (*_CONV, nn.Linear)):
            last = m
    return last


def _ultra_head(model):
    seq = getattr(model, "model", None)
    try:
        return list(seq)[-1] if seq is not None else None
    except TypeError:
        return None


def _layer_census(model):
    c = {"conv": 0, "linear": 0, "bn": 0, "frozen_bn": 0, "depthwise": 0, "grouped": 0, "attention": 0}
    for m in model.modules():
        if isinstance(m, _CONV):
            c["conv"] += 1
            g = getattr(m, "groups", 1)
            if g > 1:
                c["grouped"] += 1
                if g == m.in_channels and g == m.out_channels:
                    c["depthwise"] += 1
        elif isinstance(m, nn.Linear):
            c["linear"] += 1
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            c["bn"] += 1
        elif type(m).__name__ == "FrozenBatchNorm2d":
            c["frozen_bn"] += 1
        if _is_attention(m):
            c["attention"] += 1
    c["weighted"] = c["conv"] + c["linear"]
    return c


def profile(model):
    mod = f"{type(model).__module__}.{type(model).__name__}"
    is_ultra = type(model).__module__.startswith("ultralytics")
    has_rpn = hasattr(model, "rpn")
    has_roi = hasattr(model, "roi_heads")
    has_fpn = (hasattr(getattr(model, "backbone", None), "fpn")
               or any("featurepyramid" in type(m).__name__.lower() for m in model.modules()))
    notes = []

    head = _ultra_head(model) if is_ultra else None
    dense_head = end2end = None
    nc = None
    if has_roi and has_rpn:
        task, family = "detection", "frcnn_twostage"
        cs = getattr(getattr(model.roi_heads, "box_predictor", None), "cls_score", None)
        nc = getattr(cs, "out_features", None)
    elif is_ultra:
        family = "ultralytics_yolo"
        hn = type(head).__name__ if head is not None else ""
        task = {"Detect": "detection", "Segment": "segmentation",
                "Classify": "classification", "Pose": "detection"}.get(hn, "detection")
        dense_head = head is not None
        end2end = getattr(head, "end2end", None)
        nc = getattr(head, "nc", None)
        if end2end:
            notes.append("end2end glava (NMS-free, (N,300,6)); za gusti KD/predict iskljuciti end2end")
    else:
        last = _last_weighted(model)
        if isinstance(last, nn.Linear):
            task, family = "classification", "generic_cnn"
            nc = last.out_features
        else:
            task, family = "unknown", "generic"
            notes.append("task nije auto-prepoznat (nema roi/rpn, nije ultralytics, zadnji sloj nije Linear)")

    classifier_head = task == "classification"
    layers = _layer_census(model)

    taps = []
    if has_fpn:
        taps.append({"name": "fpn", "type": "feature", "note": "FPN/neck izlaz -> MSE"})
    if has_rpn:
        taps += [{"name": "rpn_obj", "type": "objectness", "note": "RPN objectness -> BCE"},
                 {"name": "rpn_box", "type": "box", "note": "RPN box delte -> SmoothL1"}]
    if has_roi:
        taps += [{"name": "roi_cls", "type": "logit", "note": "ROI klase -> KL@T"},
                 {"name": "roi_box", "type": "box", "note": "ROI box -> SmoothL1"}]
    if dense_head:
        taps += [{"name": "neck_feat", "type": "feature", "note": "neck mape -> MSE (1x1 proj ako kanali ne pasu)"},
                 {"name": "head_cls", "type": "dense_cls", "note": "per-anchor klase (sigmoid) -> focal"},
                 {"name": "head_box", "type": "box", "note": "per-anchor box -> SmoothL1"}]
    if classifier_head:
        taps += [{"name": "logits", "type": "logit", "note": "izlazni logiti -> KL@T"},
                 {"name": "penult_feat", "type": "feature", "note": "pretposljednje znacajke -> MSE (opc.)"}]

    return {
        "module": mod, "task": task, "family": family, "num_classes": nc,
        "components": {"fpn": has_fpn, "rpn": has_rpn, "roi": has_roi,
                       "dense_head": bool(dense_head), "end2end": end2end, "classifier_head": classifier_head},
        "layers": layers, "kd_taps": taps, "notes": notes,
    }


def print_profile(p):
    print(f"\n=== PROFILE: {p['module']} ===")
    print(f"  task={p['task']} | family={p['family']} | klase={p['num_classes']}")
    comp = ", ".join(f"{k}={v}" for k, v in p["components"].items() if v)
    print(f"  komponente: {comp or '—'}")
    L = p["layers"]
    print(f"  slojevi: conv={L['conv']} (grouped={L['grouped']}, depthwise={L['depthwise']}) "
          f"linear={L['linear']} bn={L['bn']} frozen_bn={L['frozen_bn']} attention={L['attention']} | weighted={L['weighted']}")
    print("  KD-tapovi (dostupno):")
    for t in p["kd_taps"]:
        print(f"    - {t['name']:<10} [{t['type']}]  {t['note']}")
    for n in p["notes"]:
        print(f"  ! {n}")


if __name__ == "__main__":
    import analysis as A
    import config
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for spec in ("fasterrcnn", config.YOLO_PATH):
        m = A.load_any(spec, dev)
        print_profile(profile(m))
