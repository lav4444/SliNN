"""
qat_schoolcnn.py — Quantization-Aware Training za SchoolCNN, KD loss (bez GT).

QAT = fine-tune s fake-quant čvorovima (FX graph QAT, fbgemm/x86). Student uči težine otporne na
int8 zaokruživanje oponašajući FROZEN FP32 original (logit-destilacija, soft BCE) — BEZ pravih labela.
Kreće od treniranih FP32 težina, kratki fine-tune uz niski LR, pa convert u pravi int8.

Cilj: pokazati MANJI pad kvalitete od PTQ-a (PTQ fbgemm static ~0.6075 -> QAT bliže FP32 0.6165).
Reuse: ptq_schoolcnn (eval_panel, load_fp32, podaci, konstante) + quant_common. BEZ CLI argumenata.
"""

import copy
import os
import sys

import torch
import torch.nn.functional as F
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

_HERE = os.path.dirname(os.path.abspath(__file__))
_PTQ_SC = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "PTQ", "schoolcnn")
sys.path.insert(0, _PTQ_SC)
import ptq_schoolcnn as PT                      # eval_panel, load_fp32, C2, Q, NUM_CLASSES, INPUT_SIZE, ...

Q = PT.Q
C2 = PT.C2
NC = PT.NUM_CLASSES

QENGINE = "x86"
EPOCHS = 12
LR = 1e-4                                       # niski LR (QAT fine-tune)
BATCH = 32
EVAL_BATCH = 32
OUT_CSV = os.path.join(_HERE, "schoolcnn_qat_report.csv")
OUT_JSON = os.path.join(_HERE, "schoolcnn_qat_report.json")
CSV_COLS = ["format", "backend", "cpu_ms", "size_mb",
            "map_macro", "map_micro", "f1_macro", "acc_macro", "auroc_macro", "bce"]


def main():
    torch.backends.quantized.engine = QENGINE
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = PT.load_fp32().to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    train_loader = C2.make_loader("train", BATCH, shuffle=True, num_workers=4)
    val_loader = C2.make_loader("val", EVAL_BATCH, shuffle=False, num_workers=4)
    example = next(iter(val_loader))[0][:1].clone()                 # [1,3,320,320] cpu

    # --- prepare QAT (FX graph) ---
    student = PT.load_fp32().train()
    qmap = tq.get_default_qat_qconfig_mapping(QENGINE)
    prepared = prepare_qat_fx(student, qmap, example_inputs=(example,)).to(dev)

    def kd_loss(x):
        with torch.no_grad():
            t_prob = torch.sigmoid(teacher(x))
        return F.binary_cross_entropy_with_logits(prepared(x), t_prob)

    opt = torch.optim.Adam(prepared.parameters(), lr=LR)
    print(f"\n########## QAT — SchoolCNN (KD loss, fake-quant {QENGINE}, {EPOCHS} ep, LR {LR}) ##########")

    best = {"map": -1.0, "state": None}
    for ep in range(1, EPOCHS + 1):
        prepared.train()
        tot = 0.0; n = 0
        for x, _y in train_loader:
            x = x.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = kd_loss(x)
            loss.backward(); opt.step()
            tot += float(loss) * x.size(0); n += x.size(0)
        prepared.eval()                                            # fake-quant simulira int8 u eval-u
        panel = PT.eval_panel(PT.make_infer(prepared, dev), val_loader, NC)
        m = panel["map_macro"]
        star = m > best["map"]
        if star:
            best = {"map": m, "state": copy.deepcopy(prepared.state_dict())}
        print(f"  [ep{ep:2d}] kd={tot/max(n,1):.4f} | val_mAP(fake-quant)={m:.4f}{'  *best' if star else ''}")

    prepared.load_state_dict(best["state"])
    int8 = convert_fx(prepared.cpu().eval())                       # pravi int8 model (CPU)

    # --- finalna evaluacija + latencija + veličina ---
    cpu = torch.device("cpu")
    panel = PT.eval_panel(PT.make_infer(int8, cpu), val_loader, NC)
    with torch.no_grad():
        lat = Q.benchmark(lambda: int8(example.to(cpu)), cpu, 15, 100)["median_ms"]
    size = Q.model_size_mb(int8)

    rows = [{"format": "INT8-QAT", "backend": f"PyTorch {QENGINE}", "cpu_ms": round(lat, 4),
             "size_mb": round(size, 4),
             **{k: round(panel[k], 5) for k in ("map_macro", "map_micro", "f1_macro", "acc_macro", "auroc_macro", "bce")}}]

    # usporedba s PTQ + FP32 (iz postojećeg PTQ izvještaja)
    cmp = {}
    import json
    if os.path.exists(PT.OUT_JSON):
        for r in json.load(open(PT.OUT_JSON))["rows"]:
            if r["format"] in ("FP32", "INT8-PT-static"):
                cmp[r["format"]] = r.get("map_macro")

    meta = {"model": "SchoolCNN", "method": "QAT", "loss": "KD (logit, no GT)", "quant_engine": QENGINE,
            "epochs": EPOCHS, "lr": LR, "device_train": str(dev),
            "compare_map": {"FP32": cmp.get("FP32"), "INT8-PTQ-static": cmp.get("INT8-PT-static"),
                            "INT8-QAT": round(panel["map_macro"], 5)}}
    Q.write_report(rows, CSV_COLS, OUT_CSV, OUT_JSON, meta)

    print("\n=== USPOREDBA (macro mAP) ===")
    print(f"  FP32            : {cmp.get('FP32')}")
    print(f"  INT8 PTQ static : {cmp.get('INT8-PT-static')}")
    print(f"  INT8 QAT (KD)   : {panel['map_macro']:.5f}   | CPU {lat:.3f} ms | {size:.2f} MB")
    print("########## QAT GOTOVO ##########")


if __name__ == "__main__":
    main()
