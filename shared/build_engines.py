# -*- coding: utf-8 -*-
"""build_engines.py — ONNX -> TensorRT FP16 engine, NA UREDJAJU.

Engine je vezan uz konkretan GPU, driver i verziju TRT-a, pa se ne prenosi i ne commita.

SAMO FP16. Nikakav INT8 i nikakva kalibracija — to je efekt SliNN-a i ako procuri u
baseline, omjer SLINN_OPTIM / BASELINE_OPTIM vise nista ne znaci. Zato se `BuilderFlag.INT8`
ovdje ni ne spominje, a broj kvantizacijskih cvorova se na kraju provjerava brojanjem.

PROFILI: engine se gradi za BATCH 1 (v. runners.TrtShim — veci batch ide petljom), pa je
batch os svugdje min=opt=max=1. Ostale dinamicne osi su IZMJERENE skriptom probe_shapes.py
nad stvarnim val skupom, ne pogodjene:

    voc_deeplabv3    H 520..1494, W 520..1721   (233 razlicita oblika; smaller-edge 520 uz
                                                 cuvanje omjera stranica)
    sst2_distilbert  s 4..55                    (collate pada=True -> duljina po recenici)
    midas_depth      konstantno 192x256         (NYU je sav istog oblika -> min=opt=max)
    housing, m5      samo batch os
    yolo26n/26l      graf je potpuno statican   -> profil se ne postavlja
"""
import io
import json
import os
import time

WORKSPACE_GB = 2          # limit, ne rezervacija; Orin Nano dijeli 7.4 GB s ostatkom sustava
OPT_LEVEL = None          # None = zadano; nizi ubrzava gradnju, usporava engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL = os.environ.get("EVAL_DIR", "BASELINE_OPTIM")
BASE = os.path.join(ROOT, CELL)
ENGINE_NAME = "model_fp16.engine"

ORDER = ["housing_mlp", "speechcommands_m5", "midas_depth", "sst2_distilbert",
         "voc_deeplabv3", "yolo26n", "yolo26l"]

# ime ulaza -> (min, opt, max)
PROFILES = {
    "housing_mlp":       {"input": ((1, 8), (1, 8), (1, 8))},
    "speechcommands_m5": {"input": ((1, 1, 8000),) * 3},
    "midas_depth":       {"input": ((1, 3, 192, 256),) * 3},
    "voc_deeplabv3":     {"input": ((1, 3, 520, 520), (1, 3, 520, 693), (1, 3, 1494, 1721))},
    "sst2_distilbert":   {"input_ids":      ((1, 1), (1, 16), (1, 64)),
                          "attention_mask": ((1, 1), (1, 16), (1, 64))},
    "yolo26n": {},
    "yolo26l": {},
}


def build(onnx_path, engine_path, profile):
    import tensorrt as trt
    lg = trt.Logger(trt.Logger.ERROR)
    b = trt.Builder(lg)
    net = b.create_network(0)
    parser = trt.OnnxParser(net, lg)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX parse: " + str(parser.get_error(0))[:200])

    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB * (1 << 30))
    cfg.set_flag(trt.BuilderFlag.FP16)
    if OPT_LEVEL is not None:
        cfg.builder_optimization_level = OPT_LEVEL

    dyn = []
    for i in range(net.num_inputs):
        t = net.get_input(i)
        if any(d < 0 for d in t.shape):
            dyn.append(t.name)
    if dyn:
        missing = [n for n in dyn if n not in profile]
        if missing:
            raise RuntimeError(f"dinamicne osi bez profila: {missing}")
        p = b.create_optimization_profile()
        for n in dyn:
            lo, opt, hi = profile[n]
            p.set_shape(n, lo, opt, hi)
        cfg.add_optimization_profile(p)

    ser = b.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError("build_serialized_network vratio None")
    with open(engine_path, "wb") as f:
        f.write(ser)
    return dyn


def qdq_count(onnx_path):
    """Nula kvantizacijskih cvorova je UVJET, ne ocekivanje — provjerava se brojanjem."""
    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    return sum(1 for n in m.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear"))


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"### TensorRT FP16 -> {BASE}")
    log(f"### workspace {WORKSPACE_GB} GB   batch 1   INT8: nigdje")
    ok = 0
    for name in ORDER:
        d = os.path.join(BASE, name)
        onnx_path = os.path.join(d, "model.onnx")
        eng = os.path.join(d, ENGINE_NAME)
        if not os.path.isfile(onnx_path):
            log(f"  {name:20} PRESKOCEN  nema model.onnx")
            continue
        q = qdq_count(onnx_path)
        if q:
            log(f"  {name:20} ODBIJEN    {q} Q/DQ cvorova — INT8 ne smije u baseline")
            continue
        t0 = time.time()
        try:
            dyn = build(onnx_path, eng, PROFILES.get(name, {}))
            mb = os.path.getsize(eng) / 1024 ** 2
            shape = "statican" if not dyn else "dinamican: " + ",".join(dyn)
            log(f"  {name:20} OK    {mb:8.1f} MB   {time.time() - t0:6.1f}s   {shape}")
            ok += 1
        except Exception as e:
            if os.path.isfile(eng):
                os.remove(eng)      # bolje glasan pad u evalu nego stari engine
            log(f"  {name:20} PAO   {type(e).__name__}: {str(e)[:110]}")
    log(f"### {ok}/{len(ORDER)} enginea")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(BASE, f"build_log_{stamp}.txt")
    io.open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    log(f"[save] -> {out}")


if __name__ == "__main__":
    main()
