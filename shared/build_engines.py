# -*- coding: utf-8 -*-
"""build_engines.py — ONNX -> TensorRT engine, NA UREDJAJU.

Engine je vezan uz konkretan GPU, driver i verziju TRT-a, pa se ne prenosi i ne commita.

PRECIZNOST ODLUCUJE GRAF, NE MAPA. Nosi li ONNX Q/DQ cvorove, gradi se INT8 engine; ako ih
nema, FP16. To nije udobnost nego jedina obrana od tihe zamjene: kad bi preciznost dolazila
iz imena mape, mogli bismo dobiti FP16 engine u SLINN_OPTIM-u i izmjeriti ga kao INT8.

U BASELINE_OPTIM Q/DQ cvorova NEMA po konstrukciji — SliNN ondje nije prosao. Ako se ipak
pojave, gradnja se ODBIJA: kvantizacija u baselineu bi obesmislila omjer
SLINN_OPTIM / BASELINE_OPTIM. Obrnuto vrijedi za celiju s `_qat` u imenu: graf bez Q/DQ znaci
da se kvantizacija nije materijalizirala, pa se i to odbija.

INT8 IDE IZ QDQ GRAFA, BEZ KALIBRATORA. Nauceno u PTQ fazi: `BuilderFlag.INT8` je DOPUSTENJE,
a Q/DQ cvorovi su NAREDBA. Kalibratora ovdje nema jer skale vec postoje — izracunao ih je QAT.

POZNATO OGRANICENJE (izmjereno na laptopu, TRT 10.12, prije nego je isto proslo Jetsonom):
yolo26n INT8 engine se NE GRADI. TRT odbija QDQ uzorak unutar C2f bloka:

    filterQDQFormats: /m/model.8/m.0/cv1/conv/fq_in/QuantizeLinear[QUANTIZE]:
    All of the candidates were removed, which points to the node being incorrectly
    marked as an int8 node.

Isti graf u ONNX Runtimeu prolazi (384 Q/DQ cvorova, odstupanje 5.8%), pa nije rijec o
slomljenom izvozu nego o TRT-ovoj propagaciji preciznosti kroz taj blok. Rucno prisiljavanje
slojeva u FP32 (`OBEY_PRECISION_CONSTRAINTS` + `layer.precision`) je PROBANO i ODBACENO:
yolo i dalje pada, a voc_deeplabv3 — koji je bez toga gradio engine od 12.9 MB — prestaje
raditi, jer je ta zastavica globalna i stroga. Rjesenje je vjerojatno u izuzimanju po uzorku
imena PRIJE izvoza (kako to radi ultralytics/NNCF IgnoredScope), ne pri gradnji enginea.

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
FP16_NAME = "model_fp16.engine"
INT8_NAME = "model_int8.engine"

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


def _base(cell):
    """Ime modela iz imena celije: `voc_deeplabv3__ckpt_2_qat` -> `voc_deeplabv3`."""
    return cell.split("__", 1)[0]


def _cells():
    """Celije u mjernoj mapi, poredane kao ORDER. BASELINE_OPTIM ima po jednu celiju po modelu,
    SLINN_OPTIM po jednu po checkpointu — zato se popis CITA S DISKA."""
    if not os.path.isdir(BASE):
        return []
    out = [c for c in os.listdir(BASE)
           if os.path.isdir(os.path.join(BASE, c)) and _base(c) in PROFILES]
    rank = {m: i for i, m in enumerate(ORDER)}
    return sorted(out, key=lambda c: (rank.get(_base(c), 99), c))


def build(onnx_path, engine_path, profile, int8=False):
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
    cfg.set_flag(trt.BuilderFlag.FP16)      # i u INT8 enginu: sloj koji ne moze u int8 ide u fp16
    if int8:
        cfg.set_flag(trt.BuilderFlag.INT8)  # DOPUSTENJE; naredbu daju Q/DQ cvorovi iz grafa
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

    cells = _cells()
    baseline = CELL.upper().startswith("BASELINE")
    log(f"### TensorRT -> {BASE}   ({len(cells)} celija)")
    log(f"### workspace {WORKSPACE_GB} GB   batch 1   preciznost odlucuje graf")
    if baseline:
        log("### baseline mapa — Q/DQ cvorovi se ODBIJAJU (INT8 je efekt SliNN-a)")
    ok = 0
    for name in cells:
        d = os.path.join(BASE, name)
        onnx_path = os.path.join(d, "model.onnx")
        if not os.path.isfile(onnx_path):
            log(f"  {name:34} PRESKOCEN  nema model.onnx")
            continue
        q = qdq_count(onnx_path)
        if q and baseline:
            log(f"  {name:34} ODBIJEN    {q} Q/DQ cvorova — INT8 ne smije u baseline")
            continue
        if not q and "_qat" in name:
            log(f"  {name:34} ODBIJEN    0 Q/DQ cvorova u `_qat` celiji — kvantizacija se "
                f"nije materijalizirala")
            continue
        eng = os.path.join(d, INT8_NAME if q else FP16_NAME)
        for stale in (INT8_NAME, FP16_NAME):        # nikad dva enginea u istoj celiji
            p = os.path.join(d, stale)
            if p != eng and os.path.isfile(p):
                os.remove(p)
        t0 = time.time()
        try:
            dyn = build(onnx_path, eng, PROFILES.get(_base(name), {}), int8=bool(q))
            mb = os.path.getsize(eng) / 1024 ** 2
            shape = "statican" if not dyn else "dinamican: " + ",".join(dyn)
            kind = f"INT8 {q:>4} Q/DQ" if q else "FP16          "
            log(f"  {name:34} OK  {kind}  {mb:8.1f} MB   {time.time() - t0:6.1f}s   {shape}")
            ok += 1
        except Exception as e:
            if os.path.isfile(eng):
                os.remove(eng)      # bolje glasan pad u evalu nego stari engine
            log(f"  {name:34} PAO   {type(e).__name__}: {str(e)[:110]}")
    log(f"### {ok}/{len(cells)} enginea")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(BASE, f"build_log_{stamp}.txt")
    io.open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    log(f"[save] -> {out}")


if __name__ == "__main__":
    main()
