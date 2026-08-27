"""prep_worker.py — PRIPREMA prije kompresije, u koracima (semafor). Pokrece ga GUI Compress.
Koraci -> prep_status.json (GUI cita i crta semafor):
  1. gpu        — postoji li GPU i je li slobodan (<50% VRAM zauzeto)
  2. plan       — dinamicki sizing feature/logit cachea; AUTO-potvrda (green ok / yellow degradirano / red ne stane)
  3. precompute — prikupi sve teacher feature/logite (cache)
  4. perf       — baseline performanse (val_map, val_mar, gflops, params)
Model je FIKSIRAN u config.MODEL_SPEC. Ispis ide i u terminal.
"""

import json
import os
import traceback

import torch

import analysis as A
import compress as C
import config

JOB = os.path.join(os.path.dirname(__file__), "tmp", "gui_job")
STATUS = os.path.join(JOB, "prep_status.json")


def main():
    os.makedirs(JOB, exist_ok=True)
    try:                                                    # cist start
        os.remove(STATUS)
    except OSError:
        pass
    steps = {"gpu": {"light": "gray"}, "batch": {"light": "gray"}, "plan": {"light": "gray"},
             "precompute": {"light": "gray"}, "perf": {"light": "gray"}}

    def write(state):
        json.dump({"state": state, "pid": os.getpid(), "steps": steps}, open(STATUS, "w"))

    write("preparing")
    try:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        spec = config.MODEL_SPEC

        # 1) GPU --------------------------------------------------------------
        g = C.gpu_status()
        if not g["available"]:
            steps["gpu"] = {"light": "yellow", "msg": g["msg"]}
        elif g["ok"]:
            steps["gpu"] = {"light": "green", "msg": f"{g['name']} · zauzeto {g['used_gb']:.1f}/{g['total_gb']:.1f} GB ({g['used_pct']:.0f}%)"}
        else:
            steps["gpu"] = {"light": "yellow", "msg": f"{g['name']} ZAUZET {g['used_pct']:.0f}% (>50%) · oslobodi VRAM"}
        write("preparing")

        # model (za sizing + precompute + perf) -------------------------------
        model = A.load_any(spec, dev); adapter = A.pick_adapter(model)

        # 2) AUTO BATCH — probaj najtezi mod (full FT korak) prije teachera (samo 1 model na GPU = tocno mjerenje)
        steps["batch"] = {"light": "yellow", "msg": "tražim najveći batch koji stane (≤90% slob. VRAM-a)…"}
        write("preparing")
        tb = C.autobatch(model, adapter, dev, free_frac=0.9, cap=64) if dev.type == "cuda" else config.TRAIN_BATCH
        eb = gb = tb                                             # TRAIN je najtezi (fwd+KD+bwd+opt+teacher) -> EVAL/GRAD (laksi) sigurno stanu na istom batchu
        config.TRAIN_BATCH, config.EVAL_BATCH, config.GRAD_BATCH = tb, eb, gb   # za PREP-ove loadere (cache se gradi na tb)
        steps["batch"] = {"light": "green" if dev.type == "cuda" else "yellow",
                          "msg": (f"TRAIN={tb} · EVAL={eb} · GRAD={gb}" if dev.type == "cuda"
                                  else f"CPU — bez probe, koristim config (TRAIN={tb})"),
                          "data": {"train": tb, "eval": eb, "grad": gb}}
        write("preparing")

        teacher = C.copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        train_loader = A.make_gt_loader("train", bs=config.TRAIN_BATCH)

        # 2) PLAN memorije -> AUTO-potvrda ------------------------------------
        plan = C.teacher_mem_plan(teacher, adapter, train_loader, A.model_name(spec))
        GB = C.GB
        steps["plan"] = {
            "light": "green" if plan["cached"] else ("red" if not plan["fits_disk"] else "yellow"),
            "data": {"per_batch_gb": plan["per_batch"] / GB, "total_gb": plan["total"] / GB,
                     "n_batches": plan["n_batches"], "free_disk_gb": plan["free_disk"] / GB,
                     "free_ram_gb": (plan["free_ram"] / GB) if plan["free_ram"] else None,
                     "fits_disk": plan["fits_disk"], "in_ram": plan["in_ram"],
                     "cache_dir": plan["cache_dir"], "cached": plan["cached"],
                     "train_batch": config.TRAIN_BATCH,
                     "vram_gb": (None if plan["vram"] is None else {k: plan["vram"][k] / GB for k in plan["vram"]})}}
        # AUTO-potvrda: green ako je sve uredu, yellow uz objasnjenje ako radi-ali-degradirano, red (stop) ako ne stane.
        gbf = lambda b: b / GB
        if not plan["fits_disk"]:                               # tvrdi blokator -> stop
            steps["plan"]["light"] = "red"
            steps["plan"]["msg"] = (f"NE STANE na disk: treba {gbf(plan['total']):.1f} GB, "
                                    f"slobodno {gbf(plan['free_disk']):.1f} GB — oslobodi prostor")
            write("error"); return
        if plan["cached"]:
            steps["plan"]["light"] = "green"
            steps["plan"]["msg"] = "cache već postoji (valjan) — preskačem precompute"
        elif not plan["in_ram"]:                                # radi, ali sporije (cache ne stane u RAM -> disk-stream)
            ram_s = f"{gbf(plan['free_ram']):.1f} GB" if plan["free_ram"] else "?"
            steps["plan"]["light"] = "yellow"
            steps["plan"]["msg"] = (f"radi, ali sporije: cache {gbf(plan['total']):.1f} GB ne stane u RAM "
                                    f"(slobodno {ram_s}) → disk-stream")
        else:
            steps["plan"]["light"] = "green"
            steps["plan"]["msg"] = f"plan OK ({gbf(plan['total']):.1f} GB, RAM-rezident) — automatski potvrđeno"
        write("preparing")

        # 3) PRECOMPUTE -------------------------------------------------------
        steps["precompute"] = {"light": "yellow", "msg": f"prikupljam {plan['n_batches']} batcheva…"}
        write("preparing")
        C.precompute_teacher(teacher, adapter, train_loader, A.model_name(spec), plan=plan)
        steps["precompute"] = {"light": "green", "msg": f"{plan['n_batches']} batcheva u cacheu"}
        write("preparing")

        # 4) BASELINE PERF (citaj iz cachea kao i kompresija; izmjeri samo ako ga nema — bez dupliciranja Overviewa)
        cached_perf = os.path.exists(A.perf_path(A.model_name(spec)))
        steps["perf"] = {"light": "yellow", "msg": "čitam baseline iz cachea…" if cached_perf else "mjerim baseline…"}
        write("preparing")
        rep = A.baseline_perf(spec, dev, recompute=False, model=model, adapter=adapter)
        v = rep["maps"].get("val", {})
        steps["perf"] = {"light": "green",
                         "data": {"val_map": v.get("map", 0.0), "val_mar": v.get("mar_100", 0.0),
                                  "gflops": rep["gflops"], "params": rep["params"]}}
        write("ready")
    except KeyboardInterrupt:
        write("stopped")
    except BaseException:
        steps.setdefault("err", {})["msg"] = traceback.format_exc()[-1500:]
        write("error")


if __name__ == "__main__":
    main()
