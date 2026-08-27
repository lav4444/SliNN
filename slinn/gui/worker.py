"""slinn/gui/worker.py — 6.1 engine-vodjena KOMPRESIJA. BEZ argv; čita tmp/gui_job/config.json
{model_path, dataset_path, target_frac?, ft_steps?, max_steps?, metric_tol?, dead?, batch?}.
Vrti `engine.full_cycle` (KD-only prune+grow, TASK-UVJETNI enhaneri, PER-TASK real-metric gate) i piše
trajektoriju u trajectory.jsonl + stanje u status.json (GUI tail-a). Best model -> tmp/gui_job/compressed.pt.
morphology se NE dira — samo read-only decode plug (mAP) preko backend.build_metric_fn.
"""
import copy
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backend as B                                          # noqa: E402
import torch                                                 # noqa: E402
import engine as E                                           # noqa: E402
import settings as _CFG                                      # noqa: E402

CFG_STEP_FRAC = _CFG.PHASE2_PRUNE_STEP_FRAC                  # cilj REZA po koraku (za ispis bilance)


def _g(v):
    """GFLOPs citljivo bez obzira na red velicine (yolo ~6, housing ~0.0002)."""
    return "{:.4f}".format(v) if v >= 0.01 else "{:.3e}".format(v)


# =========================== FAZE 1 i 2 (nezavisni izbori, mogu oba) =========================== #
# `cfg["phase"]` je STRING zastavica: "1", "2" ili "12". Kad su obje, Faza 2 krece od izlaza Faze 1;
# kad je SAMO Faza 2, krece od ORIGINALA — odluka dolazi iz configa, NIKAD iz "postoji li datoteka na
# disku" (stari best_quality_model.pt iz ranijeg runa bi inace tiho postao ulaz).

def _setup(cfg, dev):
    """Zajednicka priprema za obje faze — radi se JEDNOM i dijeli, pa lancani run ne ucitava model,
    ne materijalizira batcheve i ne mjeri baseline dvaput."""
    if cfg.get("align_m"):
        import settings as CFG
        CFG.ALIGN_M = int(cfg["align_m"])
        CFG.PHASE2_MIN_ALIVE = CFG.ALIGN_M // 2
    model, adapter, ctx = B.load_ctx(cfg, dev)
    teacher = B.frozen_teacher(model, dev)                   # ucitelj je UVIJEK original
    bs = int((cfg.get("batch") or {}).get("train", 8))
    mfn, mname, monfn = B.build_metric_fn(ctx, model, adapter, cfg["dataset_path"], dev,
                                          n_gate=None, monitor_frac=_CFG.METRIC_MONITOR_FRAC,
                                          teacher=teacher)
    batches, _ = E.materialize_train_batches(cfg["dataset_path"], adapter, dev, ctx["split_plan"], bs, None, 0)
    loss_fn = None
    try:
        import enhancers as _ENH
        loss_fn = _ENH.enhancer_loss_fn(ctx, teacher)        # task-uvjetni enhaneri (detekcija)
    except BaseException:
        pass
    cache = (None if loss_fn else
             E.precompute_teacher(teacher, adapter, batches, ctx["taps"], os.path.basename(cfg["model_path"]),
                                  split="train"))
    return {"cfg": cfg, "dev": dev, "model": model, "adapter": adapter, "ctx": ctx, "teacher": teacher,
            "bs": bs, "mfn": mfn, "mname": mname, "monfn": monfn, "batches": batches,
            "cache": cache, "loss_fn": loss_fn, "name": os.path.basename(cfg["model_path"])}


def main_phases():
    """Orkestrator: Faza 1 i/ili Faza 2 prema `cfg["phase"]`."""
    os.makedirs(B.JOB, exist_ok=True)
    cfg0 = json.load(open(os.path.join(B.JOB, "config.json")))
    # JEDAN FOLDER PO KOMPRESIJI. `status.json` ostaje u JOB jer GUI anketira fiksnu putanju; sve
    # STO RUN PROIZVEDE (log, trajektorija, checkpointi, manifest) zivi u run-folderu i nista se ne
    # dijeli s drugim runovima. Originalni model se ne kopira — `run_meta.json` cuva putanju.
    RUN = B.new_run_dir(cfg0["model_path"])
    B.tee_log("worker.log", out_dir=RUN)
    traj_p = os.path.join(RUN, "trajectory.jsonl")
    status = os.path.join(B.JOB, "status.json")
    open(traj_p, "w").close()
    loop = {"n": 1}                                          # koja faza pise u trajektoriju

    def set_status(d):
        d["pid"] = os.getpid()
        d["run_dir"] = RUN                                   # GUI odavde cita log i trajektoriju
        json.dump(d, open(status, "w"))

    def write(rec):
        rec["loop"] = loop["n"]                              # GUI zna iz koje faze je tocka
        with open(traj_p, "a") as f:
            f.write(json.dumps(rec) + "\n")
        set_status({"state": "running", "loop": loop["n"], "phase": rec.get("phase"),
                    "step": rec["step"], "gflops": rec["gflops"], "metric": rec.get("metric")})

    def on_batch(i, nb, mean):                               # epoha na punom splitu traje -> javi napredak
        if nb >= 10 and i % max(nb // 5, 1) == 0 and i < nb:
            print("        ... epoha {:>3}/{} · KD {:.4f}".format(i, nb, mean), flush=True)

    set_status({"state": "running", "phase": "start", "step": -1})
    try:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = cfg0
        ph = str(cfg.get("phase") or "1")
        do1, do2 = "1" in ph, "2" in ph
        tol = float(cfg.get("metric_tol", _CFG.FT_RECOVERY_FRAC))

        print("=" * 78)
        print("izlazi ovog runa: {}".format(RUN))
        print("SliNN · {} · {}".format(
            " + ".join([x for x in ["FAZA 1", "FAZA 2"] if (x.endswith("1") and do1) or (x.endswith("2") and do2)]),
            os.path.basename(cfg["model_path"])))
        print("  priprema (model, adapter, task, batchevi, teacher cache, metrika)...", flush=True)
        sh = _setup(cfg, dev)
        mfn, monfn, mname = sh["mfn"], sh["monfn"], sh["mname"]
        base_full = float(mfn(sh["model"])) if mfn is not None else None
        base_mon = float(monfn(sh["model"])) if monfn is not None else None
        print("  task={} · metrika={} · tolerancija={:.0%} · batch={} · epoha={} batcheva".format(
            sh["ctx"]["task"], mname, tol, sh["bs"], len(sh["batches"])))
        if base_full is not None:
            print("  baseline {}={:.4f} -> prag {:.4f}".format(mname, base_full, tol * base_full))
            if base_mon is not None:
                print("  monitor  {}={:.4f} -> prag {:.4f}  (fiksni podskup {:.0%} val skupa)".format(
                    mname, base_mon, tol * base_mon, _CFG.METRIC_MONITOR_FRAC))
        else:
            print("  UPOZORENJE: nema NIKAKVE metrike kvalitete (ni task-metrike ni ucitelja) — "
                  "kompresija radi, ali nista ne cuva kvalitetu.")
        if mname == "teacher_agreement":
            print("  napomena: slaganje krece od 1.0 (student JE ucitelj) i mjeri UDIO ulaza koji dobiju "
                  "isti odgovor — nije isto sto i mAP/f1. Mjeri se na VAL splitu, kao i sve ostalo.")
        print("=" * 78)

        student = copy.deepcopy(sh["model"]).to(dev)
        common = dict(loss_fn=sh["loss_fn"], batches=sh["batches"], cache=sh["cache"],
                      batch_size=sh["bs"], on_batch=on_batch)

        # ---------------------------------------------------------------- FAZA 1
        if do1:
            loop["n"] = 1
            print("\n### FAZA 1 — najmanji model iznad praga")
            print("  granice oporavka: max {} FT epoha / patience {} (PO EPIZODI)".format(
                _CFG.F1_FT_MAX_EPOCHS, _CFG.F1_FT_PATIENCE))
            print("{:>4}  {:>5}  {:>9}  {:>10}  {:>8}  {:>8}  {:>5}  {:>7}".format(
                "kor", "mod", "GFLOPs", "params", "monitor", "puna", "rez", "KD"))
            print("-" * 78)
            st1 = {"g0": None}

            def on_step1(rec):
                write(rec)
                if st1["g0"] is None:
                    st1["g0"] = rec["gflops"]
                cut = 100.0 * (st1["g0"] - rec["gflops"]) / st1["g0"] if st1["g0"] else 0.0
                mode = {"baseline": "base", "morph": "MORPH", "ft": "FT"}.get(rec.get("phase"), "?")
                print("{:>4}  {:>5}  {:>9}  {:>10,}  {:>8}  {:>8}  {:>4.1f}%  {:>7}{}".format(
                    rec["step"], mode, _g(rec["gflops"]), rec["params"],
                    "{:.4f}".format(rec["monitor"]) if rec.get("monitor") is not None else "—",
                    "{:.4f}".format(rec["metric_full"]) if rec.get("metric_full") is not None else "—",
                    cut, "{:.4f}".format(rec["kd"]) if rec.get("kd") is not None else "—",
                    "  <- NAJBOLJI" if rec.get("is_best") and rec["step"] else ""))
                if rec.get("phase") == "morph" and rec.get("act_freed") is not None:
                    print("        rez −{:.4f} ({:.0f}% cilja, {} krug(a)) · {} kanala{}".format(
                        rec["act_freed"], 100 * rec["act_freed"] / (rec["step_target"] or 1),
                        rec["prune_rounds"], rec.get("removed_ch", 0),
                        (" · naraslo " + ", ".join("{} +{}".format(".".join(str(n).split(".")[-2:]), k)
                                                   for n, k in rec["grown"])) if rec.get("grown") else ""))
                elif rec.get("phase") == "ft":
                    print("        oporavak: {}/{} epoha · bez novog najboljeg {}/{}".format(
                        rec.get("ft_used", 0), _CFG.F1_FT_MAX_EPOCHS,
                        rec.get("no_imp", 0), _CFG.F1_FT_PATIENCE))

            r1 = E.run_phase1(student, sh["teacher"], sh["adapter"], dev, sh["ctx"], cfg["dataset_path"],
                              sh["name"], metric_fn=mfn, monitor_fn=monfn, metric_baseline=base_full,
                              monitor_baseline=base_mon, metric_tol=tol,
                              max_steps=int(cfg.get("max_steps", 200)), on_step=on_step1, **common)
            student = r1["model"]
            out = os.path.join(RUN, "best_quality_model.pt")
            torch.save(student, out)                         # full-eager
            print("-" * 78)
            print("FAZA 1 GOTOVA · razlog: {}".format(r1["reason"]))
            print("  isporuceno: korak {} · GFLOPs {} -> {} ({:.1f}% rez)".format(
                r1["step"], _g(r1["g0"]), _g(r1["gflops"]),
                100.0 * (r1["g0"] - r1["gflops"]) / r1["g0"] if r1["g0"] else 0.0))
            if r1.get("metric") is not None:
                print("  {} = {:.4f} (prag {:.4f}) — POTVRDENO na punom val skupu".format(
                    mname, r1["metric"], r1["floor_full"]))
            if r1["step"] == 0:
                print("  NIJEDAN rez nije prosao prag -> isporucen je POCETNI model (garancija Faze 1).")
            print("  spremljeno: {}".format(out))

        # ---------------------------------------------------------------- FAZA 2
        if do2:
            loop["n"] = 2
            src = "izlaz Faze 1" if do1 else "ORIGINAL (Faza 1 nije trazena)"
            print("\n### FAZA 2 — ljestvica do strukturnog minimuma")
            print("  start: {} · {} GFLOPs".format(src, _g(E.gflops(student, sh["adapter"], dev))))
            print("  {} verzija · grow {:.0%} (OFF na zadnjoj precki) · FT max {} / patience {}".format(
                _CFG.F2_CHECKPOINTS, _CFG.PHASE2_REINVEST_FRAC,
                _CFG.F2_FT_MAX_EPOCHS, _CFG.F2_FT_PATIENCE))
            print("  mjerim strukturni minimum (probni rez svih slojeva na floor)...", flush=True)
            st2 = {"hdr": False}

            def on_probe(k, n, nm):
                if n >= 10 and k % max(n // 4, 1) == 0:
                    print("        ... proba {:>3}/{}".format(k, n), flush=True)

            def on_step2(rec):
                write(rec)
                if rec.get("phase") == "checkpoint":
                    print("  >>> ckpt_{} · {} GFLOPs · {}={} · cilj {} {}".format(
                        rec["rung"], _g(rec["gflops"]), mname,
                        "{:.4f}".format(rec["metric"]) if rec.get("metric") is not None else "—",
                        _g(rec["target"] or 0), "DOSEGNUT" if rec.get("reached") else "PROMASEN"))
                    return
                if not st2["hdr"]:
                    st2["hdr"] = True
                    print("{:>6}  {:>6}  {:>9}  {:>10}  {:>8}  {:>9}  {:>7}".format(
                        "kor", "precka", "GFLOPs", "params", "metrika", "cilj", "KD"))
                    print("-" * 78)
                # FT epohe unutar iste precke dijele broj koraka -> dopisi redni broj epohe (74.3),
                # inace log izgleda kao da se vrti u mjestu.
                _e = rec.get("ft_epoch")
                _lbl = "{}.{}".format(rec["step"], _e) if _e and _e > 1 else str(rec["step"])
                print("{:>6}  {:>6}  {:>9}  {:>10,}  {:>8}  {:>9}  {:>7}".format(
                    _lbl, rec.get("rung", 0), _g(rec["gflops"]), rec["params"],
                    "{:.4f}".format(rec["metric"]) if rec.get("metric") is not None else "—",
                    _g(rec["target"]) if rec.get("target") else "—",
                    "{:.4f}".format(rec["kd"]) if rec.get("kd") is not None else "—"))
                if rec.get("phase") == "morph" and rec.get("act_freed") is not None:
                    print("        rez −{:.4f} ({:.0f}% cilja, {} krug(a)) · {} kanala".format(
                        rec["act_freed"], 100 * rec["act_freed"] / (rec["step_target"] or 1),
                        rec["prune_rounds"], rec.get("removed_ch", 0)))

            r2 = E.run_phase2(student, sh["teacher"], sh["adapter"], dev, sh["ctx"], cfg["dataset_path"],
                              sh["name"], metric_fn=mfn, monitor_fn=monfn, out_dir=RUN,
                              step_frac=_CFG.F2_PRUNE_STEP_FRAC,   # Faza 2 reze krupnije (v. settings)
                              on_step=on_step2, on_probe=on_probe, **common)
            print("-" * 78)
            print("FAZA 2 GOTOVA · {} -> {} GFLOPs (izmjereno dno)".format(
                _g(r2["g_start"]), _g(r2["g_min"])))
            if r2.get("exhausted"):
                print("  PREKID: {}".format(r2["exhausted"]))
            print("  {:<10}{:>10}{:>10}{:>12}{:>10}".format("verzija", "GFLOPs", "% starta", mname, "cilj"))
            for c in r2["checkpoints"]:
                print("  {:<10}{:>10}{:>9.1f}%{:>12}{:>10}".format(
                    "ckpt_{}".format(c["i"]), _g(c["gflops"]), 100 * c["gflops"] / r2["g_start"],
                    "{:.4f}".format(c["metric"]) if c.get("metric") is not None else "—",
                    "OK" if c["reached"] else "promasen"))
            print("  manifest: {}".format(r2["manifest"]))

        produced = sorted(f for f in os.listdir(RUN) if f.endswith((".pt", ".json", ".jsonl", ".log")))
        json.dump({"model_path": cfg["model_path"], "dataset_path": cfg["dataset_path"],
                   "code_dirs": cfg.get("code_dirs"), "phase": ph, "metric_name": mname,
                   "metric_baseline": base_full, "monitor_baseline": base_mon, "metric_tol": tol,
                   "config": cfg, "produced": produced},
                  open(os.path.join(RUN, "run_meta.json"), "w"), indent=1)
        print("=" * 78)
        print("izlazi: {}".format(RUN))
        for f in produced:
            print("   {}".format(f))
        print("=" * 78)
        set_status({"state": "done", "phase": "final", "metric_name": mname,
                    "did": ph, "metric_baseline": base_full})
    except KeyboardInterrupt:
        set_status({"state": "stopped"})
    except BaseException:
        set_status({"state": "error", "msg": traceback.format_exc()[-2500:]})



def main():
    os.makedirs(B.JOB, exist_ok=True)
    B.tee_log("worker.log")                                  # ispis ide u terminal I u log (GUI ga tail-a)
    traj = os.path.join(B.JOB, "trajectory.jsonl")
    status = os.path.join(B.JOB, "status.json")
    open(traj, "w").close()

    def set_status(d):
        d["pid"] = os.getpid()
        json.dump(d, open(status, "w"))

    state = {"g0": None, "mname": "metrika", "floor": None,
             "prev_g": None, "cum_pruned": 0.0, "cum_grown": 0.0}

    def on_step(rec):                                        # trajektorija-točka (isti dict kao morph_loop rec)
        with open(traj, "a") as f:
            f.write(json.dumps(rec) + "\n")
        set_status({"state": "running", "phase": "morph", "step": rec["step"],
                    "gflops": rec["gflops"], "metric": rec.get("metric")})

        # --- ispis napretka (ide u terminal I u worker.log; GUI ga prikazuje) ---
        if state["g0"] is None:                              # prva tocka = baseline
            state["g0"] = state["prev_g"] = rec["gflops"]
            print("{:>4}  {:>9}  {:>10}  {:>9}  {:>7}  {:>6}  {:>5}  {:>8}".format(
                "kor", "GFLOPs", "params", state["mname"], "align", "MB", "rez", "KD"))
            print("-" * 74)
        cut = 100.0 * (state["g0"] - rec["gflops"]) / state["g0"] if state["g0"] else 0.0
        met = rec.get("metric")
        flag = ""
        if met is not None and state["floor"] is not None:
            flag = "  <-- ISPOD praga" if met < state["floor"] else ""
        print("{:>4}  {:>9}  {:>10,}  {:>9}  {:>7.3f}  {:>6.2f}  {:>4.1f}%  {:>8}{}".format(
            rec["step"], _g(rec["gflops"]), rec["params"],
            "{:.4f}".format(met) if met is not None else "—",
            rec.get("align_score") or 0.0, rec.get("size_mb") or 0.0, cut,
            "{:.4f}".format(rec["kd"]) if rec.get("kd") is not None else "—", flag))
        # --- BILANCA GFLOPs po koraku (rec nosi KUMULATIVNE zbrojeve, ovdje se rade prirasti) ---
        # `rez` gore je kumulativan i NETO; ovdje se vidi zasto neto nije jednak cilju od 1.5%:
        # PHASE2_PRUNE_STEP_FRAC je cilj REZA, a grow vrati do PHASE2_REINVEST_FRAC oslobodjenog.
        if rec["step"]:
            cp, cg = rec.get("gflops_freed") or 0.0, rec.get("gflops_reinvested") or 0.0
            d_pruned, d_grown = cp - state["cum_pruned"], cg - state["cum_grown"]
            state["cum_pruned"], state["cum_grown"] = cp, cg
            net = (state["prev_g"] or rec["gflops"]) - rec["gflops"]
            state["prev_g"] = rec["gflops"]
            g0 = state["g0"] or 1.0
            print("      GFLOPs: −{:.4f} rez  +{:.4f} rast  =  −{:.4f} neto "
                  "({:.2f}% od originala; cilj reza {:.2f}%)".format(
                      d_pruned, d_grown, net, 100 * net / g0, 100 * CFG_STEP_FRAC))
            tg, est = rec.get("step_target"), rec.get("est_freed")
            act, rnd = rec.get("act_freed"), rec.get("prune_rounds")
            r1 = rec.get("r1_freed") or 0.0
            if tg is not None:                               # cilj -> stvarni rez (zatvorena petlja, 6.11)
                lab = "cilj pogoden" if act >= 0.98 * tg else "MANJAK — kandidati iscrpljeni"
                print("      plan:   cilj {:.4f} -> stvarno {:.4f} ({:.0f}% cilja) u {} krug(a)  · {}".format(
                    tg, act, 100 * act / tg if tg else 0, rnd, lab))
                if est:                                      # 1. krug = sto bi stari jednoprolazni kod dao
                    print("      cost:   1. krug procijenio {:.4f} -> isporucio {:.4f} ({:.0f}% procjene); "
                          "doplan nadoknadio {:.4f}".format(est, r1, 100 * r1 / est if est else 0,
                                                            max(act - r1, 0.0)))

        grown = rec.get("grown") or []
        if rec.get("removed_ch") or grown:
            det = "      rezano {} kanala".format(rec.get("removed_ch", 0))
            if grown:                                        # skrati ime sloja na zadnja 2 dijela
                det += " · naraslo " + ", ".join(
                    "{} +{}".format(".".join(str(n).split(".")[-2:]), k) for n, k in grown)
            if rec.get("banned"):
                det += " · zabranjenih slojeva {}".format(rec["banned"])
            print(det)

    set_status({"state": "running", "phase": "start", "step": -1})
    try:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.load(open(os.path.join(B.JOB, "config.json")))
        if cfg.get("align_m"):                               # GUI kartica kvantizacije: INT8->32, FP16->8.
            import settings as CFG                           # morph cita ALIGN_M/MIN_ALIVE DINAMICKI
            CFG.ALIGN_M = int(cfg["align_m"])                # -> override mora ici PRIJE full_cycle
            CFG.PHASE2_MIN_ALIVE = CFG.ALIGN_M // 2          # min kanala/sloj = pola plocice (vezano uz M)
        model, adapter, ctx = B.load_ctx(cfg, dev)
        teacher = B.frozen_teacher(model, dev)
        student = copy.deepcopy(model).to(dev)
        bs = int((cfg.get("batch") or {}).get("train", 8))
        # NISTA se ne pita: KD uzima CIJELI train split (n_batches=None), metrika mjeri na CIJELOM
        # val splitu (n_gate=None). Kao morphology. Jedino sto se mjeri je stane li cache na disk.
        mfn, mname, _mon = B.build_metric_fn(ctx, model, adapter, cfg["dataset_path"], dev, n_gate=None)
        state["mname"] = mname

        tol = float(cfg.get("metric_tol", 0.90))
        print("=" * 74)
        print("SliNN kompresija · {}".format(os.path.basename(cfg["model_path"])))
        print("  task={} · mode={} · enhaneri={} · metrika={}".format(
            ctx["task"], ctx.get("mode"), "da" if ctx.get("enhancers") else "ne", mname))
        print("  cilj={:.0%} GFLOPs · tolerancija={:.0%} · batch={} · FT={} · max koraka={}".format(
            float(cfg.get("target_frac", 0.15)), tol, bs,
            cfg.get("ft_steps", 6), cfg.get("max_steps", 200)))
        print("  KD = CIJELI train split · metrika = CIJELI val split (nista se ne kapira)")
        if cfg.get("align_m"):
            print("  poravnanje M={} (min kanala/sloj {})".format(cfg["align_m"], int(cfg["align_m"]) // 2))
        if mfn is not None:
            base = mfn(model)
            state["floor"] = tol * base
            print("  baseline {}={:.4f} -> prag {:.4f}".format(mname, base, state["floor"]))
        else:
            print("  bez task-metrike -> gate = slaganje s uciteljem")
        print("=" * 74)

        res = E.full_cycle(student, teacher, adapter, dev, ctx, cfg["dataset_path"],
                           os.path.basename(cfg["model_path"]),
                           target_frac=float(cfg.get("target_frac", 0.15)),
                           ft_steps=int(cfg.get("ft_steps", 6)), max_steps=int(cfg.get("max_steps", 200)),
                           batch_size=bs, dead=bool(cfg.get("dead", False)),
                           on_step=on_step, metric_fn=mfn, metric_tol=float(cfg.get("metric_tol", 0.90)))
        best = res["best_model"] if res.get("best_model") is not None else res["student"]
        out = os.path.join(B.JOB, "compressed.pt")
        torch.save(best, out)
        g0, bg, fg = res.get("g0"), res.get("best_gflops"), res.get("final_gflops")
        print("=" * 74)
        print("GOTOVO · najbolji korak {} · GFLOPs {} -> {} ({:.1f}% rez)".format(
            res.get("best_step"), _g(g0 or 0), _g(bg or 0),
            100.0 * (g0 - bg) / g0 if (g0 and bg) else 0.0))
        cp, cg = state["cum_pruned"], state["cum_grown"]
        print("  bilanca GFLOPs: −{:.4f} rezano  +{:.4f} naraslo  =  −{:.4f} neto".format(
            cp, cg, cp - cg))
        if cp:
            print("  rast je vratio {:.1f}% oslobodjenog (granica PHASE2_REINVEST_FRAC = {:.0f}%)".format(
                100 * cg / cp, 100 * _CFG.PHASE2_REINVEST_FRAC))
        if fg is not None and bg is not None and abs(fg - bg) > 1e-9:
            print("  petlja je isla dalje do {} GFLOPs, ali je tamo probila prag kvalitete —".format(_g(fg)))
            print("  zato se uzima korak {}, a ne zadnji.".format(res.get("best_step")))
        print("  spremljeno: {}".format(out))
        print("=" * 74)
        set_status({"state": "done", "phase": "final", "metric_name": mname,
                    "best_step": res.get("best_step"), "best_gflops": res.get("best_gflops"),
                    "final_gflops": res.get("final_gflops"), "g0": res.get("g0"),
                    "metric_baseline": res.get("metric_baseline")})
    except KeyboardInterrupt:
        set_status({"state": "stopped"})
    except BaseException:
        set_status({"state": "error", "msg": traceback.format_exc()[-2500:]})


if __name__ == "__main__":
    # `phase` u config.json bira petlju: 1 = nova Faza 1 (PLAN_KOMPRESIJA), inace stara `full_cycle`.
    # Stara ostaje dok Faza 1 ne prodje pun run — ne rusimo radni put prije nego novi dokaze da radi.
    _cfg_p = os.path.join(B.JOB, "config.json")
    _phase = json.load(open(_cfg_p)).get("phase") if os.path.exists(_cfg_p) else None
    (main_phases if _phase else main)()      # bez `phase` -> stara `full_cycle` petlja (vise nije u GUI-ju)
