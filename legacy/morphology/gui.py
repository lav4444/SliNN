
import json
import os
import signal
import subprocess
import sys
import time

import streamlit as st

import analysis as A
import config

JOB = os.path.join(config.TMP_ROOT, "gui_job")

METRIC_OPTS = {"val mAP@[.5:.95]": "map", "val mAP@.50": "map_50", "val mAP@.75": "map_75", "val mAR@100": "mar_100"}
METRIC_NICE = {v: k for k, v in METRIC_OPTS.items()}

st.set_page_config(page_title="Morphology — compression", layout="wide")
PAGE = st.sidebar.radio("Stranica", ["Overview", "Compress", "About solution"])
st.sidebar.caption("morphology · lokalni pipeline za kompresiju")
if config.DEV_DATA_SUBSET:
    st.warning(f"⚠️ DEV način: koristi se samo prvih **{config.DEV_DATA_SUBSET}** slika po splitu (niska vjernost). "
               f"Postavi `DEV_DATA_SUBSET = None` u config.py prije pravih runova.")


def page_overview():
    st.title("Overview")
    spec = st.selectbox("Model", ["fasterrcnn", A.YOLO_PATH])
    cache = st.session_state.setdefault("reports", {})
    if spec not in cache:
        bar = st.progress(0.0, text="analiza...")
        cache[spec] = A.analyze_report(spec, "cuda",
                                       progress=lambda f, m: bar.progress(f, text=f"{int(f*100)}% — {m}"))
        bar.empty()
    rep = cache[spec]

    st.markdown(f"**Task:** `{rep.get('task', '?')}`  ·  **Tip modela:** `{rep['kind']}`  ·  **Klase:** {rep['classes']}")
    c = st.columns(4)
    c[0].metric("params", f"{rep['params']/1e6:.2f} M")
    c[1].metric("GFLOPs", f"{rep['gflops']:.3f}")
    c[2].metric("filteri / neuroni", f"{rep['n_filters']} / {rep['n_neurons']}")
    c[3].metric("klase", rep["classes"])
    c2 = st.columns(4)
    c2[0].metric("slojeva", rep["total_layers"])
    c2[1].metric("prunable/growable", rep["prunable"] if rep["prunable"] is not None else "?")
    c2[2].metric("dead (kanali)", rep["dead_total"])
    c2[3].metric("near-dead <1%", rep["near_total"])
    st.caption(f"kind={rep['kind']} · tipovi: " + ", ".join(f"{k}×{v}" for k, v in rep["types"].items())
               + (f" · {rep['struct_note']}" if rep.get("struct_note") else ""))

    perf = rep.get("perf")
    if perf:
        st.subheader("Performanse (mjereno — compress ovo čita kao baseline, bez ponovnog računanja)")
        val = perf["maps"].get("val", {})
        cp = st.columns(4)
        cp[0].metric("val mAP@[.5:.95]", f"{val.get('map', 0):.4f}")
        cp[1].metric("val mAR@100", f"{val.get('mar_100', 0):.4f}")
        cp[2].metric("GPU ms/img", f"{perf['gpu_ms']:.2f}" if perf.get("gpu_ms") else "—")
        cp[3].metric("CPU ms/img", f"{perf['cpu_ms']:.2f}" if perf.get("cpu_ms") else "—")
        with st.expander("Detaljne performanse (sve metrike, po splitu)"):
            keys = []
            for m in perf["maps"].values():
                for k in m:
                    if k not in keys:
                        keys.append(k)
            rows = [{"split": s, "n": perf["n_eval"][s], **{k: round(m.get(k, float("nan")), 4) for k in keys}}
                    for s, m in perf["maps"].items()]
            st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("Per-layer (GFLOPs / prune / grow; crveno = off-limits)")
    for p in rep["plots"]:
        if os.path.exists(p):
            st.image(p, width="stretch")

    col = st.columns(2)
    with col[0]:
        st.subheader("Top-10 PRUNE (najjeftiniji rez)")
        st.dataframe(rep["top_prune"], width="stretch", hide_index=True)
    with col[1]:
        st.subheader("Top-10 GROW (najveći dobitak/FLOP)")
        st.dataframe(rep["top_grow"], width="stretch", hide_index=True)

    with st.expander("Puna per-layer tablica"):
        st.dataframe(rep["layers"], width="stretch", hide_index=True)


def _read_traj():
    pts = []
    p = os.path.join(JOB, "trajectory.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    pts.append(json.loads(line))
    return pts


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _read_status():
    p = os.path.join(JOB, "status.json")
    s = json.load(open(p)) if os.path.exists(p) else {"state": "idle"}
    if s.get("state") == "running" and s.get("pid") and not _pid_alive(s["pid"]):
        s = {"state": "error", "msg": "worker proces nije ziv (pao bez statusa)"}
    return s


@st.cache_resource
def _startup_cleanup():
    s = _read_status()
    alive = s.get("state") == "running" and s.get("pid") and _pid_alive(s["pid"])
    if not alive and os.path.isdir(JOB):
        for f in ("trajectory.jsonl", "status.json", "worker.log"):
            try:
                os.remove(os.path.join(JOB, f))
            except OSError:
                pass
    ps = _read_prep_status()
    prep_alive = ps.get("state") == "preparing" and ps.get("pid") and _pid_alive(ps["pid"])
    if not prep_alive and os.path.isdir(JOB):
        try:
            os.remove(os.path.join(JOB, "prep_status.json"))
        except OSError:
            pass
    return True


_LIGHTS = {"green": "🟢", "yellow": "🟡", "red": "🔴", "gray": "⚪"}
PHASE2_OPTS = {"GFLOPs": "gflops", "Params": "params", "val mAP@[.5:.95]": "map", "val mAR@100": "mar_100"}


def _kill(pid):
    if pid:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass


def _read_prep_status():
    p = os.path.join(JOB, "prep_status.json")
    if not os.path.exists(p):
        return {"state": "idle", "steps": {}}
    try:
        s = json.load(open(p))
    except Exception:
        return {"state": "idle", "steps": {}}
    if s.get("state") == "preparing" and s.get("pid") and not _pid_alive(s["pid"]):
        s["state"] = "error"
    return s


def _render_prep_steps(prep):
    steps = prep.get("steps", {})
    for key, label in [("gpu", "GPU dostupan i slobodan (<50% VRAM)"),
                       ("batch", "Auto batch size (proba VRAM-a)"),
                       ("plan", "Plan memorije (feature/logit cache)"),
                       ("precompute", "Precompute feature/logiti"),
                       ("perf", "Baseline performanse")]:
        s = steps.get(key, {})
        st.markdown(f"{_LIGHTS.get(s.get('light', 'gray'), '⚪')} **{label}**"
                    + (f" — {s['msg']}" if s.get("msg") else ""))
        if key == "plan" and s.get("data"):
            d = s["data"]
            r1 = st.columns(3)
            r1[0].metric("Potrebno ukupno", f"{d['total_gb']:.2f} GB", help=f"{d['n_batches']} batcheva feature/logiti (fp16)")
            r1[1].metric("Disk: koristi / slobodno", f"{d['total_gb']:.2f} / {d['free_disk_gb']:.0f} GB",
                         "stane" if d["fits_disk"] else "NE STANE", delta_color="off" if d["fits_disk"] else "inverse")
            ram_use = d["total_gb"] if d["in_ram"] else 0.0
            r1[2].metric("RAM: koristi / slobodno",
                         (f"{ram_use:.2f} / {d['free_ram_gb']:.1f} GB" if d.get("free_ram_gb") else "—"),
                         "RAM-rezident" if d["in_ram"] else "disk-stream", delta_color="off")
            vr = d.get("vram_gb")
            if vr:
                r2 = st.columns(3)
                r2[0].metric("GPU VRAM ukupno", f"{vr['total']:.1f} GB")
                r2[1].metric("GPU VRAM slobodno", f"{vr['free']:.1f} GB")
                r2[2].metric(f"Vršno za batch={d['train_batch']}", f"{vr['batch_peak']:.2f} GB", "✓ stane", delta_color="off")
                st.caption("Konfigurirani batch je testiran sizing-forwardom i stane u VRAM (nema OOM).")
            st.caption(f"Sve ide na disk: `{d['cache_dir']}`"
                       + (" · radni cache drži se u RAM-u (brže)" if d["in_ram"] else " · čita se s diska (disk-stream)"))
        if key == "batch" and s.get("data"):
            d = s["data"]; cb = st.columns(3)
            cb[0].metric("TRAIN batch", d["train"])
            cb[1].metric("EVAL batch", d["eval"])
            cb[2].metric("GRAD batch", d["grad"])
            st.caption("Najveći batch koji stane uz ≤90% slobodnog VRAM-a (proba punog FT koraka). Sva tri ista — TRAIN je najteži pa EVAL/GRAD sigurno stanu.")
        if key == "perf" and s.get("data"):
            d = s["data"]; cc = st.columns(4)
            cc[0].metric("val mAP@[.5:.95]", f"{d['val_map']:.4f}")
            cc[1].metric("val mAR@100", f"{d['val_mar']:.4f}")
            cc[2].metric("GFLOPs", f"{d['gflops']:.3f}")
            cc[3].metric("params", f"{d['params']/1e6:.2f} M")


@st.fragment(run_every=2)
def _prep_fragment():
    prep = _read_prep_status()
    _render_prep_steps(prep)
    if prep.get("state") != "preparing":
        st.rerun()


def _launch_prep():
    for f in ("prep_status.json",):
        try:
            os.remove(os.path.join(JOB, f))
        except OSError:
            pass
    proc = subprocess.Popen([sys.executable, "-u", os.path.join(os.path.dirname(__file__), "prep_worker.py")],
                            cwd=os.path.dirname(__file__))
    json.dump({"state": "preparing", "pid": proc.pid, "steps": {}}, open(os.path.join(JOB, "prep_status.json"), "w"))


def _render_running_view(cstatus):
    cfg = {}
    cfgp = os.path.join(JOB, "config.json")
    if os.path.exists(cfgp):
        try:
            cfg = json.load(open(cfgp))
        except Exception:
            pass
    perf = _read_prep_status().get("steps", {}).get("perf", {}).get("data", {})
    _init = {"gflops": perf.get("gflops", 0.0), "params": perf.get("params", 0.0),
             "map": perf.get("val_map", 0.0), "mar_100": perf.get("val_mar", 0.0)}

    def _f(k, v):
        return f"{v/1e6:.2f} M" if k == "params" else f"{v:.3f}"

    mets = cfg.get("metrics", ["map"])
    nice = ", ".join(f"{METRIC_NICE.get(m, m)} ({_f(m, _init.get(m, 0.0))})" for m in mets)
    tf = config.FT_RECOVERY_FRAC
    tol = ", ".join(f"{METRIC_NICE.get(m, m)} ≥ {_f(m, _init.get(m, 0.0) * tf)}" for m in mets)
    st.info(f"**Praćene metrike (početno):** {nice}  ·  **Tolerancija:** {tol} ({int(tf*100)}% originala)  \n"
            f"Faza 1: dead/near-dead rez + KD oporavak → Faza 2: kontinuirani prune (+uvjetni grow) do **najmanjeg modela** koji drži toleranciju.")
    if st.button("⏹ Zaustavi trening"):
        _kill(cstatus.get("pid"))
        json.dump({"state": "stopped"}, open(os.path.join(JOB, "status.json"), "w"))
        st.rerun()
    _live_fragment()


def page_compress():
    st.title("Compress")
    os.makedirs(JOB, exist_ok=True)
    st.markdown(f"**Model:** `{config.MODEL_SPEC}` — fiksiran u config.py (nije promjenjiv ovdje).")

    cstatus0 = _read_status()
    if cstatus0.get("state") == "running":
        _render_running_view(cstatus0)
        return

    st.subheader("Priprema")
    prep = _read_prep_status()
    pstate = prep.get("state", "idle")

    if pstate in ("idle", "stopped"):
        _launch_prep()
        st.rerun()

    if pstate == "preparing":
        _prep_fragment()
        return

    _render_prep_steps(prep)

    if pstate == "error":
        st.error("Priprema nije uspjela (vidi terminal). " + str(prep.get("steps", {}).get("err", {}).get("msg", ""))[-700:])
        return
    if pstate != "ready":
        return

    cstatus = _read_status()
    crunning = cstatus.get("state") == "running"
    _def = list(config.FT_METRICS)
    _p2keys = list(PHASE2_OPTS.keys())
    _p2idx = next((i for i, k in enumerate(_p2keys) if PHASE2_OPTS[k] == config.PHASE2_STOP_METRIC), 0)
    perf = prep.get("steps", {}).get("perf", {}).get("data", {})
    _init = {"gflops": perf.get("gflops", 0.0), "params": perf.get("params", 0.0),
             "map": perf.get("val_map", 0.0), "mar_100": perf.get("val_mar", 0.0)}

    def _fmt(k, v):
        return f"{v/1e6:.2f} M" if k == "params" else f"{v:.3f}"

    with st.container(border=True):
        h0 = st.empty()
        quant = st.radio("Ciljana kvantizacija (HW poravnanje kanala)", ["INT8  (M=32)", "FP16  (M=8)"],
                         horizontal=True, key="c0_quant")
        align_m = 32 if quant.startswith("INT8") else 8
        st.caption(f"Poravnava broj kanala na višekratnik **{align_m}** (INT8/CHW32→32, FP16→8) za brži inference na "
                   f"GPU/Jetson/RPi. Min kanala po sloju = M/2 = **{align_m // 2}**.")
        green0 = st.session_state.get("c0_ok") == align_m
        h0.markdown(f"{'🟢' if green0 else '🟡'} **Kvantizacija / poravnanje kanala** (align M)")
        if not green0 and st.button("Potvrdi kvantizaciju", key="c0_btn"):
            st.session_state["c0_ok"] = align_m; st.rerun()

    with st.container(border=True):
        h1 = st.empty()
        opt_map = st.checkbox("val mAP@[.5:.95]", value=("map" in _def or not _def))
        opt_mar = st.checkbox("val mAR@100", value=("mar_100" in _def))
        st.caption("Više metrika = bolja kontrola kvalitete (SVE moraju na ≥ -2% početne), ali duži trening i manja "
                   "vjerojatnost da se takvo rješenje nađe.")
        sel1 = (opt_map, opt_mar)
        green1 = (opt_map or opt_mar) and st.session_state.get("c1_ok") == sel1
        h1.markdown(f"{'🟢' if green1 else '🟡'} **Metrika za praćenje / early-stop** (bar jedna)")
        if not green1 and st.button("Potvrdi metriku", key="c1_btn", disabled=not (opt_map or opt_mar)):
            st.session_state["c1_ok"] = sel1; st.rerun()

    with st.container(border=True):
        h2 = st.empty()
        p2_metric = st.selectbox("Train until", _p2keys, index=_p2idx)
        p2_key = PHASE2_OPTS[p2_metric]
        info_ph = st.empty()
        p2_frac = st.slider("Ciljna vrijednost (% početne)", 5, 95, int(config.PHASE2_STOP_FRAC * 100), step=5)
        init_v = _init[p2_key]; target_v = init_v * p2_frac / 100.0
        info_ph.markdown(f"Početna **{p2_metric}**: `{_fmt(p2_key, init_v)}`  ·  "
                         f"**Train until**: {_fmt(p2_key, init_v)} × {p2_frac}% = `{_fmt(p2_key, target_v)}`")
        st.caption("Samo 1 uvjet; još se ne koristi — sprema se za kasnije.")
        sel2 = (p2_key, p2_frac)
        green2 = st.session_state.get("c2_ok") == sel2
        h2.markdown(f"{'🟢' if green2 else '🟡'} **Uvjet zaustavljanja** (Train until)")
        if not green2 and st.button("Potvrdi uvjet", key="c2_btn"):
            st.session_state["c2_ok"] = sel2; st.rerun()

    st.caption(f"Ostali parametri (val_cap={config.VAL_CAP}, patience={config.FT_PATIENCE}, "
               f"recovery≥{config.FT_RECOVERY_FRAC:.0%}…) iz config.py.")
    if not (green0 and green1 and green2):
        st.info("Potvrdi sve kartice (🟢) da otključaš pokretanje kompresije.")
    start = st.button("Pokreni kompresiju", disabled=not (green0 and green1 and green2), type="primary")
    if start:
        metrics = (["map"] if opt_map else []) + (["mar_100"] if opt_mar else [])
        batch = _read_prep_status().get("steps", {}).get("batch", {}).get("data", {})
        json.dump({"metrics": metrics, "align_m": align_m, "phase2": {"metric": p2_key, "frac": p2_frac / 100},
                   "batch": batch},
                  open(os.path.join(JOB, "config.json"), "w"))
        open(os.path.join(JOB, "trajectory.jsonl"), "w").close()
        proc = subprocess.Popen([sys.executable, "-u", os.path.join(os.path.dirname(__file__), "worker.py")],
                                cwd=os.path.dirname(__file__))
        json.dump({"state": "running", "phase": "launching", "pid": proc.pid},
                  open(os.path.join(JOB, "status.json"), "w"))
        st.rerun()

    pts = _read_traj()
    if pts:
        st.divider()
        st.caption(f"Zadnji run: status **{cstatus.get('state')}**")
        _render_live(pts, cstatus)


def _trend_chart(pts, key, ylabel, color="#4c78a8", tol_val=None):
    import altair as alt
    import pandas as pd
    pairs = [(i, p.get(key)) for i, p in enumerate(pts) if p.get(key) is not None]
    if not pairs:
        return alt.Chart(pd.DataFrame({"korak": [], "value": []})).mark_line().properties(height=240, width="container")
    xs = [i for i, _ in pairs]; vals = [v for _, v in pairs]
    base = vals[0]
    lo, hi = min(vals), max(vals)
    if tol_val is not None:
        lo, hi = min(lo, tol_val), max(hi, tol_val)
    span = hi - lo
    pad = span * 0.1 if span > 0 else (abs(hi) * 0.1 if hi else 1.0)
    df = pd.DataFrame({"korak": xs, "value": vals})
    line = alt.Chart(df).mark_line(point=True, color=color).encode(
        x=alt.X("korak:Q", title="korak"),
        y=alt.Y("value:Q", title=ylabel, scale=alt.Scale(domain=[lo - pad, hi + pad])))
    rule = alt.Chart(pd.DataFrame({"base": [base]})).mark_rule(
        color="#e45756", strokeDash=[6, 4], size=2).encode(y="base:Q")
    chart = line + rule
    if tol_val is not None:
        tol_rule = alt.Chart(pd.DataFrame({"tol": [tol_val]})).mark_rule(
            color="#54a24b", strokeDash=[5, 3], size=2).encode(y="tol:Q")
        chart = chart + tol_rule
    return chart.properties(height=240, width="container")


def _render_live(pts, status):
    running = status.get("state") == "running"
    cfgp = os.path.join(JOB, "config.json")
    mets = ["map"]
    if os.path.exists(cfgp):
        try:
            mets = json.load(open(cfgp)).get("metrics", ["map"]) or ["map"]
        except Exception:
            pass
    if pts:
        base, cur = pts[0], pts[-1]

        def pct(key):
            b = base.get(key); c_ = cur.get(key)
            return (c_ - b) / b * 100 if (b and c_ is not None) else 0.0
        otag = lambda k: " *" if k in mets else ""
        r1 = st.columns(3)
        r1[0].metric("val mAP@[.5:.95]" + otag("map"), f"{cur.get('val_map', 0) or 0:.4f}", f"{pct('val_map'):+.1f}%")
        r1[1].metric("val mAR@100" + otag("mar_100"), f"{cur.get('val_mar', 0) or 0:.4f}", f"{pct('val_mar'):+.1f}%")
        r1[2].metric("params", f"{cur['params']/1e6:.2f} M", f"{pct('params'):+.1f}%", delta_color="inverse")
        r2 = st.columns(3)
        r2[0].metric("GFLOPs", f"{cur['gflops']:.3f}", f"{pct('gflops'):+.1f}%", delta_color="inverse")
        r2[1].metric("GFLOPs freed", f"{cur['gflops_freed']:.3f}")
        r2[2].metric("GFLOPs reinvested", f"{cur['gflops_reinvested']:.3f}")
        r3 = st.columns(3)
        r3[0].metric("size (MB)", f"{cur['size_mb']:.1f}", f"{pct('size_mb'):+.1f}%", delta_color="inverse")
        r3[1].metric("align score", f"{(cur.get('align_score') or 0)*100:.1f}%", f"{pct('align_score'):+.1f}%",
                     help="Prosječna iskoristivost ×M pločice po sloju (100% = svi slojevi poravnati na M). Veće = bolje za int8/fp16.")
        r3[2].metric("Quantization score", "— (placeholder)")

        st.caption(f"početno (baseline) — mAP={base.get('val_map', 0) or 0:.4f} · mAR100={base.get('val_mar', 0) or 0:.4f} · "
                   f"GFLOPs={base['gflops']:.3f} · params={base['params']/1e6:.2f}M  ·  * = optimizirana  ·  🔴 crvena = početno  ·  🟢 zelena = tolerancija (kvaliteta) / dostižni max (align)")
        GRAPHABLE = [("val_map", "val mAP@[.5:.95]"), ("val_mar", "val mAR@100"), ("gflops", "GFLOPs"),
                     ("params", "params"), ("align_score", "align score"), ("size_mb", "size (MB)"),
                     ("gflops_freed", "GFLOPs freed"), ("gflops_reinvested", "GFLOPs reinvested")]
        _def_on = lambda k: (k == "val_map" and "map" in mets) or (k == "val_mar" and "mar_100" in mets) \
            or k in ("gflops", "params", "align_score")
        st.caption("Grafovi za praćenje (odaberi):")
        cbcols = st.columns(4)
        chosen = []
        for i, (k, lbl) in enumerate(GRAPHABLE):
            has = any(p.get(k) is not None for p in pts)
            if cbcols[i % 4].checkbox(lbl, value=_def_on(k), key=f"g_{k}", disabled=not has) and has:
                chosen.append((k, lbl))
        for i in range(0, len(chosen), 2):
            gcols = st.columns(2)
            for j, (k, lbl) in enumerate(chosen[i:i + 2]):
                with gcols[j]:
                    st.markdown(f"**{lbl}**")
                    tol_val = None
                    if (k == "val_map" and "map" in mets) or (k == "val_mar" and "mar_100" in mets):
                        b = next((p.get(k) for p in pts if p.get(k) is not None), None)
                        tol_val = b * config.FT_RECOVERY_FRAC if b is not None else None
                    elif k == "align_score":
                        tol_val = next((p.get("align_best") for p in reversed(pts) if p.get("align_best") is not None), None)
                    st.altair_chart(_trend_chart(pts, k, lbl, tol_val=tol_val))

    st.caption(f"status: **{status.get('state')}** · faza: {status.get('phase', '-')}")
    if status.get("state") == "error":
        st.error(status.get("msg", "")[-1800:])

    logp = os.path.join(JOB, "worker.log")
    if os.path.exists(logp):
        with st.expander("Trening log", expanded=running):
            st.code(open(logp, errors="replace").read()[-6000:] or "(prazno)", language="text")


@st.fragment(run_every=2)
def _live_fragment():
    status = _read_status()
    pts = _read_traj()
    _render_live(pts, status)
    if status.get("state") != "running":
        st.rerun()


def page_about():
    st.title("About solution")
    cap = A.capabilities()
    st.caption("Općenito o rješenju — što je podržano, što ne. Sve se generira dinamički iz registara u kodu "
               "(kd._LOSS · _EVALUATORS · profiles.PROFILES · TASK_METRICS · LAYER_POLICY · PHASES); ne učitava nijedan model.")
    st.markdown(f"**Što radi.** {cap['what']}")
    st.markdown(f"**Trening-loss:** {cap['training_loss']}  \n"
                f"**Optimizer:** {cap['optimizer']}  \n"
                f"**Metrika:** {cap['metric_note']}  \n"
                f"**Budžet:** {cap['budget']}")

    st.subheader("Podržani taskovi")
    for t in cap["tasks"]:
        icon = "🟢" if t["status"] == "full" else "🟡"
        head = ", ".join(t["metrics"].get(k, k) for k in t["headline"]) or "—"
        st.markdown(f"{icon} **{t['task']}** — metrika {'✓' if t['metric'] else '✗'} · "
                    f"profil {'✓' if t['profile'] else '✗'}  ·  glavne metrike: {head}")
        if t["metrics"]:
            with st.expander(f"sve metrike — {t['task']}"):
                for k, lbl in t["metrics"].items():
                    star = " ⭐" if k in t["headline"] else ""
                    st.markdown(f"- `{k}` — {lbl}{star}")
    st.caption("🟢 pun (ima evaluator + profil) · 🟡 djelomičan · task bez evaluatora i profila se ne prikazuje (nepodržan).")

    st.subheader("Validirane arhitekture (profili)")
    st.caption("Konkretne mreže na kojima je generalni pristup proveden — sve izvedeno iz profila (bez ručne proze).")
    for p in cap["profiles"]:
        st.markdown(f"- **{p['kind']}** ({p['task']}, imgsz {p['imgsz']})  \n"
                    f"   KD tapovi: {', '.join(p['kd_types'])}  ·  off-limits: {p['protect']}")

    st.subheader("KD tapovi (čime distiliramo)")
    for d in cap["kd_types"]:
        st.markdown(f"- **{d['type']}** — {d['doc']}")

    st.subheader("Layer-role policy")
    st.caption("Dvije nezavisne osi: STRUKTURNO (prune/grow ↔ off-limits) i TRENING (trainable ↔ frozen).")
    st.table([{"tip sloja": r["tip"], "strukturno": r["strukturno"], "trening": r["trening"], "razlog": r["razlog"]}
              for r in cap["layer_policy"]])

    st.subheader("Faze")
    st.markdown("**Implementirano:** " + ", ".join(cap["phases"]["done"]))
    st.markdown("**Planirano:** " + ", ".join(cap["phases"]["planned"]))


_startup_cleanup()
PAGES = {"Overview": page_overview, "Compress": page_compress, "About solution": page_about}
PAGES[PAGE]()
