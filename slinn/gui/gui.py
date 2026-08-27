"""slinn/gui/gui.py — lokalni GUI (Streamlit). TANKI sloj: sav compute je u jezgri, ovdje samo render.

Pokretanje:  streamlit run slinn/gui/gui.py

RAZLIKA OD MORPHOLOGY GUI-ja (6.2): nema biraca modela ("fasterrcnn"/yolo) ni hardkodiranih mAP/mAR
kartica. Ulaz su PUTANJE (model + dataset), a task, metrika, tapovi, format dataseta i enhaneri se
AUTO-detektiraju (`pipeline.prepare`). Ime metrike dolazi iz taska, pa isti ekran radi za regresiju,
segmentaciju, klasifikaciju i detekciju.

Stranice:
  Overview  — analiza UCITANOG modela (generic: velicina, sposobnosti, per-layer, trosak reza, poravnanje)
  Compress  — priprema (semafor) -> postavke -> kompresija + zivi prikaz trajektorije
  About     — sto rjesenje podrzava; sve iz registara (SUPPORTED_TASKS / DATASET_FORMATS / LAYER_REGISTER)
"""

import json
import os
import signal
import subprocess
import sys

import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_SLINN = os.path.dirname(_HERE)
for _p in (_SLINN, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import settings as CFG                                       # noqa: E402
import backend as B                                          # noqa: E402

JOB = B.JOB                                                  # radni prostor GUI-ja (config, status)


def _run_dir():
    """Folder izlaza TEKUCEG (ili zadnjeg) runa. Worker ga javlja kroz `status.json`; ako ga nema,
    padni na JOB (stari runovi, prije uvodjenja runs/)."""
    try:
        d = json.load(open(os.path.join(JOB, "status.json"))).get("run_dir")
        if d and os.path.isdir(d):
            return d
    except BaseException:
        pass
    return JOB
LIGHTS = {"green": "🟢", "yellow": "🟡", "red": "🔴", "gray": "⚪"}

st.set_page_config(page_title="SliNN — kompresija", layout="wide")


# =========================== job IO =========================== #
def _read(name, default):
    p = os.path.join(JOB, name)
    if not os.path.exists(p):
        return default
    try:
        return json.load(open(p))
    except Exception:
        return default


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)                                 # signal 0 = samo provjera postojanja
        return True
    except Exception:
        return False


def _read_status():
    s = _read("status.json", {"state": "idle"})
    if s.get("state") == "running" and s.get("pid") and not _pid_alive(s["pid"]):
        return {"state": "error", "msg": "worker proces nije živ (pao bez statusa)"}
    return s


def _read_prep():
    s = _read("prep_status.json", {"state": "idle", "steps": {}})
    if s.get("state") == "preparing" and s.get("pid") and not _pid_alive(s["pid"]):
        s["state"] = "error"
    return s


def _read_traj():
    pts, p = [], os.path.join(_run_dir(), "trajectory.jsonl")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    pts.append(json.loads(line))
                except Exception:
                    pass
    return pts


def _kill(pid):
    try:
        os.kill(int(pid), signal.SIGTERM)
    except Exception:
        pass


def _launch(script, status_name, payload):
    for f in (status_name, "trajectory.jsonl"):
        try:
            os.remove(os.path.join(JOB, f))
        except OSError:
            pass
    proc = subprocess.Popen([sys.executable, "-u", os.path.join(_HERE, script)], cwd=_HERE)
    payload["pid"] = proc.pid
    json.dump(payload, open(os.path.join(JOB, status_name), "w"))


@st.cache_resource                                           # tocno jednom po pokretanju servera
def _startup_cleanup():
    st_ = _read_status()
    if not (st_.get("state") == "running" and st_.get("pid") and _pid_alive(st_["pid"])):
        for f in ("trajectory.jsonl", "status.json", "worker.log"):
            try:
                os.remove(os.path.join(JOB, f))
            except OSError:
                pass
    ps = _read_prep()
    if not (ps.get("state") == "preparing" and ps.get("pid") and _pid_alive(ps["pid"])):
        for f in ("prep_status.json", "prep.log"):
            try:
                os.remove(os.path.join(JOB, f))
            except OSError:
                pass
    return True


# =========================== zajednicki unos (putanje) =========================== #
def _paths_input():
    """Ulaz = PUTANJE. Nema popisa modela: jezgra je agnosticna, pa i unos mora biti."""
    st.sidebar.markdown("### Ulaz")
    mp = st.sidebar.text_input("Putanja modela (.pt, pun eager modul)",
                               value=st.session_state.get("model_path", ""),
                               placeholder="/home/.../model.pt")
    dp = st.sidebar.text_input("Putanja dataseta (korijen)",
                               value=st.session_state.get("dataset_path", ""),
                               placeholder="/home/.../dataset")
    cd = st.sidebar.text_input("Dodatne kod-putanje (za eager reload klase)",
                               value=st.session_state.get("code_dirs", ""),
                               placeholder="opcionalno, zarezom odvojeno")
    st.session_state.update({"model_path": mp.strip(), "dataset_path": dp.strip(),
                             "code_dirs": cd.strip()})
    ok = bool(mp.strip()) and os.path.exists(mp.strip())
    if mp.strip() and not ok:
        st.sidebar.error("Model nije pronađen na toj putanji.")
    if dp.strip() and not os.path.exists(dp.strip()):
        st.sidebar.warning("Dataset nije pronađen — task/metrika će pasti na KD-only.")
    return ok


def _cfg_payload():
    cds = [c.strip() for c in st.session_state.get("code_dirs", "").split(",") if c.strip()]
    return {"model_path": st.session_state.get("model_path", ""),
            "dataset_path": st.session_state.get("dataset_path", ""),
            "code_dirs": cds or None}


# =========================== OVERVIEW =========================== #
def page_overview():
    st.title("Overview")
    if not st.session_state.get("model_path"):
        st.info("Upiši putanju modela i dataseta u lijevoj traci.")
        return

    key = (st.session_state["model_path"], st.session_state["dataset_path"])
    cache = st.session_state.setdefault("ov_cache", {})
    if key not in cache:
        if not st.button("Analiziraj model", type="primary"):
            return
        bar = st.progress(0.0, text="analiza…")
        try:
            import torch
            import introspect
            import overview as OV
            from classify import probe_adapter
            from pipeline import prepare
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cfg = _cfg_payload()
            model = introspect.load_any(cfg["model_path"], dev, code_dirs=cfg["code_dirs"])
            adapter = probe_adapter(model, dev, verbose=False)
            ctx = prepare(model, adapter, dev, cfg["dataset_path"]) if cfg["dataset_path"] else None
            cache[key] = OV.report(model, adapter, dev, ctx,
                                   progress=lambda f, m: bar.progress(f, text=f"{int(f*100)}% — {m}"))
        except BaseException as e:
            bar.empty()
            st.error(f"Analiza nije uspjela: {type(e).__name__}: {e}")
            return
        bar.empty()
    rep = cache[key]
    s = rep["summary"]

    c = st.columns(4)
    c[0].metric("parametri", f"{s['params']/1e6:.3f} M")
    c[1].metric("GFLOPs", f"{s['gflops']:.4f}")
    c[2].metric("veličina (MB)", f"{s['size_mb']:.1f}")
    c[3].metric(f"poravnanje ×{s['align_m']}", f"{s['align_score']*100:.1f}%",
                help="Prosječna iskoristivost ×M pločice. 100% = svi slojevi poravnati (brži int8/fp16).")

    st.subheader("Što pipeline zna na OVOM modelu")
    st.caption("Sve auto-detektirano mjerenjem — ništa nije upisano po imenu modela.")
    if s.get("task"):
        a = st.columns(4)
        a[0].metric("task", s["task"], help=f"izvor: {s.get('task_source', '?')}")
        a[1].metric("mode", s.get("mode") or "—",
                    help="full = ima čitljive oznake; inače KD-only (gate = teacher-agreement)")
        a[2].metric("format dataseta", s.get("dataset_format") or "—")
        a[3].metric("uzoraka", f"{s.get('n_samples') or 0:,}")
        b = st.columns(4)
        b[0].metric("KD način", s.get("kd_mode") or "—")
        b[1].metric("tapova (feature-KD)", len(s.get("taps") or []))
        b[2].metric("prunable slojeva", f"{s.get('n_prunable', 0)} / {s['n_weighted']}")
        b[3].metric("enhaneri", "da" if s.get("enhancers") else "ne",
                    help="task-uvjetni decode-svjesni KD članovi (npr. detekcija)")
        if s.get("taps"):
            st.caption("Tapovi: " + ", ".join(f"`{t}`" for t in s["taps"]))
        if s.get("metrics"):
            st.caption("Metrike ovog taska: " + ", ".join(f"`{m}`" for m in s["metrics"]))
        if s.get("split_plan"):
            st.caption(f"Split plan: `{s['split_plan']}`")
    else:
        st.warning("Bez dataseta — task i metrika nisu određeni. Upiši putanju dataseta za punu sliku.")

    st.caption("Tipovi slojeva: " + ", ".join(f"{k}×{v}" for k, v in sorted(
        s["leaf_types"].items(), key=lambda kv: -kv[1])[:12]))

    col = st.columns(2)
    with col[0]:
        st.subheader("Najjeftiniji rez (spregnuta cijena/kanal)")
        st.caption("Samo trošak, bez važnosti — risk/reward se računa tijekom kompresije.")
        st.dataframe([{"sloj": r["sloj"], "širina": r["sirina"],
                       "GFLOPs/kanal": r["cijena/kanal GFLOPs"]} for r in rep["top_prune"]],
                     width="stretch", hide_index=True)
    with col[1]:
        st.subheader("Najgore poravnati (padding otpad)")
        st.caption(f"Koliko kanala fali do ×{s['align_m']}; niže poravnanje = više otpada na pločici.")
        st.dataframe([{"sloj": r["sloj"], "širina": r["sirina"],
                       "poravnanje": r["poravnanje"], f"do ×{s['align_m']}": r["do_xM"]}
                      for r in rep["worst_aligned"]], width="stretch", hide_index=True)

    with st.expander("Puna per-layer tablica"):
        st.dataframe(rep["layers"], width="stretch", hide_index=True)


# =========================== COMPRESS =========================== #
def _render_prep(prep):
    steps = prep.get("steps", {})
    for k, label in [("gpu", "GPU dostupan i slobodan"),
                     ("task", "Auto-detekcija taska i dataseta"),
                     ("batch", "Auto batch size (proba VRAM-a)"),
                     ("plan", "Plan podataka i teacher cachea"),
                     ("perf", "Baseline performanse")]:
        s = steps.get(k, {})
        st.markdown(f"{LIGHTS.get(s.get('light', 'gray'), '⚪')} **{label}**"
                    + (f" — {s['msg']}" if s.get("msg") else ""))
        d = s.get("data") or {}
        if k == "task" and d:
            cc = st.columns(4)
            cc[0].metric("task", d.get("task") or "—")
            cc[1].metric("mode", d.get("mode") or "—")
            cc[2].metric("enhaneri", "da" if d.get("enhancers") else "ne")
            cc[3].metric("metrike", len(d.get("metrics") or []))
        if k == "batch" and d:
            st.caption(f"TRAIN batch = **{d.get('train')}** (najveći koji stane uz ≤90% slobodnog VRAM-a).")
        if k == "perf" and d:
            cc = st.columns(4)
            mn = d.get("metric_name", "metrika")
            cc[0].metric(mn, f"{d['metric']:.4f}" if d.get("metric") is not None else "—",
                         help="teacher_agreement = nema čitljive metrike, gate ide na slaganje s učiteljem")
            cc[1].metric("GFLOPs", f"{d.get('gflops', 0):.4f}")
            cc[2].metric("parametri", f"{d.get('params', 0)/1e6:.3f} M")
            cc[3].metric("task", d.get("task") or "—")


@st.fragment(run_every=2)
def _prep_fragment():
    prep = _read_prep()
    _render_prep(prep)
    _render_log("prep.log", "Log pripreme", expanded=False)
    if prep.get("state") != "preparing":
        st.rerun()


def _trend(pts, key, ylabel, tol=None, color="#4c78a8"):
    """Altair linija + crvena crta na POCETNOM stanju (+ zelena na pragu/stropu ako je zadan).

    X-OS JE REDNI BROJ TOCKE, ne `step`. `step` se PONAVLJA — FT epohe unutar iste precke dijele broj
    koraka, a Faza 2 krece ponovno od 0 — pa bi se vise tocaka sudaralo na istom x. Linija bi skakala
    naprijed-nazad i izgledalo bi kao da stare tocke mijenjaju vrijednost. Pravi korak i faza idu u
    tooltip, a okomita siva crta oznacava gdje pocinje Faza 2."""
    import altair as alt
    import pandas as pd
    rows = []
    for p_ in pts:
        v = p_.get(key)
        if v is None:
            continue
        lp = int(p_.get("loop") or 1)
        stp, ep = p_.get("step"), p_.get("ft_epoch")
        lbl = "{}.{}".format(stp, ep) if (ep and ep > 1) else str(stp)
        rows.append({"i": len(rows), "value": v, "faza": "Faza {}".format(lp),
                     "korak": lbl, "mod": p_.get("phase") or ""})
    if not rows:
        return None
    vals = [r["value"] for r in rows]
    lo, hi = min(vals), max(vals)
    if tol is not None:
        lo, hi = min(lo, tol), max(hi, tol)
    pad = (hi - lo) * 0.1 if hi > lo else (abs(hi) * 0.1 or 1.0)
    df = pd.DataFrame(rows)
    ch = alt.Chart(df).mark_line(point=True, color=color).encode(
        x=alt.X("i:Q", title="točka trajektorije"),
        y=alt.Y("value:Q", title=ylabel, scale=alt.Scale(domain=[lo - pad, hi + pad])),
        tooltip=["faza:N", "korak:N", "mod:N", alt.Tooltip("value:Q", format=".4f")])
    ch = ch + alt.Chart(pd.DataFrame({"b": [vals[0]]})).mark_rule(
        color="#e45756", strokeDash=[6, 4], size=2).encode(y="b:Q")
    if tol is not None:
        ch = ch + alt.Chart(pd.DataFrame({"t": [tol]})).mark_rule(
            color="#54a24b", strokeDash=[5, 3], size=2).encode(y="t:Q")
    b2 = next((r["i"] for r in rows if r["faza"] == "Faza 2"), None)
    if b2:                                                   # granica faza
        ch = ch + alt.Chart(pd.DataFrame({"x": [b2]})).mark_rule(
            color="#999", strokeDash=[2, 3], size=1).encode(x="x:Q")
    return ch.properties(height=240, width="container")


def _render_live(pts, status, metric_name="metrika", metric_tol=None):
    if not pts:
        st.caption(f"status: **{status.get('state')}** · faza: {status.get('phase', '-')}")
        return
    base, cur = pts[0], pts[-1]

    def pct(k):
        b, c = base.get(k), cur.get(k)
        return (c - b) / b * 100 if (b and c is not None) else 0.0

    r1 = st.columns(3)
    r1[0].metric(metric_name, f"{cur.get('metric') or 0:.4f}", f"{pct('metric'):+.1f}%")
    r1[1].metric("GFLOPs", f"{cur['gflops']:.4f}", f"{pct('gflops'):+.1f}%", delta_color="inverse")
    r1[2].metric("parametri", f"{cur['params']/1e6:.3f} M", f"{pct('params'):+.1f}%", delta_color="inverse")
    r2 = st.columns(3)
    r2[0].metric("veličina (MB)", f"{cur.get('size_mb', 0):.1f}", f"{pct('size_mb'):+.1f}%", delta_color="inverse")
    r2[1].metric("GFLOPs oslobođeno", f"{cur.get('gflops_freed', 0):.4f}")
    r2[2].metric("GFLOPs reinvestirano", f"{cur.get('gflops_reinvested', 0):.4f}",
                 help="grow smije potrošiti dio oslobođenog; neto i dalje pada")
    r3 = st.columns(3)
    r3[0].metric("poravnanje", f"{(cur.get('align_score') or 0)*100:.1f}%", f"{pct('align_score'):+.1f}%",
                 help="prosječna iskoristivost ×M pločice (veće = bolje za int8/fp16)")
    r3[1].metric("rezano kanala (korak)", cur.get("removed_ch", 0))
    r3[2].metric("naraslo slojeva (korak)", len(cur.get("grown") or []))

    tol_val = (base.get("metric") or 0) * metric_tol if (metric_tol and base.get("metric")) else None
    st.caption("🔴 crvena = početno stanje  ·  🟢 zelena = prag tolerancije (metrika) / dostižni max "
               "(poravnanje)  ·  ⋮ siva okomita = početak Faze 2.  X-os je redni broj točke, ne korak "
               "(FT epohe dijele broj koraka); pravi korak je u tooltipu.")
    GRAPHS = [("metric", metric_name), ("metric_full", f"{metric_name} (puni skup)"),
              ("gflops", "GFLOPs"), ("params", "parametri"),
              ("align_score", "poravnanje"), ("size_mb", "veličina (MB)"),
              ("gflops_freed", "GFLOPs oslobođeno"), ("gflops_reinvested", "GFLOPs reinvestirano")]
    avail = [(k, l) for k, l in GRAPHS if any(p.get(k) is not None for p in pts)]
    cols = st.columns(4)
    chosen = [(k, l) for i, (k, l) in enumerate(avail)
              if cols[i % 4].checkbox(l, value=k in ("metric", "gflops", "params", "align_score"), key=f"g_{k}")]
    for i in range(0, len(chosen), 2):
        gc = st.columns(2)
        for j, (k, l) in enumerate(chosen[i:i + 2]):
            with gc[j]:
                st.markdown(f"**{l}**")
                t = tol_val if k in ("metric", "metric_full") else (
                    next((p.get("align_best") for p in reversed(pts) if p.get("align_best") is not None), None)
                    if k == "align_score" else None)
                ch = _trend(pts, k, l, tol=t)
                if ch is not None:
                    st.altair_chart(ch)

    st.caption(f"status: **{status.get('state')}** · faza: {status.get('phase', '-')}")
    if status.get("state") == "error":
        st.error(str(status.get("msg", ""))[-1800:])
    _render_log("worker.log", "Trening log", expanded=status.get("state") == "running")


def _render_log(name, label, expanded=False, tail=8000):
    """Ispis workera (isto sto bi islo u terminal) — worker ga tee-a u run-folder."""
    p = os.path.join(_run_dir(), name)
    if not os.path.exists(p):
        return
    with st.expander(label, expanded=expanded):
        txt = open(p, errors="replace").read()
        if len(txt) > tail:
            st.caption(f"(prikazano zadnjih {tail:,} znakova od {len(txt):,})")
            txt = txt[-tail:]
        st.code(txt or "(prazno)", language="text")


@st.fragment(run_every=2)
def _live_fragment(metric_name, metric_tol):
    status = _read_status()
    _render_live(_read_traj(), status, metric_name, metric_tol)
    if status.get("state") != "running":
        st.rerun()


def page_compress():
    st.title("Compress")
    os.makedirs(JOB, exist_ok=True)
    cfg_now = _read("config.json", {})
    prep = _read_prep()
    perf = (prep.get("steps", {}).get("perf", {}) or {}).get("data", {}) or {}
    metric_name = perf.get("metric_name", "metrika")

    status = _read_status()
    if status.get("state") == "running":
        st.info(f"**Praćena metrika:** {metric_name}  ·  **Tolerancija:** "
                f"{cfg_now.get('metric_tol', 0.9):.0%} početne vrijednosti.  \n"
                "Faza 1: dead/near-dead rez + KD oporavak → Faza 2: kontinuirani prune (+uvjetni grow).")
        if st.button("⏹ Zaustavi"):
            _kill(status.get("pid"))
            json.dump({"state": "stopped"}, open(os.path.join(JOB, "status.json"), "w"))
            st.rerun()
        _live_fragment(metric_name, cfg_now.get("metric_tol", 0.9))
        return

    if not st.session_state.get("model_path"):
        st.info("Upiši putanju modela i dataseta u lijevoj traci.")
        return

    # ---------- PRIPREMA ----------
    st.subheader("Priprema")
    pstate = prep.get("state", "idle")
    if pstate in ("idle", "stopped") or st.session_state.pop("reprep", False):
        if st.button("Pokreni pripremu", type="primary"):
            json.dump(_cfg_payload(), open(os.path.join(JOB, "config.json"), "w"))
            _launch("prep_worker.py", "prep_status.json", {"state": "preparing", "steps": {}})
            st.rerun()
        return
    if pstate == "preparing":
        _prep_fragment()
        return

    _render_prep(prep)
    _render_log("prep.log", "Log pripreme", expanded=pstate == "error")
    if st.button("↻ Ponovi pripremu"):
        st.session_state["reprep"] = True
        try:
            os.remove(os.path.join(JOB, "prep_status.json"))
        except OSError:
            pass
        st.rerun()
    if pstate == "error":
        st.error("Priprema nije uspjela: "
                 + str((prep.get("steps", {}).get("err", {}) or {}).get("msg", ""))[-900:])
        return
    if pstate != "ready":
        return

    # ---------- POSTAVKE ----------
    st.subheader("Postavke")
    with st.container(border=True):
        quant = st.radio("Ciljana kvantizacija (HW poravnanje kanala)",
                         ["INT8  (M=32)", "FP16  (M=8)"], horizontal=True)
        align_m = 32 if quant.startswith("INT8") else 8
        st.caption(f"Poravnava broj kanala na višekratnik **{align_m}** za brži inference. "
                   f"Min kanala po sloju = M/2 = **{align_m // 2}**.")

    with st.container(border=True):
        st.markdown(f"🟢 **Metrika kvalitete: `{metric_name}`** — auto iz taska, nije izbor.")
        _agree = metric_name == "teacher_agreement"
        _dflt = int((CFG.AGREEMENT_SUGGEST if _agree else CFG.FT_RECOVERY_FRAC) * 100)
        if _agree:
            st.warning(
                f"Nema čitljive metrike za ovaj task/dataset — prag se mjeri **slaganjem s učiteljem**.\n\n"
                f"To NIJE ista vrsta broja kao mAP ili f1: kreće od **točno 1.0** (student *je* original) "
                f"i mjeri **udio ulaza koji dobiju isti odgovor**, ne koliko je model dobar. "
                f"Tolerancija 0.90 ovdje znači „svaki deseti ulaz smije promijeniti odgovor“ — "
                f"u najgorem slučaju 10 postotnih bodova kvalitete.\n\n"
                f"Slaganje je usto **zamjena**, ne ono što te zanima. Zato je klizač postavljen na "
                f"**{CFG.AGREEMENT_SUGGEST:.2f}** umjesto uobičajenih {CFG.FT_RECOVERY_FRAC:.2f}. "
                f"Slobodno ga pomakni — prag je jedan i tvoj je.")
        base_v = perf.get("metric")
        tol = st.slider("Tolerancija (udio početne vrijednosti)", 50, 99, _dflt, step=1) / 100.0
        if base_v:
            st.caption(f"Početno `{base_v:.4f}` → prag `{base_v * tol:.4f}`. "
                       "Traži se NAJMANJI model koji drži prag.")

    with st.container(border=True):
        st.markdown("**Što pokrenuti** — može i oboje; tada Faza 2 kreće od izlaza Faze 1.")
        f1c, f2c = st.columns(2)
        phase1 = f1c.checkbox("**FAZA 1** — najmanji model iznad praga", value=True)
        phase2 = f2c.checkbox("**FAZA 2** — ljestvica do strukturnog minimuma", value=False)
        target, ft_steps, dead = 1.0, 0, False            # stara petlja vise nije u GUI-ju
        if phase1:
            st.caption(
                f"**Faza 1** · Reže dok drži prag; kad padne → do **{CFG.F1_FT_MAX_EPOCHS}** FT epoha "
                f"oporavka (patience **{CFG.F1_FT_PATIENCE}**, po epizodi). Jedan FT = PUNA epoha preko "
                f"svih batcheva. Metrika se svaku epohu mjeri na fiksnih **{CFG.METRIC_MONITOR_FRAC:.0%}** "
                "val skupa, a na punom tek kad model kandidira za isporuku. Izlaz: "
                "`best_quality_model.pt`, zajamčeno iznad praga.")
        if phase2:
            st.caption(
                f"**Faza 2** · Start: **{'izlaz Faze 1' if phase1 else 'ORIGINAL'}**"
                f"{'' if phase1 else ' (Faza 1 nije označena)'}. Prvo se IZMJERI strukturni minimum "
                f"(probni rez svih slojeva na floor), pa se raspon dijeli na **{CFG.F2_CHECKPOINTS}** "
                f"razine — zadnja JE minimum i na njoj se grow gasi. Između rezova FT: max "
                f"**{CFG.F2_FT_MAX_EPOCHS}** epoha, patience **{CFG.F2_FT_PATIENCE}**. Kvaliteta se "
                "ovdje NE koristi kao prag, samo se mjeri i zapisuje uz svaku verziju.")
        if not (phase1 or phase2):
            st.error("Označi barem jednu fazu.")
        max_steps = st.number_input("max koraka", 1, 1000, CFG.PHASE2_MAX_STEPS,
                                    help="Sigurnosna granica po fazi (u Fazi 2 po prečki). NIJE cilj.")

    plan = (prep.get("steps", {}).get("plan", {}) or {}).get("data") or {}
    if plan:
        with st.container(border=True):
            st.markdown("**Podaci** — ništa se ne bira, uzima se sve što ima")
            pc = st.columns(3)
            pc[0].metric("KD uzoraka", f"{plan.get('n_samples', 0):,}",
                         help="Cijeli train split. Teacher signali se predračunaju jednom i keširaju.")
            pc[1].metric("teacher cache", f"{plan.get('total_gb', 0):.2f} GB",
                         f"slobodno {plan.get('free_gb', 0):.0f} GB", delta_color="off")
            pc[2].metric("metrika mjeri na", f"{plan.get('n_gate', 0):,}",
                         help="Cijeli val split — jedina prava cijena po koraku.")
            if not plan.get("fits_disk", True):
                st.error("Teacher cache ne stane na disk — kompresija bi pukla usred računa.")

    if st.button("Pokreni kompresiju", type="primary", disabled=not (phase1 or phase2)):
        payload = _cfg_payload()
        payload.update({"align_m": align_m, "metric_tol": tol, "target_frac": target,
                        "max_steps": int(max_steps), "ft_steps": int(ft_steps), "dead": bool(dead),
                        # zastavica: "1", "2" ili "12" (oba -> lancano, Faza 2 od izlaza Faze 1)
                        "phase": ("1" if phase1 else "") + ("2" if phase2 else ""),
                        "batch": (prep.get("steps", {}).get("batch", {}) or {}).get("data", {})})
        json.dump(payload, open(os.path.join(JOB, "config.json"), "w"))
        _launch("worker.py", "status.json", {"state": "running", "phase": "launching"})
        st.rerun()

    pts = _read_traj()
    if pts:
        st.divider()
        st.caption(f"Zadnji run: **{status.get('state')}**")
        _render_live(pts, status, metric_name, cfg_now.get("metric_tol", 0.9))


# =========================== ABOUT =========================== #
def page_about():
    st.title("About")
    import overview as OV
    cap = OV.capabilities()
    st.caption("Sve se generira iz registara u kodu (SUPPORTED_TASKS · SUPPORTED_DATASET_FORMATS · "
               "LAYER_REGISTER · outfmt). Ne učitava nijedan model.")

    if cap.get("dev_subset"):
        st.warning(f"⚠️ DEV način: `DEV_DATA_SUBSET = {cap['dev_subset']}` — svaki split je ograničen na "
                   f"prvih {cap['dev_subset']} uzoraka (niska vjernost). Postavi `None` u `slinn/settings.py` "
                   "prije pravih runova.")

    st.subheader("Podržani taskovi")
    st.table([{"task": t["task"], "metrike": ", ".join(t["metrics"]) or "—",
               "KD jezgra": ", ".join(t["kd_core"]) or "—",
               "enhaneri": ", ".join(t["enhancers"]) or "—",
               "treba decode": "da" if t["decode"] else "ne"} for t in cap["tasks"]])
    for t in cap["tasks"]:
        if t.get("note"):
            with st.expander(f"napomena — {t['task']}"):
                st.caption(t["note"])

    st.subheader("Prepoznati formati dataseta")
    st.table([{"format": k, "prepoznaje se po": str(v.get("fingerprint", v))[:150]}
              for k, v in cap["formats"].items()])

    st.subheader("Detekcija — prepoznavanje formata IZLAZA")
    st.caption("Model koji nikad nismo vidjeli: prvo poznata obitelj, pa oblik izlaza, pa pošteno "
               "priznanje neznanja (KD-only, bez prekida).")
    st.table([{"format": k, "opis": v, "decode": cap["out_adapters"].get(k) or "—"}
              for k, v in cap["out_formats"].items()])

    st.subheader("Registar tipova slojeva")
    st.caption("Sposobnosti dobivene STVARNIM pokušajem prune/grow kroz pipeline, ne čitanjem koda.")
    st.dataframe(cap["layers"], width="stretch", hide_index=True)


# =========================== PRETHODNE KOMPRESIJE =========================== #
_ICON = {"best_quality_model.pt": "🏆", "ladder.json": "🪜", "trajectory.jsonl": "📈",
         "worker.log": "📜", "run_meta.json": "🧾"}


def _run_icon(fname):
    """Ikona po ULOZI datoteke, ne po ekstenziji — checkpointi se moraju razlikovati od zapisa."""
    if fname in _ICON:
        return _ICON[fname]
    if fname.startswith("ckpt_") and fname.endswith(".pt"):
        return "📦"
    return "📄" if not fname.endswith(".pt") else "🧠"


def _scan_runs():
    """Popis runova iz RUNS_DIR, novi prvi. Mapa `_prije_run_foldera` (arhiva iz doba prije
    per-run foldera) se preskace jer nije run i nema metapodatke."""
    root = CFG.RUNS_DIR
    if not os.path.isdir(root):
        return []
    out = []
    for d in sorted(os.listdir(root), reverse=True):
        full = os.path.join(root, d)
        if not os.path.isdir(full) or d.startswith("_"):
            continue
        # ime foldera je "<model>_<YYYYmmdd>_<HHMMSS>" -> odvoji model od datuma
        model, when = d, ""
        parts = d.rsplit("_", 2)
        if len(parts) == 3 and len(parts[1]) == 8 and len(parts[2]) == 6 and parts[1].isdigit():
            model = parts[0]
            when = "{}-{}-{} {}:{}:{}".format(parts[1][:4], parts[1][4:6], parts[1][6:],
                                              parts[2][:2], parts[2][2:4], parts[2][4:])
        meta = {}
        try:
            meta = json.load(open(os.path.join(full, "run_meta.json")))
        except BaseException:
            pass
        files = sorted(f for f in os.listdir(full) if os.path.isfile(os.path.join(full, f)))
        out.append({"dir": d, "path": full, "model": model, "when": when,
                    "meta": meta, "files": files})
    return out


def _fmt_mb(p):
    try:
        return "{:.2f} MB".format(os.path.getsize(p) / (1024 ** 2))
    except OSError:
        return "—"


def page_runs():
    st.title("Prethodne kompresije")
    runs = _scan_runs()
    if not runs:
        st.info("Još nema runova. Svaka kompresija stvara `runs/<model>_<datum>_<vrijeme>/` "
                "s logom, trajektorijom i svim checkpointima.")
        return
    st.caption("{} run(ova) u `{}` — noviji prvi.".format(len(runs), CFG.RUNS_DIR))

    for r in runs:
        m = r["meta"]
        ph = str(m.get("phase") or "?")
        faze = " + ".join(["Faza " + c for c in "12" if c in ph]) or "?"
        cks = [f for f in r["files"] if f.endswith(".pt")]
        head = "**{}**  ·  {}  ·  {}  ·  {} checkpoint(a)".format(
            r["model"], r["when"] or "bez datuma", faze, len(cks))
        with st.expander(head):
            if m:
                c = st.columns(4)
                c[0].metric("metrika", str(m.get("metric_name") or "—"))
                bl = m.get("metric_baseline")
                c[1].metric("baseline", "{:.4f}".format(bl) if isinstance(bl, (int, float)) else "—")
                tl = m.get("metric_tol")
                c[2].metric("tolerancija", "{:.0%}".format(tl) if isinstance(tl, (int, float)) else "—")
                c[3].metric("prag", "{:.4f}".format(bl * tl)
                            if isinstance(bl, (int, float)) and isinstance(tl, (int, float)) else "—")
                st.caption("model: `{}`".format(m.get("model_path") or "—"))
                st.caption("dataset: `{}`".format(m.get("dataset_path") or "—"))
            else:
                st.warning("Nema `run_meta.json` — run je vjerojatno prekinut prije kraja.")

            # --- ljestvica Faze 2, ako postoji ---
            lp = os.path.join(r["path"], "ladder.json")
            if os.path.exists(lp):
                try:
                    L = json.load(open(lp))
                    st.markdown("**Ljestvica Faze 2** — `g_start` {:.4g} → `g_min` {:.4g}".format(
                        L.get("g_start", 0), L.get("g_min", 0)))
                    st.table([{"verzija": "ckpt_{}".format(c["i"]),
                               "GFLOPs": round(c["gflops"], 6),
                               "% starta": "{:.1f}%".format(100 * c["gflops"] / L["g_start"])
                                           if L.get("g_start") else "—",
                               "params": "{:,}".format(c.get("params", 0)),
                               "metrika": ("{:.4f}".format(c["metric"])
                                           if isinstance(c.get("metric"), (int, float)) else "—"),
                               "cilj": "dosegnut" if c.get("reached") else "promašen"}
                              for c in L.get("checkpoints", [])])
                except BaseException:
                    pass

            st.markdown("**Datoteke**")
            for f in r["files"]:
                fp = os.path.join(r["path"], f)
                st.markdown("{}  `{}`  ·  {}".format(_run_icon(f), f, _fmt_mb(fp)))
            st.code(r["path"], language="text")


# =========================== ROUTER =========================== #
_startup_cleanup()
st.sidebar.title("SliNN")
st.sidebar.caption("Slimming via Imitation — lokalni pipeline za kompresiju")
PAGE = st.sidebar.radio("Stranica", ["Overview", "Compress", "Prethodne kompresije", "About"])
_paths_input()
{"Overview": page_overview, "Compress": page_compress,
 "Prethodne kompresije": page_runs, "About": page_about}[PAGE]()
