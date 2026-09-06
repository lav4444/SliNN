
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analyze as A
import deep_analyze as D

DEEP_ANALYSIS   = True
GRAD_MAX_IMAGES = 512


def build_insights(data):
    rec = data["rec"]; s = data["summary"]
    ins = []
    if not rec:
        return ["Nema tezinskih slojeva za analizu."]

    by_flops = sorted(rec, key=lambda r: r["gflops"], reverse=True)
    tot_f = s["total_flops"] or 1e-9
    top1 = by_flops[0]
    ins.append(f"Najteži sloj po računu: <b>{top1['name']}</b> ({top1['type']}) — "
               f"{top1['gflops']:.3f} GFLOPs ({top1['gflops']/tot_f*100:.0f}% ukupnog).")
    top5 = sum(r["gflops"] for r in by_flops[:5])
    ins.append(f"Top 5 slojeva nosi <b>{top5/tot_f*100:.0f}%</b> ukupnog računa "
               f"(od {s['n_layers']} slojeva) — tu se isplati optimizirati/rezati.")

    by_w = sorted(rec, key=lambda r: r["weights"], reverse=True)
    topw = by_w[0]
    ins.append(f"Najviše parametara u: <b>{topw['name']}</b> — {topw['weights']:,} "
               f"({topw['weights']/max(s['total_params'],1)*100:.0f}% ukupnih).")

    if s["n_conv"] and not s["n_lin"]:
        ins.append("Model je <b>isključivo konvolucijski</b> (nema dense slojeva) — "
                   "tipično za moderne detektore / backbone-ove.")
    elif s["n_lin"] and not s["n_conv"]:
        ins.append("Model je <b>isključivo dense</b> (linearni slojevi).")
    elif s["n_conv"] and s["n_lin"]:
        ins.append(f"Hibrid: <b>{s['n_conv']} conv</b> + <b>{s['n_lin']} linear</b> slojeva "
                   "(conv značajke → dense glava).")

    cpu, gpu = s["cpu_ms"], s["gpu_ms"]
    if cpu == cpu and gpu == gpu and gpu > 0:
        ins.append(f"GPU je <b>{cpu/gpu:.1f}×</b> brži od CPU-a "
                   f"({gpu:.1f} vs {cpu:.1f} ms/sliku).")
    dens = s["total_flops"] / max(s["total_params"] / 1e6, 1e-9)
    ins.append(f"Gustoća računa: <b>{dens:.2f} GFLOPs / milijun param.</b> "
               "(visoko = računski intenzivno na malo težina, npr. velika rezolucija).")
    return ins


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="hr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --bg:#0f1117; --panel:#181b24; --panel2:#1f2330; --line:#2a2f3d;
    --txt:#e7e9ee; --mut:#9aa3b2; --accent:#5b8cff; --accent2:#3ad6a0;
    --warn:#f0a64a; --det:#5b8cff; --cls:#3ad6a0; --oth:#b07bff;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    line-height:1.5;padding:0 0 60px}
  .wrap{max-width:1100px;margin:0 auto;padding:0 20px}
  header{background:linear-gradient(135deg,#1a1f2e,#12151d);border-bottom:1px solid var(--line);
    padding:26px 0 22px;margin-bottom:24px}
  h1{margin:0 0 6px;font-size:24px;font-weight:650;letter-spacing:.2px}
  h1 .mono{color:var(--accent);font-family:ui-monospace,Menlo,Consolas,monospace}
  .badges{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
  .badge{background:var(--panel2);border:1px solid var(--line);border-radius:999px;
    padding:3px 12px;font-size:12.5px;color:var(--mut)}
  .badge b{color:var(--txt);font-weight:600}
  h2{font-size:15px;text-transform:uppercase;letter-spacing:1.2px;color:var(--mut);
    margin:30px 0 12px;font-weight:600}
  .taskbar{border-radius:14px;padding:16px 18px;font-size:16px;font-weight:550;
    border:1px solid var(--line);display:flex;align-items:center;gap:12px}
  .taskbar .dot{width:12px;height:12px;border-radius:50%}
  .taskbar .sub{font-size:13px;color:var(--mut);font-weight:400;margin-top:3px}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:15px 16px}
  .card .k{font-size:12.5px;color:var(--mut);margin-bottom:6px}
  .card .v{font-size:23px;font-weight:660;letter-spacing:.3px}
  .card .v small{font-size:13px;color:var(--mut);font-weight:400}
  .insights{list-style:none;padding:0;margin:0;display:grid;gap:9px}
  .insights li{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);
    border-radius:10px;padding:11px 14px;font-size:14px;color:var(--txt)}
  .insights li b{color:#fff}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
  .toggle{display:inline-flex;background:var(--panel2);border:1px solid var(--line);
    border-radius:10px;overflow:hidden;margin-bottom:14px}
  .toggle button{background:none;border:none;color:var(--mut);padding:7px 14px;cursor:pointer;
    font-size:13px;font-weight:550}
  .toggle button.on{background:var(--accent);color:#fff}
  .bar-row{display:flex;align-items:center;gap:10px;margin:3px 0;cursor:default}
  .bar-lab{width:230px;flex:0 0 230px;font-size:11.5px;color:var(--mut);white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis;font-family:ui-monospace,Menlo,Consolas,monospace}
  .bar-track{flex:1;background:var(--panel2);border-radius:5px;height:15px;position:relative;overflow:hidden}
  .bar-fill{height:100%;border-radius:5px;min-width:2px;transition:width .25s}
  .bar-val{flex:0 0 84px;text-align:right;font-size:11.5px;color:var(--txt);font-variant-numeric:tabular-nums}
  .bar-row:hover .bar-lab{color:var(--txt)}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
  th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left}
  th{color:var(--mut);font-weight:600;cursor:pointer;user-select:none;position:sticky;top:0;background:var(--panel)}
  th:hover{color:var(--txt)}
  td:nth-child(2){font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
  tr:hover td{background:var(--panel2)}
  .typetag{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:middle}
  .search{background:var(--panel2);border:1px solid var(--line);border-radius:9px;color:var(--txt);
    padding:8px 12px;font-size:13px;width:240px;margin-bottom:12px}
  .tablewrap{max-height:480px;overflow:auto;border:1px solid var(--line);border-radius:12px}
  details{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:4px 16px;margin-top:12px}
  summary{cursor:pointer;padding:11px 0;font-weight:600;color:var(--txt)}
  details img{max-width:100%;border-radius:8px;background:#fff;margin:10px 0}
  pre{background:#0b0d13;border:1px solid var(--line);border-radius:8px;padding:14px;overflow:auto;
    font-size:12px;color:#cdd3df;max-height:460px}
  .legend{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0 0;font-size:12px;color:var(--mut)}
  .legend span{display:inline-flex;align-items:center;gap:5px}
  #tip{position:fixed;pointer-events:none;background:#000;border:1px solid var(--line);
    border-radius:8px;padding:8px 10px;font-size:12px;color:var(--txt);opacity:0;transition:opacity .1s;
    z-index:99;max-width:300px;box-shadow:0 6px 24px rgba(0,0,0,.5)}
  #tip b{color:var(--accent)}
  .foot{color:var(--mut);font-size:12px;margin-top:30px;text-align:center}
  svg{width:100%;height:160px;display:block}
</style>
</head>
<body>
<header><div class="wrap">
  <h1>Analiza modela &mdash; <span class="mono" id="mname"></span></h1>
  <div class="badges" id="badges"></div>
</div></header>

<div class="wrap">
  <div class="taskbar" id="taskbar"><div class="dot"></div><div>
    <div id="task-main"></div><div class="sub" id="task-sub"></div></div></div>

  <h2>Sažetak</h2>
  <div class="cards" id="cards"></div>

  <h2>Ključni uvidi</h2>
  <ul class="insights" id="insights"></ul>

  <h2>Po slojevima</h2>
  <div class="panel">
    <div class="toggle" id="toggle">
      <button data-m="gflops" class="on">GFLOPs</button>
      <button data-m="weights">Parametri</button>
      <button data-m="units">Filteri/neuroni</button>
    </div>
    <div id="bars"></div>
    <div class="legend" id="legend"></div>
  </div>

  <h2>Koncentracija računa (kumulativno, redoslijed sloja)</h2>
  <div class="panel"><svg id="cum" viewBox="0 0 1000 160" preserveAspectRatio="none"></svg></div>

  <h2>Tablica slojeva</h2>
  <input class="search" id="search" placeholder="filtriraj po imenu/tipu...">
  <div class="tablewrap"><table id="tbl"><thead><tr>
    <th data-k="idx">#</th><th data-k="name">sloj</th><th data-k="type">tip</th>
    <th data-k="weights">težine</th><th data-k="filters">filt</th>
    <th data-k="neurons">neur</th><th data-k="gflops">GFLOPs</th>
  </tr></thead><tbody id="tbody"></tbody></table></div>

  <div id="deepwrap" style="display:none">
    <h2>🔬 Mrtve / slabo iskorištene jedinice <span id="dead-sub" style="text-transform:none;color:var(--mut);font-weight:400"></span></h2>
    <div class="cards" id="dead-cards"></div>
    <div class="panel" style="margin-top:12px"><div id="dead-bars"></div></div>

    <h2>✂️ Pruning potencijal <span style="text-transform:none;color:var(--mut);font-weight:400">(kriterij: gradient)</span></h2>
    <ul class="insights" id="prune-ins"></ul>
    <div class="panel"><div id="prune-bars"></div></div>

    <h2>🌱 Growing potencijal <span style="text-transform:none;color:var(--mut);font-weight:400">(GradMax)</span></h2>
    <ul class="insights" id="grow-ins"></ul>
    <div class="panel"><div id="grow-bars"></div></div>
  </div>

  <details><summary>📊 layers.png (matplotlib graf)</summary>
    <img id="layimg" alt="layers.png"></details>
  <details><summary>📄 Sirovi report (analysis_report.txt)</summary>
    <pre id="rawreport"></pre></details>

  <div class="foot">Generirano s analyze_gui.py · self-contained · podaci ugrađeni u datoteku</div>
</div>

<div id="tip"></div>

<script>
const DATA = __DATA__;
const PALETTE = ["#5b8cff","#3ad6a0","#f0a64a","#e0607a","#b07bff","#46c4d6","#c2a24a","#7c8aa0"];
const typeColors = {};
function colorFor(t){ if(!(t in typeColors)) typeColors[t]=PALETTE[Object.keys(typeColors).length%PALETTE.length]; return typeColors[t]; }
const fmt = n => n.toLocaleString("en-US");
const esc = s => String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

// header
document.getElementById("mname").textContent = DATA.model_name;
document.getElementById("badges").innerHTML =
  `<span class="badge">ulaz <b>${DATA.input_size}×${DATA.input_size}</b></span>`+
  `<span class="badge">uređaj <b>${DATA.device}</b></span>`+
  `<span class="badge">format <b>eager full module</b></span>`+
  `<span class="badge">slojeva <b>${DATA.summary.n_layers}</b></span>`;
document.getElementById("layimg").src = "data:image/png;base64,"+DATA.layers_png_b64;
document.getElementById("rawreport").textContent = DATA.report_text;

// task banner
(function(){
  const t = DATA.task.toLowerCase(); let c=getCSS("--oth"), kind="Ostalo / složeno";
  if(t.includes("detekcij")){c=getCSS("--det");kind="Detekcija — lokalizacija + klasifikacija";}
  else if(t.includes("klasifikacij")){c=getCSS("--cls");kind="Klasifikacija";}
  const tb=document.getElementById("taskbar");
  tb.querySelector(".dot").style.background=c; tb.style.borderColor=c+"55";
  document.getElementById("task-main").textContent=kind;
  document.getElementById("task-sub").textContent=DATA.task;
})();
function getCSS(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}

// cards
const s=DATA.summary;
const cards=[
  ["Parametri", fmt(s.total_params), `~${s.params_mb.toFixed(2)} MB fp32`],
  ["GFLOPs", s.total_flops.toFixed(3), `@${DATA.input_size}px · 2·MAC`],
  ["Slojevi", s.n_layers, `${s.n_conv} conv · ${s.n_lin} linear`],
  ["Filteri / neuroni", fmt(s.total_filters)+" / "+fmt(s.total_neurons), "conv out / linear out"],
  ["CPU", isFinite(s.cpu_ms)?s.cpu_ms.toFixed(1)+" ms":"n/a", isFinite(s.cpu_ms)?(1000/s.cpu_ms).toFixed(0)+" FPS":""],
  ["GPU", isFinite(s.gpu_ms)?s.gpu_ms.toFixed(1)+" ms":"n/a", isFinite(s.gpu_ms)?(1000/s.gpu_ms).toFixed(0)+" FPS":""],
];
document.getElementById("cards").innerHTML = cards.map(c=>
  `<div class="card"><div class="k">${c[0]}</div><div class="v">${c[1]} <small>${c[2]}</small></div></div>`).join("");

// insights
document.getElementById("insights").innerHTML = DATA.insights.map(i=>`<li>${i}</li>`).join("");

// bars
const tip=document.getElementById("tip");
function showTip(html,e){tip.innerHTML=html;tip.style.opacity=1;moveTip(e);}
function moveTip(e){let x=e.clientX+14,y=e.clientY+14;
  if(x+310>innerWidth)x=e.clientX-310;tip.style.left=x+"px";tip.style.top=y+"px";}
function hideTip(){tip.style.opacity=0;}
function metricVal(r,m){return m==="gflops"?r.gflops:m==="weights"?r.weights:(r.filters||r.neurons);}
function renderBars(m){
  const rows=DATA.rec, max=Math.max(...rows.map(r=>metricVal(r,m)),1e-9);
  const unit=m==="gflops"?"":"";
  document.getElementById("bars").innerHTML = rows.map(r=>{
    const v=metricVal(r,m), w=Math.max(v/max*100,0.5), col=colorFor(r.type);
    const vs = m==="gflops"?v.toFixed(3):fmt(v);
    return `<div class="bar-row" data-n="${esc(r.name)}" data-t="${esc(r.type)}" data-w="${r.weights}" data-f="${r.filters}" data-u="${r.neurons}" data-g="${r.gflops}">`+
      `<div class="bar-lab">${esc(r.name)}</div>`+
      `<div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${col}"></div></div>`+
      `<div class="bar-val">${vs}</div></div>`;
  }).join("");
  // legend
  document.getElementById("legend").innerHTML = Object.entries(typeColors).map(([t,c])=>
    `<span><span class="typetag" style="background:${c}"></span>${esc(t)}</span>`).join("");
  document.querySelectorAll(".bar-row").forEach(el=>{
    el.onmousemove=e=>showTip(
      `<b>${el.dataset.n}</b> · ${el.dataset.t}<br>težine: ${fmt(+el.dataset.w)}<br>`+
      (el.dataset.f>0?`filteri: ${el.dataset.f}<br>`:"")+(el.dataset.u>0?`neuroni: ${el.dataset.u}<br>`:"")+
      `GFLOPs: ${(+el.dataset.g).toFixed(3)}`,e);
    el.onmouseleave=hideTip;
  });
}
document.querySelectorAll("#toggle button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("#toggle button").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); renderBars(b.dataset.m);
});
renderBars("gflops");

// cumulative compute curve
(function(){
  const rows=DATA.rec, tot=rows.reduce((a,r)=>a+r.gflops,0)||1e-9;
  let acc=0; const pts=rows.map((r,i)=>{acc+=r.gflops;return [i/(rows.length-1||1)*1000, 160-acc/tot*150-5];});
  const svg=document.getElementById("cum");
  let d="M0,155 "+pts.map(p=>`L${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ")+" L1000,155 Z";
  let line="M"+pts.map(p=>`${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" L");
  svg.innerHTML=`<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">`+
    `<stop offset="0" stop-color="#5b8cff" stop-opacity=".45"/><stop offset="1" stop-color="#5b8cff" stop-opacity="0"/></linearGradient></defs>`+
    `<path d="${d}" fill="url(#g)"/><path d="${line}" fill="none" stroke="#5b8cff" stroke-width="2"/>`+
    `<line x1="0" y1="80" x2="1000" y2="80" stroke="#2a2f3d" stroke-dasharray="4"/>`+
    `<text x="6" y="16" fill="#9aa3b2" font-size="11">100% računa</text>`+
    `<text x="6" y="150" fill="#9aa3b2" font-size="11">0% (prvi sloj → zadnji)</text>`;
})();

// table
let sortK="idx", sortDir=1;
const rowsT = DATA.rec.map((r,i)=>({idx:i,...r}));
function renderTable(){
  const q=document.getElementById("search").value.toLowerCase();
  let rs=rowsT.filter(r=>!q||r.name.toLowerCase().includes(q)||r.type.toLowerCase().includes(q));
  rs.sort((a,b)=>{let x=a[sortK],y=b[sortK];
    if(typeof x==="string")return sortDir*x.localeCompare(y);return sortDir*(x-y);});
  document.getElementById("tbody").innerHTML=rs.map(r=>
    `<tr><td>${r.idx}</td><td><span class="typetag" style="background:${colorFor(r.type)}"></span>${esc(r.name)}</td>`+
    `<td>${esc(r.type)}</td><td>${fmt(r.weights)}</td><td>${r.filters||""}</td>`+
    `<td>${r.neurons||""}</td><td>${r.gflops.toFixed(3)}</td></tr>`).join("");
}
document.querySelectorAll("#tbl th").forEach(th=>th.onclick=()=>{
  const k=th.dataset.k; if(k===sortK)sortDir*=-1;else{sortK=k;sortDir=(k==="name"||k==="type")?1:-1;}
  renderTable();
});
document.getElementById("search").oninput=renderTable;
renderTable();

// ---- dubinska analiza (mrtve jedinice + pruning/growing potencijal) ----
function simpleBars(id, items, labelf, pctf, valf, color, titlef){
  document.getElementById(id).innerHTML = items.map(it=>{
    const w=Math.min(Math.max(pctf(it),0.5),100);
    return `<div class="bar-row" title="${esc(titlef?titlef(it):'')}">`+
      `<div class="bar-lab">${esc(labelf(it))}</div>`+
      `<div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${color}"></div></div>`+
      `<div class="bar-val">${valf(it)}</div></div>`;
  }).join("");
}
if(DATA.deep){
  const DD=DATA.deep, at=DD.act_totals, pg=DD.potential;
  document.getElementById("deepwrap").style.display="block";
  document.getElementById("dead-sub").textContent=`— preko ${fmt(DD.n_images_act)} slika`;
  document.getElementById("dead-cards").innerHTML=[
    ["Mrtve jedinice", at.dead+" / "+at.channels, at.dead_pct.toFixed(1)+"% nikad ne opali"],
    ["Slabe (<1% aktivne)", at.weak, at.weak_pct.toFixed(1)+"% rijetko opali"],
    ["Analizirano", fmt(DD.n_images_act)+" / "+fmt(DD.n_images_grad), "aktivacije / gradijent"],
  ].map(x=>`<div class="card"><div class="k">${x[0]}</div><div class="v">${x[1]} <small>${x[2]}</small></div></div>`).join("");
  const al=[...DD.act_layers].sort((a,b)=>(b.dead+b.weak)-(a.dead+a.weak)).slice(0,40);
  simpleBars("dead-bars", al, x=>x.name+" ["+x.type+"]", x=>x.dead_pct,
    x=>x.dead+"d "+x.weak+"w", "#e0607a",
    x=>`${x.name}: ${x.dead} mrtvih + ${x.weak} slabih / ${x.channels} | mean act ${x.mean_act.toFixed(3)} | aktivan ${(x.active_frac*100).toFixed(0)}%`);

  if(pg){
    const c=pg.concentration;
    document.getElementById("prune-ins").innerHTML=[
      `Bottom ${(pg.low_pct_threshold*100).toFixed(0)}% po gradijentnoj važnosti = <b>${pg.global_low_frac.toFixed(0)}%</b> svih filtera/neurona — kandidati za rez.`,
      `Koncentracija važnosti: top 25% filtera nosi <b>${c.top25.toFixed(0)}%</b> (top 10%: ${c.top10.toFixed(0)}%) — ostatak doprinosi malo.`,
      `Mrtvih <b>${at.dead}</b> jedinica je uvijek sigurno za rez. Kriterij = <b>gradient</b> (najbolji u eksperimentima).`,
    ].map(s=>`<li>${s}</li>`).join("");
    simpleBars("prune-bars", pg.prune_layers.slice(0,40), x=>x.name+" ["+x.type+"]", x=>x.low_pct,
      x=>x.low_pct.toFixed(0)+"%", "#f0a64a",
      x=>`${x.name}: ${x.low_imp}/${x.channels} slabih (${x.low_pct.toFixed(0)}%) | mean imp ${x.mean_imp.toExponential(2)}`);

    document.getElementById("grow-ins").innerHTML=[
      `Najisplativiji slojevi za rast (GradMax benefit = sr. gradijentna važnost/jedinici): <b>${pg.grow_layers.slice(0,3).map(g=>g.name).join(", ")}</b>.`,
      `Slojevi s puno mrtvih/slabih jedinica NISU za rast — tamo radije reži.`,
    ].map(s=>`<li>${s}</li>`).join("");
    simpleBars("grow-bars", pg.grow_layers.slice(0,40), x=>x.name+" ["+x.type+"]", x=>x.benefit_norm*100,
      x=>(x.benefit_norm*100).toFixed(0)+"%", "#3ad6a0",
      x=>`${x.name}: benefit ${x.benefit.toExponential(2)} (${(x.benefit_norm*100).toFixed(0)}% od maks)`);
  } else {
    const note=`<li>Gradijentni signal <b>nedostupan</b> za ovaj model — izlaz mu je odvojen od računskog grafa (npr. ultralytics u inference-mode), pa se gradijentna važnost ne može izračunati. Analiza mrtvih jedinica gore i dalje vrijedi. Za pun pruning/growing potencijal koristi modele koji vraćaju raw izlaz spojen na graf (npr. naši StudentYOLO / SchoolCNN).</li>`;
    document.getElementById("prune-ins").innerHTML=note;
    document.getElementById("grow-ins").innerHTML=note;
  }
}
</script>
</body></html>
"""


def render_html(data):
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (HTML_TEMPLATE
            .replace("__TITLE__", f"Analiza — {data['model_name']}")
            .replace("__DATA__", blob))


def generate(model_path, data_dir, input_size, code_dirs, n_images, device, outdir):
    data = A.run(model_path, data_dir, input_size, code_dirs, n_images, device, outdir)
    outdir = Path(outdir)
    png_path = Path(data["png_path"])
    rpt_path = Path(data["report_path"])
    data["layers_png_b64"] = base64.b64encode(png_path.read_bytes()).decode() if png_path.exists() else ""
    data["report_text"] = rpt_path.read_text() if rpt_path.exists() else ""
    data["insights"] = build_insights(data)
    if DEEP_ANALYSIS and data_dir:
        try:
            data["deep"] = D.run_deep(model_path, data_dir, input_size, code_dirs, device,
                                      grad_max_images=GRAD_MAX_IMAGES)
        except Exception as e:
            print(f"[deep] preskoceno (analiza se nastavlja bez dubinskog dijela): {e}")
    out_html = outdir / "report.html"
    out_html.write_text(render_html(data), encoding="utf-8")
    print(f"Spremljeno: {out_html}")
    return out_html


if __name__ == "__main__":
    if not A.MODEL_PATH:
        sys.exit("Postavi MODEL_PATH u analyze.py prije pokretanja analyze_gui.py.")
    generate(A.MODEL_PATH, A.DATA_DIR or None, A.INPUT_SIZE, A.CODE_DIRS or None,
             A.N_IMAGES, A.DEVICE, A.OUT_DIR)
