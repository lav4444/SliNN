
import ast
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SLINN = os.path.dirname(HERE)
MORPH = "/home/tomi/code/dipl/legacy/morphology"

if not os.path.isdir(MORPH):
    print("[_move64] `legacy/morphology` ne postoji — preseljenje je vec obavljeno (6.4).")
    print("[_move64] Ovaj alat je POVIJESNI: dokumentira KAKO je kod presao u slinn/.")
    raise SystemExit(0)
OUT = os.path.join(SLINN, "REPORTS", "move64.txt")

MODULES = ["analysis", "compress", "profiles", "kd", "config"]
PATHS = {m: os.path.join(MORPH, m + ".py") for m in MODULES}

DET_RE = re.compile(r"yolo|frcnn|rcnn|fasterr|nms|letterbox|coco|anchor|profile|decode|bbox|imgsz|_map\b|eval_map",
                    re.I)


def parse(path):
    src = open(path, errors="replace").read()
    tree = ast.parse(src)
    lines = src.splitlines()
    syms, aliases = {}, {}

    def add(name, node):
        end = getattr(node, "end_lineno", node.lineno)
        syms[name] = {"lineno": node.lineno, "end": end, "node": node,
                      "kind": "const" if isinstance(node, (ast.Assign, ast.AnnAssign)) else "func",
                      "src": "\n".join(lines[node.lineno - 1:end])}

    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            aliases.update(imports_of(n))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(n.name, n)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    add(t.id, n)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            add(n.target.id, n)
    return syms, aliases


def imports_of(n):
    out = {}
    if isinstance(n, ast.Import):
        for a in n.names:
            base = a.name.split(".")[0]
            if base in MODULES:
                out[a.asname or base] = base
    elif isinstance(n, ast.ImportFrom) and n.module in MODULES:
        for a in n.names:
            out["::" + (a.asname or a.name)] = n.module + "." + a.name
    return out


ALL, ALIAS = {}, {}
for m in MODULES:
    if os.path.exists(PATHS[m]):
        ALL[m], ALIAS[m] = parse(PATHS[m])


def edges(mod, node):
    al = dict(ALIAS[mod])
    for x in ast.walk(node):
        if isinstance(x, (ast.Import, ast.ImportFrom)):
            al.update(imports_of(x))
    own = set(ALL[mod])
    out = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name):
            tgt = al.get(x.value.id)
            if tgt and x.attr in ALL.get(tgt, {}):
                out.add(tgt + "." + x.attr)
        elif isinstance(x, ast.Name):
            q = al.get("::" + x.id)
            if q:
                out.add(q)
            elif x.id in own:
                out.add(mod + "." + x.id)
    return out


GRAPH = {}
for m in ALL:
    for name, d in ALL[m].items():
        GRAPH[m + "." + name] = edges(m, d["node"])

ALIAS2MOD = {"A": "analysis", "C": "compress", "P": "profiles", "K": "kd", "CFG": "config"}
entries, callers = set(), {}
for d in (SLINN, os.path.join(SLINN, "gui")):
    for f in sorted(os.listdir(d)):
        if not f.endswith(".py"):
            continue
        txt = open(os.path.join(d, f), errors="replace").read()
        for al, sym in re.findall(r"\b(A|C|P|K|CFG)\.([A-Za-z_][A-Za-z0-9_]*)", txt):
            mod = ALIAS2MOD[al]
            if sym in ALL.get(mod, {}):
                q = mod + "." + sym
                entries.add(q)
                callers.setdefault(q, set()).add(f)

def closure(roots):
    seen, stack = set(), list(roots)
    while stack:
        q = stack.pop()
        if q in seen:
            continue
        seen.add(q)
        stack += [e for e in GRAPH.get(q, ()) if e not in seen]
    return seen


DET_ENTRIES = {"analysis.pick_adapter", "analysis.eval_map", "analysis.make_gt_loader"}
det_roots = DET_ENTRIES & entries
core_roots = entries - DET_ENTRIES
DET_CL, CORE_CL = closure(det_roots), closure(core_roots)
seen = DET_CL | CORE_CL

slinn_syms = {}
for f in sorted(os.listdir(SLINN)):
    if f.endswith(".py"):
        try:
            s, _ = parse(os.path.join(SLINN, f))
        except SyntaxError:
            continue
        for k in s:
            slinn_syms.setdefault(k, []).append(f)

L = ["===== 6.4 MOVE-LIST: morphology -> slinn =====",
     "Nacelo: slinn jezgra pobjeduje; iz morphology seli samo ono cega u slinn nema.",
     "Prati def/class + modul-konstante + lokalne uvoze.", "",
     "ULAZNE TOCKE (sto slinn zove): {}".format(len(entries)),
     "TRANZITIVNO ZATVORENJE:        {} simbola".format(len(seen)), ""]

tot_move = tot_all = 0
for m in MODULES:
    if m not in ALL:
        continue
    all_ln = sum(d["end"] - d["lineno"] + 1 for d in ALL[m].values())
    mv = [n for n in ALL[m] if m + "." + n in seen]
    mv_ln = sum(ALL[m][n]["end"] - ALL[m][n]["lineno"] + 1 for n in mv)
    tot_move += mv_ln
    tot_all += all_ln
    L.append("[{:<9}] seli {:>2}/{:<2} simbola  ·  {:>4}/{:<4} redaka  ({:.0f}%)".format(
        m, len(mv), len(ALL[m]), mv_ln, all_ln, 100.0 * mv_ln / all_ln if all_ln else 0))
L.append("")
L.append("UKUPNO SELI: {} redaka (od {} u tijelima simbola)".format(tot_move, tot_all))
L.append("")

core, det, shared, coll = [], [], [], []
for q in sorted(seen):
    m, n = q.split(".", 1)
    d = ALL[m][n]
    row = (q, d["end"] - d["lineno"] + 1, "ENTRY" if q in entries else "dep", d["kind"])
    if n in slinn_syms:
        coll.append(row + (",".join(slinn_syms[n]),))
    elif q in DET_CL and q in CORE_CL:
        shared.append(row)
    elif q in DET_CL:
        det.append(row)
    else:
        core.append(row)


def dump(title, rows, extra=False):
    L.append("--- " + title + " ---")
    if not rows:
        L.append("  (nema)")
    for r in sorted(rows, key=lambda r: -r[1]):
        line = "  {:<32} {:>4} r  [{}{}]".format(r[0], r[1], r[2], "/const" if r[3] == "const" else "")
        if extra:
            line += "  slinn: " + r[4]
        L.append(line)
    L.append("  = {} simbola, {} redaka".format(len(rows), sum(r[1] for r in rows)))
    L.append("")


dump("KOLIZIJE (slinn VEC ima -> koristi slinn, morphology odbaci)", coll, extra=True)
dump("SAMO DETEKCIJA -> slinn/plugins/detection/", det)
dump("DIJELJENO (doseze i jezgra i detekcija -> jezgra, plug ga uvozi)", shared)
dump("SAMO JEZGRA -> slinn/ (agnosticno)", core)

L.append("--- ULAZNE TOCKE po pozivatelju ---")
for q in sorted(entries):
    L.append("  {:<32} <- {}".format(q, ", ".join(sorted(callers[q]))))

txt = "\n".join(L)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(txt + "\n")
print(txt)
print("\n-> " + OUT)
