"""_extract64.py -- 6.4 mehanicka ekstrakcija jezgrenih simbola iz morphology u slinn.

NE prepisuje kod rukom: cita AST, izvlaci IZVORNE raspone trazenih simbola u
izvornom redoslijedu i sklapa novi modul. Preseljenje mora biti SEMANTICKI NEUTRALNO
(nula promjena ponasanja) -- inace `_parity60.py` gubi smisao kao provjera.

DRY_RUN = True samo ispise sto bi napravio.
OUT: slinn/morph.py (compress mehanike), slinn/introspect.py (analysis generika)
"""

import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SLINN = os.path.dirname(HERE)
MORPH = "/home/tomi/code/dipl/legacy/morphology"

if not os.path.isdir(MORPH):                                  # arhiva obrisana -> alat je odradio svoje
    print("[_extract64] `legacy/morphology` ne postoji — preseljenje je vec obavljeno (6.4).")
    print("[_extract64] Ovaj alat je POVIJESNI: dokumentira KAKO je kod presao u slinn/.")
    raise SystemExit(0)
DRY_RUN = False

# Sto seli. Popis iz REPORTS/move64.txt, kategorija "SAMO JEZGRA".
JOBS = [
    {"src": os.path.join(MORPH, "compress.py"), "out": os.path.join(SLINN, "morph.py"),
     "doc": '"""slinn/morph.py — dokazane prune/grow/dead mehanike (preseljeno iz morphology/compress.py).\n\n'
            'TASK-AGNOSTICNO: operira nad (model, adapter), ne zna nista o tasku ni obitelji modela.\n'
            'Preseljeno BEZ IZMJENA (semanticki neutralno) -- svaka izmjena ide u zaseban korak nakon pariteta.\n'
            'Sadrzi: coupled tp-group cost, GradMax grow, dead/near-dead rez, align faktore,\n'
            'i depthwise guard u _widen_* (trial + forward-validate + commit/rollback).\n"""',
     "syms": ["GB", "gpu_status", "_is_depthwise", "_forward_ok", "_max_abs_diff",
              "align_factors", "_align_prune_score", "coupled_unit_cost", "prune_costs",
              "_select_prune_plan", "_apply_prune_plan", "remove_dead_neardead",
              "grow_potential", "_insert_in_zeros", "_widen_out", "_widen_bn",
              "_widen_frozen_bn", "_widen_depthwise", "_try_grow_layer",
              "_select_grow_plan", "_grow_decide"]},
    {"src": os.path.join(MORPH, "analysis.py"), "out": os.path.join(SLINN, "introspect.py"),
     "doc": '"""slinn/introspect.py — genericka introspekcija modela (preseljeno iz morphology/analysis.py).\n\n'
            'TASK-AGNOSTICNO: layer-tablica, census aktivnosti, brojanje parametara, GFLOPs, eager load.\n'
            'Detekcijski adapteri/decode/mAP NISU ovdje -- oni su plug (slinn/plugins/detection/).\n'
            'Preseljeno BEZ IZMJENA osim sto `load_any` gubi "fasterrcnn" string-precac (zoo pogodnost).\n"""',
     "syms": ["ACT_TYPES", "weighted_leaves", "count_params", "gflops_total",
              "unfreeze_bn", "load_eager", "load_any", "activation_stats", "layer_table"]},
]


def extract(path, syms):
    """-> (import_block, [(name, src)]) u IZVORNOM redoslijedu."""
    src = open(path, errors="replace").read()
    lines = src.splitlines()
    tree = ast.parse(src)
    imports, found = [], []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            end = getattr(n, "end_lineno", n.lineno)
            imports.append("\n".join(lines[n.lineno - 1:end]))
        name = None
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = n.name
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
        if name in syms:
            end = getattr(n, "end_lineno", n.lineno)
            start = n.lineno - 1
            while start > 0 and lines[start - 1].lstrip().startswith("#"):    # ponesi komentar iznad
                start -= 1
            found.append((name, "\n".join(lines[start:end])))
    return imports, found


for job in JOBS:
    imports, found = extract(job["src"], set(job["syms"]))
    got = [n for n, _ in found]
    missing = [s for s in job["syms"] if s not in got]
    print("[{}] nasao {}/{} simbola".format(os.path.basename(job["out"]), len(got), len(job["syms"])))
    if missing:
        print("  !! NEDOSTAJE: " + ", ".join(missing))
    body = "\n\n\n".join(s for _, s in found)
    text = job["doc"] + "\n\n" + "\n".join(imports) + "\n\n\n" + body + "\n"
    print("  {} redaka".format(text.count("\n")))
    if DRY_RUN:
        continue
    open(job["out"], "w").write(text)
    print("  -> " + job["out"])
