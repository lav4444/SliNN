"""data.py — NYU Depth V2 labeled validation (654) preko vikhyatk/nyu_depth_v2, SAMO val parquet (~1 GB).

Sluzbeni NYU .mat host je mrtav; load_dataset(split="validation") je bagovit (povlaci i train, 32 GB),
pa 3 validation parquet fajla skidamo direktno (hf_hub_download) i citamo pyarrow-om.
NYUVal[i] -> (PIL RGB slika, np.float32 depth [H,W] u metrima). midas_transform() = MiDaS small preset.

PODJELA (2026-09): izvor daje SAMO validation, pa je bez podjele pipeline javljao
    "split 'val' nije nadjen -> koristi se CIJELO stablo. Metrika i KD ulazi tada gledaju
     iste datoteke."
`split_nyu.py` je 654 para fizicki razdvojio u `data/train/` i `data/val/` (400 / 254,
permutacija sa seedom 42; indeksi u data/split_manifest.json). Podjela zivi u POHRANI —
ovdje nema aritmetike nad indeksima, samo izbor mape.

Raspored je namjerno `data/<split>/<split>-N.parquet`: split stoji I u imenu mape I u imenu
datoteke, jer ga SliNN cita na dva mjesta i razlicito (dataset.py:552 iz putanje,
dataset.py:581 iz imena datoteke). Tako isti zapis citaju i ovaj modul i SliNN i edge.

    NYUVal()            -> val   (254)   <- novi default
    NYUVal(split="train")            (400)
    NYUVal(split="all")              (654)  reproducira staru brojku iz eval_result.txt

PAZI: stari `eval_result.txt` javlja "NYU VAL (654 images)". Ponovno pokretanje evaluacije
sada daje 254 — druga brojka, jer je drugi skup. Za usporedbu sa starim koristi split="all".
Zato se pri prvom citanju ISPISUJE koji je split i koliko para ucitano.
"""

import io
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# Podaci su ULAZ, dijele ih sve mjerne mape -> shared/datasets/.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "shared", "datasets", "nyu_depth")   # {train,val}/<split>-N.parquet
# Izvornih 654 stoji IZVAN data/ namjerno: SliNN rekurzivno obilazi data/ i broji SVAKI
# parquet, a te se datoteke zovu `validation-*.parquet` pa bi ih pribrojio splitu 'val'
# (izmjereno: val 908 umjesto 254, ukupno 1308 umjesto 654). Smiju se i obrisati —
# `_paths()` ih vraca s HF-a kad zatreba split='all'.
NYU_DIR = os.path.join(HERE, "nyu_val_original")
REPO = "vikhyatk/nyu_depth_v2"
VAL_FILES = [
    "data/validation-00000-of-00003-0c667cfcd871aca4.parquet",
    "data/validation-00001-of-00003-5f08f98440587ee9.parquet",
    "data/validation-00002-of-00003-2d5e2542eae4a0c6.parquet",
]
HUB_DIR = os.path.join(torch.hub.get_dir(), "intel-isl_MiDaS_master")   # MiDaS klase (eager reload)
EFFNET_DIR = os.path.join(torch.hub.get_dir(), "rwightman_gen-efficientnet-pytorch_master")  # geffnet backbone
HUB_DIRS = [HUB_DIR, EFFNET_DIR]                                        # oba treba na sys.path za reload
SPLITS = ("train", "val", "all")
_TABLE = {}                                              # split -> pa.Table (kes po splitu)


def _paths():
    """Izvornih 654 — skine s HF-a ako ih nema lokalno."""
    return [hf_hub_download(REPO, f, repo_type="dataset", local_dir=NYU_DIR) for f in VAL_FILES]


def _split_paths(split):
    """Parqueti podijeljenog splita, ili [] ako podjela nije napravljena.

    Raspored je `data/<split>/<split>-N.parquet` — split stoji I u imenu mape I u imenu
    datoteke, jer ga SliNN cita na DVA mjesta i razlicito:
        dataset.py:552  _survey        -> iz PUTANJE (najblizi predak koji je split-ime)
        dataset.py:581  parquet brojac -> iz IMENA DATOTEKE (substring)
    Samo mapa ili samo ime ne bi bilo dovoljno."""
    d = os.path.join(DATA_DIR, split)
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".parquet")]


def _table(split="val"):
    if split not in SPLITS:
        raise ValueError(f"split '{split}' nije poznat; dostupno: {SPLITS}")
    if split in _TABLE:
        return _TABLE[split]

    if split == "all":
        paths, src = _paths(), "nyu_val (izvornih 654)"
    else:
        paths = _split_paths(split)
        src = f"data/{split}"
        if not paths:
            # Podjela nije napravljena -> ne pogadjaj, reci sto nedostaje i kako se dobiva.
            raise FileNotFoundError(
                f"nema parqueta u {os.path.join(DATA_DIR, split)}. Pokreni "
                f"`python split_nyu.py` (dijeli izvornih 654 na train 400 / val 254), "
                f"ili trazi split='all' za nepodijeljenih 654 iz {NYU_DIR}.")

    t = pa.concat_tables([pq.read_table(p) for p in paths])
    print(f"[nyu] split '{split}': {len(t)} para iz {src}")
    _TABLE[split] = t
    return t


def dataset(split="val", limit=None):
    """Ugovor zoo-a: dataset(split) -> Dataset."""
    return NYUVal(limit, split)


class NYUVal:
    def __init__(self, limit=None, split="val"):
        # `limit` ostaje PRVI pozicijski argument: eval_baseline.py i build.py zovu
        # D.NYUVal(limit) / D.NYUVal(1) i moraju nastaviti raditi.
        self.split = split
        self.t = _table(split)
        self.n = len(self.t) if limit is None else min(limit, len(self.t))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        im = self.t.column("image")[i].as_py()
        dm = self.t.column("depth_map")[i].as_py()
        img = Image.open(io.BytesIO(im["bytes"])).convert("RGB")
        depth = np.array(Image.open(io.BytesIO(dm["bytes"])), dtype=np.float32)      # [H,W] metri
        return img, depth


def midas_transform():
    return torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
