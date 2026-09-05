"""data.py — uniformni loader za speechcommands_m5 (12-klasni keyword spotting, sirovi val @ 8 kHz).

Podaci se skidaju UZ model u ./data/ (torchaudio SPEECHCOMMANDS v0.02) i odatle povlace.
12 klasa (klasicni Warden zadatak): 10 naredbi + 'unknown' (ostale rijeci, subsample) + 'silence'
(generiran iz _background_noise_). Split po sluzbenim listama (validation_list/testing_list.txt).

Ugovor zoo-a (za velike datasete): loader(split, batch, ...) -> DataLoader (lazy load po uzorku),
dataset(split) -> Dataset, classes() -> imena klasa. Deterministicki (seed 42).
Svaki uzorak: waveform [1, 8000] (1 s @ 8 kHz), label int 0..11.
"""

import glob
import os
import random

import torch
import torch.nn.functional as F
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
BASE = os.path.join(DATA_DIR, "SpeechCommands", "speech_commands_v0.02")

CORE = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = CORE + ["unknown", "silence"]                 # 12
UNKNOWN, SILENCE = 10, 11
SR = 8000
LEN = 8000                                             # 1 s @ 8 kHz
RAW = 16000                                            # 1 s @ 16 kHz (izvorni)
SEED = 42


def classes():
    return list(LABELS)


def _ensure_download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isdir(BASE):
        print("[data] skidam SpeechCommands v0.02 (~2.3 GB) ...")
        torchaudio.datasets.SPEECHCOMMANDS(root=DATA_DIR, download=True)
    return BASE


def _read_list(name):
    with open(os.path.join(BASE, name)) as f:
        return {ln.strip() for ln in f if ln.strip()}


def _to_8k_1s(wav, sr):
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    n = wav.shape[1]
    return F.pad(wav, (0, LEN - n)) if n < LEN else wav[:, :LEN]


def _build_index(split):
    """(kind, payload, label): kind='wav' payload=abspath; kind='sil' payload=(bg_abspath, frac)."""
    _ensure_download()
    valset, testset = _read_list("validation_list.txt"), _read_list("testing_list.txt")
    core_idx = {w: i for i, w in enumerate(CORE)}
    rng = random.Random(SEED + {"train": 0, "val": 1, "test": 2}[split])

    words = [d for d in os.listdir(BASE)
             if os.path.isdir(os.path.join(BASE, d)) and d != "_background_noise_"]
    core, unknown = [], []
    for w in words:
        for fp in glob.glob(os.path.join(BASE, w, "*.wav")):
            rel = w + "/" + os.path.basename(fp)
            sp = "test" if rel in testset else "val" if rel in valset else "train"
            if sp != split:
                continue
            (core if w in core_idx else unknown).append((fp, core_idx.get(w, UNKNOWN)))

    per_core = max(1, len(core) // len(CORE))          # cilj za unknown/silence ~ prosjek jedne klase
    rng.shuffle(unknown)
    items = [("wav", p, i) for p, i in core] + [("wav", p, UNKNOWN) for p, _ in unknown[:per_core]]

    bg = sorted(glob.glob(os.path.join(BASE, "_background_noise_", "*.wav")))
    items += [("sil", (rng.choice(bg), rng.random()), SILENCE) for _ in range(per_core)]
    rng.shuffle(items)
    return items


class SC12(torch.utils.data.Dataset):
    def __init__(self, split):
        self.items = _build_index(split)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        kind, payload, label = self.items[i]
        if kind == "wav":
            wav, sr = torchaudio.load(payload)
            return _to_8k_1s(wav, sr), label
        bf, frac = payload                             # silence: nasumican 1 s izrezak iz sum-datoteke
        wav, sr = torchaudio.load(bf)
        start = int(frac * max(1, wav.shape[1] - RAW))
        return _to_8k_1s(wav[:, start:start + RAW], sr), label


def dataset(split):
    return SC12(split)


def loader(split, batch=256, shuffle=None, workers=4):
    shuffle = (split == "train") if shuffle is None else shuffle
    return torch.utils.data.DataLoader(SC12(split), batch_size=batch, shuffle=shuffle,
                                       num_workers=workers, pin_memory=True, drop_last=False)


if __name__ == "__main__":                             # download + sanity (broj po klasi + jedan batch)
    from collections import Counter
    for split in ("train", "val", "test"):
        ds = SC12(split)
        cnt = Counter(lbl for _, _, lbl in ds.items)
        print(f"[{split:5s}] ukupno {len(ds):6d} | " + "  ".join(f"{LABELS[k]}={cnt.get(k, 0)}" for k in range(12)))
    xb, yb = next(iter(loader("val", batch=8, workers=0)))
    print("batch:", tuple(xb.shape), "labels", yb.tolist())
