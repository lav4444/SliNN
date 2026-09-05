"""data.py — SST-2 (stanfordnlp/sst2) + DistilBERT tokenizer, lokalno u ./data/.

Uniformni loader: loader(split, batch) -> DataLoader koji daje (enc, labels), gdje je
enc = {"input_ids":[B,L], "attention_mask":[B,L]} (token-ID-jevi, ne float), labels = [B] (0/1).
Napomena: TEST oznake su SKRIVENE (-1) -> metrika samo na validation (i train).
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "data", "hf"))     # dataset + model cache uz model

import torch                                                            # noqa: E402
from datasets import load_dataset                                       # noqa: E402
from transformers import AutoTokenizer                                  # noqa: E402

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
DATASET_ID = "stanfordnlp/sst2"
MAX_LEN = 64
LABELS = ["negative", "positive"]
_tok = None


def classes():
    return list(LABELS)


def tokenizer():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_ID)
    return _tok


def loader(split, batch=64, shuffle=False, limit=None):
    ds = load_dataset(DATASET_ID)[split]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    tok = tokenizer()

    def collate(rows):
        enc = tok([r["sentence"] for r in rows], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        labels = torch.tensor([r["label"] for r in rows])
        return dict(enc), labels

    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, collate_fn=collate)
