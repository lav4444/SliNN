"""model_m5.py — M5 1D-CNN (Dai et al. 2017, "Very Deep CNNs for Raw Waveforms").

Radi DIREKTNO na sirovom valnom obliku (1D), ne na spektrogramu. Kanonske sirine 128/128/256/512
(~558k param) da kompresija ima sto rezati. Svaki blok: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d(4);
glava = global avg-pool + Linear. Aktivacije su nn.ReLU MODULI (ne functional) da ih census/analiza vide.

Definiran u uvozivom modulu (eager reload). Ulaz [B,1,8000] (1 s @ 8 kHz) -> logiti [B, n_output].
"""

import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout, k, s=1):
    return nn.Sequential(
        nn.Conv1d(cin, cout, kernel_size=k, stride=s),
        nn.BatchNorm1d(cout),
        nn.ReLU(),
        nn.MaxPool1d(4),
    )


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=12, widths=(128, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = widths
        self.body = nn.Sequential(
            _block(n_input, c1, k=80, s=16),
            _block(c1, c2, k=3),
            _block(c2, c3, k=3),
            _block(c3, c4, k=3),
        )
        self.fc = nn.Linear(c4, n_output)

    def forward(self, x):
        x = self.body(x)                          # [B, C, T]
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)   # global avg -> [B, C]
        return self.fc(x)                         # logiti [B, n_output]
