"""model_mlp.py — cisti MLP regresor za California Housing (8 znacajki -> 1 kontinuirani izlaz).

Definiran u UVOZIVOM modulu (ne u build.__main__) da torch.save(model) zapise `model_mlp.HousingMLP`
i da se reload radi s ovim folderom na sys.path (isti obrazac kao SchoolCNN model_cnn.py).

Namjerno bogat (4 skrivena Linear-a 256/256/128/64) da compress ima sto rezati. Svaki skriveni sloj:
Linear -> BatchNorm1d -> ReLU; glava = Linear(->1). Sve rezolucije su `flat` -> stresira tap net-fallback.
"""

import torch
import torch.nn as nn


class NormalizedRegressor(nn.Module):
    """Samostalan model: prima SIROVI X, vraca SIROVI y. Standardizacija ulaza ((x-mu_x)/sd_x) i
    de-standardizacija izlaza (y*sd_y+mu_y) su BUFFERI (ne tezine) -> classify ih vidi kao passthrough,
    prune/grow dira samo `inner`. Normalizacija postaje DIO MODELA (zoo ugovor: modeli self-contained na
    sirovom ulazu/izlazu -> pipeline ne treba preprocessing-ravninu). Probe i dalje vidi vektor dim(in)."""
    def __init__(self, inner, mu_x, sd_x, mu_y, sd_y):
        super().__init__()
        self.inner = inner
        self.register_buffer("mu_x", torch.as_tensor(mu_x, dtype=torch.float32).flatten())
        self.register_buffer("sd_x", torch.as_tensor(sd_x, dtype=torch.float32).flatten())
        self.register_buffer("mu_y", torch.as_tensor(float(mu_y), dtype=torch.float32))
        self.register_buffer("sd_y", torch.as_tensor(float(sd_y), dtype=torch.float32))

    def forward(self, x):
        return self.inner((x - self.mu_x) / self.sd_x) * self.sd_y + self.mu_y


class HousingMLP(nn.Module):
    def __init__(self, in_dim=8, widths=(256, 256, 128, 64)):
        super().__init__()
        layers = []
        d = in_dim
        for w in widths:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.ReLU(inplace=False)]
            d = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.body(x))
