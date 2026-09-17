
import torch
import torch.nn as nn


class NormalizedRegressor(nn.Module):
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
