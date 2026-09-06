
import os
import torch
import numpy as np

from skip_model_versus import NoSkipNet

import matplotlib
for _backend in ("TkAgg", "QtAgg", "GTK3Agg"):
    try:
        matplotlib.use(_backend, force=True)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt

if matplotlib.get_backend().lower() == "agg":
    print("UPOZORENJE: nije dostupan interaktivni backend (samo Agg) -> "
          "prozori se nece prikazati. Provjeri DISPLAY/WSLg ili instaliraj tkinter.")


HERE       = os.path.dirname(os.path.abspath(__file__))
WEIGHTS    = os.path.join(HERE, "results", "NoSkipNet.pt")


def load_model():
    model = NoSkipNet()
    state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def collect_conv_filters(model):
    filters = []
    w1 = model.conv1.weight.detach().cpu().numpy()
    for oc in range(w1.shape[0]):
        filters.append((f"conv1 / filter {oc}", w1[oc, 0]))
    w2 = model.conv2.weight.detach().cpu().numpy()
    for oc in range(w2.shape[0]):
        filters.append((f"conv2 / filter {oc}", w2[oc, 0]))
    return filters


def print_conv_filters(model, filters):
    print("\n" + "=" * 60)
    print("KONVOLUCIJSKI FILTERI  (ukupno {})".format(len(filters)))
    print("=" * 60)
    b1 = model.conv1.bias.detach().cpu().numpy()
    b2 = model.conv2.bias.detach().cpu().numpy()
    biases = list(b1) + list(b2)
    for (name, f), bias in zip(filters, biases):
        print(f"\n{name}  (shape {f.shape}, bias={bias:+.4f})")
        with np.printoptions(precision=4, suppress=True):
            print(f)
        print(f"   mean={f.mean():+.4f}  mean|w|={np.abs(f).mean():.4f}  "
              f"min={f.min():+.4f}  max={f.max():+.4f}")


def plot_conv_filters(filters):
    n = len(filters)
    vmax = max(np.abs(f).max() for _, f in filters)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (name, f) in zip(axes, filters):
        im = ax.imshow(f, cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{name}\n{f.shape[0]}x{f.shape[1]}")
        for (i, j), val in np.ndenumerate(f):
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=8, color="black")
        ax.set_xticks(range(f.shape[1]))
        ax.set_yticks(range(f.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("NoSkipNet — konvolucijski filteri", fontsize=13)
    fig.tight_layout()


def collect_neurons(model):
    neurons = []
    for layer_name, layer in [("fc1", model.fc1), ("out", model.out)]:
        W = layer.weight.detach().cpu().numpy()
        b = layer.bias.detach().cpu().numpy()
        for i in range(W.shape[0]):
            w_in = W[i]
            neurons.append({
                "layer": layer_name,
                "idx": i,
                "label": f"{layer_name}#{i}",
                "w_in": w_in,
                "bias": float(b[i]),
                "mean_w": float(w_in.mean()),
                "mean_abs_w": float(np.abs(w_in).mean()),
                "n_in": w_in.size,
            })
    return neurons


def print_neuron_stats(neurons):
    print("\n" + "=" * 60)
    print("ANALIZA NEURONA (ulazne tezine)")
    print("=" * 60)
    print(f"{'neuron':<10}{'#ulaza':>8}{'bias':>10}{'mean w in':>14}{'mean |w| in':>14}")
    print("-" * 56)
    for nrn in neurons:
        print(f"{nrn['label']:<10}{nrn['n_in']:>8}{nrn['bias']:>+10.4f}"
              f"{nrn['mean_w']:>+14.4f}{nrn['mean_abs_w']:>14.4f}")


def plot_neuron_magnitudes(neurons):
    labels   = [n["label"] for n in neurons]
    mean_abs = [n["mean_abs_w"] for n in neurons]
    mean_w   = [n["mean_w"] for n in neurons]
    x = np.arange(len(labels))
    colors = ["tab:green" if n["layer"] == "fc1" else "tab:purple" for n in neurons]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].bar(x, mean_abs, color=colors)
    axes[0].set_title("Iznos ulaznih tezina po neuronu  (mean |w| in)")
    axes[0].set_ylabel("mean |w| in")
    for xi, v in zip(x, mean_abs):
        axes[0].text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, mean_w, color=colors)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Prosjek (predznak) ulaznih tezina po neuronu  (mean w in)")
    axes[1].set_ylabel("mean w in")
    for xi, v in zip(x, mean_w):
        axes[1].text(xi, v, f"{v:+.3f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color="tab:green", label="fc1"),
                            Patch(color="tab:purple", label="out")])
    fig.suptitle("NoSkipNet — analiza tezina neurona po iznosu", fontsize=13)
    fig.tight_layout()


def plot_neurons_per_layer(neurons, layer_name, ncols):
    sub = [n for n in neurons if n["layer"] == layer_name]
    nrows = int(np.ceil(len(sub) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows),
                             squeeze=False)
    vmax = max(np.abs(n["w_in"]).max() for n in sub)
    for ax in axes.flat:
        ax.axis("off")
    for ax, nrn in zip(axes.flat, sub):
        ax.axis("on")
        w = nrn["w_in"]
        ax.bar(range(len(w)), w,
               color=["tab:red" if v < 0 else "tab:blue" for v in w])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(-vmax * 1.1, vmax * 1.1)
        ax.set_title(f"{nrn['label']}  (bias={nrn['bias']:+.2f})\n"
                     f"mean w in={nrn['mean_w']:+.3f}\n"
                     f"mean |w| in={nrn['mean_abs_w']:.3f}",
                     fontsize=9)
        ax.set_xlabel("ulazni indeks")
        ax.tick_params(labelsize=7)
    fig.suptitle(f"NoSkipNet — neuroni sloja '{layer_name}' "
                 f"(ulazne tezine)", fontsize=13)
    fig.tight_layout()


def main():
    print(f"Ucitavam tezine (read-only): {WEIGHTS}")
    model = load_model()

    filters = collect_conv_filters(model)
    print_conv_filters(model, filters)
    plot_conv_filters(filters)

    neurons = collect_neurons(model)
    print_neuron_stats(neurons)
    plot_neuron_magnitudes(neurons)
    plot_neurons_per_layer(neurons, "fc1", ncols=3)
    plot_neurons_per_layer(neurons, "out", ncols=5)

    print("\nPrikazujem vizualizacije (zatvori prozore za kraj). Nista nije spremljeno.")
    plt.show()


if __name__ == "__main__":
    main()
