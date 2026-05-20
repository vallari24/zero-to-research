from __future__ import annotations

import math
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "names.txt"
ASSET_DIR = ROOT / "docs" / "assets"


def build_dataset(words, stoi, block_size):
    xs, ys = [], []
    for word in words:
        context = [0] * block_size
        for ch in word + ".":
            ix = stoi[ch]
            xs.append(context)
            ys.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(xs), torch.tensor(ys)


def load_data(block_size=3):
    words = DATA_PATH.read_text().splitlines()
    chars = sorted(set("".join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    xtr, ytr = build_dataset(words[:n1], stoi, block_size)
    return xtr, ytr, len(stoi), block_size


XTR, YTR, VOCAB_SIZE, BLOCK_SIZE = load_data()


class Linear:
    def __init__(self, fan_in, fan_out, generator, *, gain=1.0, weight_scale=None, bias=True, name="linear"):
        std = gain / math.sqrt(fan_in) if weight_scale is None else weight_scale
        self.W = torch.randn((fan_in, fan_out), generator=generator) * std
        self.b = torch.zeros(fan_out) if bias else None
        self.name = name
        self.out = None

    def __call__(self, x):
        self.out = x @ self.W
        if self.b is not None:
            self.out = self.out + self.b
        return self.out

    def parameters(self):
        params = [(f"{self.name}.W", self.W)]
        if self.b is not None:
            params.append((f"{self.name}.b", self.b))
        return params


class BatchNorm1d:
    def __init__(self, dim, *, momentum=0.01, eps=1e-5, name="bn"):
        self.gamma = torch.ones((1, dim))
        self.beta = torch.zeros((1, dim))
        self.running_mean = torch.zeros((1, dim))
        self.running_var = torch.ones((1, dim))
        self.momentum = momentum
        self.eps = eps
        self.training = True
        self.name = name
        self.out = None

    def __call__(self, x):
        if self.training:
            mean = x.mean(0, keepdim=True)
            var = x.var(0, keepdim=True, unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        self.out = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        return self.out

    def parameters(self):
        return [(f"{self.name}.gamma", self.gamma), (f"{self.name}.beta", self.beta)]


class Tanh:
    def __init__(self, name="tanh"):
        self.name = name
        self.out = None

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Identity:
    def __init__(self, name="identity"):
        self.name = name
        self.out = None

    def __call__(self, x):
        self.out = x
        return self.out

    def parameters(self):
        return []


def make_model(
    *,
    seed=2147483647,
    n_embd=10,
    n_hidden=100,
    n_layers=5,
    gain=5 / 3,
    activation="tanh",
    use_bn=False,
    final_weight_scale=0.01,
):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((VOCAB_SIZE, n_embd), generator=g)
    layers = []
    fan_in = BLOCK_SIZE * n_embd
    for i in range(n_layers):
        layers.append(
            Linear(fan_in, n_hidden, g, gain=gain, bias=not use_bn, name=f"linear{i + 1}")
        )
        if use_bn:
            layers.append(BatchNorm1d(n_hidden, name=f"bn{i + 1}"))
        if activation == "identity":
            layers.append(Identity(name=f"identity{i + 1}"))
        else:
            layers.append(Tanh(name=f"tanh{i + 1}"))
        fan_in = n_hidden
    layers.append(Linear(fan_in, VOCAB_SIZE, g, weight_scale=final_weight_scale, name="linear_out"))

    params = [("C", C)]
    for layer in layers:
        params.extend(layer.parameters())
    for _, p in params:
        p.requires_grad = True
    return {"C": C, "layers": layers, "params": params, "generator": g}


def forward(model, X, *, retain=False):
    emb = model["C"][X]
    x = emb.view(emb.shape[0], -1)
    for layer in model["layers"]:
        x = layer(x)
        if retain and getattr(layer, "out", None) is not None:
            layer.out.retain_grad()
    return x


def zero_grad(model):
    for _, p in model["params"]:
        p.grad = None


def snapshot(**config):
    model = make_model(**config)
    g = model["generator"]
    ix = torch.randint(0, XTR.shape[0], (64,), generator=g)
    Xb, Yb = XTR[ix], YTR[ix]
    logits = forward(model, Xb, retain=True)
    loss = F.cross_entropy(logits, Yb)
    zero_grad(model)
    loss.backward()
    return {"model": model, "loss": loss.item()}


def monitor_layers(snap, kinds):
    return [layer for layer in snap["model"]["layers"] if isinstance(layer, kinds)]


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(path.relative_to(ROOT))


def activation_gain_sweep():
    snaps = {
        "gain=1.0": snapshot(gain=1.0),
        "gain=5/3": snapshot(gain=5 / 3),
        "gain=3.0": snapshot(gain=3.0),
    }
    ncols = 5
    plt.figure(figsize=(16, 8))
    plot_i = 1
    for label, snap in snaps.items():
        for layer in monitor_layers(snap, (Tanh,)):
            values = layer.out.detach().view(-1)
            sat = (values.abs() > 0.97).float().mean().item() * 100
            ax = plt.subplot(len(snaps), ncols, plot_i)
            ax.hist(values.tolist(), bins=50, range=(-1, 1), color="#4C78A8", alpha=0.85)
            ax.set_title(f"{label}\n{layer.name}: sat {sat:.1f}%", fontsize=9)
            ax.set_xlim(-1, 1)
            ax.grid(alpha=0.15)
            plot_i += 1
    savefig(ASSET_DIR / "06_diagnostic_activation_gain_sweep.png")


def gradient_comparison():
    snaps = {
        "tanh gain=5/3": snapshot(gain=5 / 3, activation="tanh"),
        "identity gain=5/3": snapshot(gain=5 / 3, activation="identity"),
        "tanh gain=3": snapshot(gain=3.0, activation="tanh"),
    }
    ncols = 5
    plt.figure(figsize=(16, 8))
    plot_i = 1
    for label, snap in snaps.items():
        for layer in monitor_layers(snap, (Tanh, Identity)):
            grad = layer.out.grad.detach().view(-1)
            ax = plt.subplot(len(snaps), ncols, plot_i)
            ax.hist(grad.tolist(), bins=50, color="#54A24B", alpha=0.85)
            ax.set_title(f"{label}\n{layer.name}: std {grad.std().item():.1e}", fontsize=9)
            ax.grid(alpha=0.15)
            plot_i += 1
    savefig(ASSET_DIR / "06_diagnostic_gradient_comparison.png")


def grad_data_ratio():
    snap = snapshot(gain=5 / 3, final_weight_scale=0.01)
    labels, ratios = [], []
    for name, p in snap["model"]["params"]:
        if p.ndim == 2 and p.grad is not None:
            ratio = (p.grad.std() / (p.data.std() + 1e-20)).item()
            labels.append(name)
            ratios.append(math.log10(max(ratio, 1e-20)))
    plt.figure(figsize=(10, 4))
    plt.bar(labels, ratios, color="#B279A2")
    plt.axhline(-3, color="black", linestyle="--", linewidth=1, label="1e-3 reference")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("log10(std(grad) / std(data))")
    plt.title("Parameter grad:data ratio after one backward pass")
    plt.legend()
    savefig(ASSET_DIR / "06_diagnostic_grad_data_ratio.png")


def train_update_ratio(*, lr=0.1, max_steps=250, gain=5 / 3, use_bn=False):
    model = make_model(gain=gain, use_bn=use_bn)
    g = model["generator"]
    weight_names = [name for name, p in model["params"] if p.ndim == 2]
    ud = []
    for _ in range(max_steps):
        ix = torch.randint(0, XTR.shape[0], (64,), generator=g)
        Xb, Yb = XTR[ix], YTR[ix]
        logits = forward(model, Xb)
        loss = F.cross_entropy(logits, Yb)
        zero_grad(model)
        loss.backward()
        with torch.no_grad():
            row = []
            for _, p in model["params"]:
                if p.ndim == 2:
                    ratio = ((lr * p.grad).std() / (p.data.std() + 1e-20)).item()
                    row.append(math.log10(max(ratio, 1e-20)))
            ud.append(row)
            for _, p in model["params"]:
                p += -lr * p.grad
    return {"ud": ud, "weight_names": weight_names}


def update_data_lr_sweep():
    runs = {
        "lr=0.01": train_update_ratio(lr=0.01),
        "lr=0.1": train_update_ratio(lr=0.1),
        "lr=1.0": train_update_ratio(lr=1.0),
    }
    plt.figure(figsize=(10, 4))
    for label, run in runs.items():
        means = [sum(row) / len(row) for row in run["ud"]]
        plt.plot(means, label=label)
    plt.axhline(-3, color="black", linestyle="--", linewidth=1, label="1e-3 reference")
    plt.xlabel("training step")
    plt.ylabel("mean log10 update:data")
    plt.title("Learning-rate sweep via update:data ratio")
    plt.legend()
    savefig(ASSET_DIR / "06_diagnostic_update_data_lr_sweep.png")


def update_data_per_layer():
    run = train_update_ratio(lr=0.1)
    plt.figure(figsize=(12, 4))
    for j, name in enumerate(run["weight_names"]):
        plt.plot([row[j] for row in run["ud"]], label=name, alpha=0.85)
    plt.axhline(-3, color="black", linestyle="--", linewidth=1, label="1e-3 reference")
    plt.xlabel("training step")
    plt.ylabel("log10 update:data")
    plt.title("Per-parameter update:data ratio at lr=0.1")
    plt.legend(ncol=3, fontsize=8)
    savefig(ASSET_DIR / "06_diagnostic_update_data_per_layer.png")


def batchnorm_comparison():
    snaps = {
        "gain=3 no BN": snapshot(gain=3.0, use_bn=False),
        "gain=3 with BN": snapshot(gain=3.0, use_bn=True),
    }
    plt.figure(figsize=(16, 5))
    plot_i = 1
    for label, snap in snaps.items():
        for layer in monitor_layers(snap, (Tanh,)):
            values = layer.out.detach().view(-1)
            sat = (values.abs() > 0.97).float().mean().item() * 100
            ax = plt.subplot(len(snaps), 5, plot_i)
            ax.hist(values.tolist(), bins=50, range=(-1, 1), color="#4C78A8", alpha=0.85)
            ax.set_title(f"{label}\n{layer.name}: sat {sat:.1f}%", fontsize=9)
            ax.set_xlim(-1, 1)
            ax.grid(alpha=0.15)
            plot_i += 1
    savefig(ASSET_DIR / "06_diagnostic_batchnorm_activation_compare.png")

    runs = {
        "gain=3 no BN": train_update_ratio(lr=0.1, gain=3.0, use_bn=False),
        "gain=3 with BN": train_update_ratio(lr=0.1, gain=3.0, use_bn=True),
    }
    plt.figure(figsize=(10, 4))
    for label, run in runs.items():
        means = [sum(row) / len(row) for row in run["ud"]]
        plt.plot(means, label=label)
    plt.axhline(-3, color="black", linestyle="--", linewidth=1, label="1e-3 reference")
    plt.xlabel("training step")
    plt.ylabel("mean log10 update:data")
    plt.title("BatchNorm makes gain=3 less destructive")
    plt.legend()
    savefig(ASSET_DIR / "06_diagnostic_batchnorm_update_compare.png")


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    activation_gain_sweep()
    gradient_comparison()
    grad_data_ratio()
    update_data_lr_sweep()
    update_data_per_layer()
    batchnorm_comparison()


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
