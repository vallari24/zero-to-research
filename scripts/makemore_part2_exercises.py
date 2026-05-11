from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "names.txt"
BOUNDARY = "."


@dataclass(frozen=True)
class Config:
    name: str
    block_size: int = 3
    embed_dim: int = 10
    hidden_dim: int = 200
    batch_size: int = 64
    steps: int = 30_000
    lr1: float = 0.1
    lr2: float = 0.01
    lr_decay_step: int = 20_000
    seed: int = 2147483647
    init: str = "careful"
    weight_decay: float = 0.0
    direct: bool = False


def load_words(path: Path = DATA_PATH) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_vocab(words: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set("".join(words)))
    stoi = {BOUNDARY: 0}
    stoi.update({ch: i + 1 for i, ch in enumerate(chars)})
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def split_words(words: list[str], seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    words = list(words)
    random.Random(seed).shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    return words[:n1], words[n1:n2], words[n2:]


def build_dataset(
    words: list[str],
    stoi: dict[str, int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[list[int]] = []
    ys: list[int] = []
    for word in words:
        context = [0] * block_size
        for ch in word + BOUNDARY:
            ix = stoi[ch]
            xs.append(context)
            ys.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def init_params(cfg: Config, vocab_size: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(cfg.seed)
    fan_in = cfg.block_size * cfg.embed_dim

    if cfg.init == "careless":
        c_scale = 1.0
        w1_scale = 1.0
        b1_scale = 1.0
        w2_scale = 1.0
        b2_scale = 1.0
    elif cfg.init == "output_only":
        c_scale = 1.0
        w1_scale = 1.0
        b1_scale = 1.0
        w2_scale = 0.01
        b2_scale = 0.0
    elif cfg.init == "careful":
        c_scale = 1.0
        w1_scale = (5.0 / 3.0) / math.sqrt(fan_in)
        b1_scale = 0.01
        w2_scale = 0.01
        b2_scale = 0.0
    else:
        raise ValueError(f"unknown init: {cfg.init}")

    c = torch.randn((vocab_size, cfg.embed_dim), generator=g) * c_scale
    w1 = torch.randn((fan_in, cfg.hidden_dim), generator=g) * w1_scale
    b1 = torch.randn(cfg.hidden_dim, generator=g) * b1_scale
    w2 = torch.randn((cfg.hidden_dim, vocab_size), generator=g) * w2_scale
    b2 = torch.randn(vocab_size, generator=g) * b2_scale
    params = [c, w1, b1, w2, b2]

    if cfg.direct:
        wskip = torch.randn((fan_in, vocab_size), generator=g) * 0.01
        params.append(wskip)

    for p in params:
        p.requires_grad = True
    return params


def forward(params: list[torch.Tensor], x: torch.Tensor, cfg: Config) -> torch.Tensor:
    c, w1, b1, w2, b2 = params[:5]
    emb = c[x]
    flat = emb.view(emb.shape[0], -1)
    h = torch.tanh(flat @ w1 + b1)
    logits = h @ w2 + b2
    if cfg.direct:
        logits = logits + flat @ params[5]
    return logits


def regularization(params: list[torch.Tensor], weight_decay: float, direct: bool) -> torch.Tensor:
    if weight_decay == 0.0:
        return torch.tensor(0.0)

    # Match the Bengio paper convention: penalize weights and C, not biases.
    reg_params = [params[0], params[1], params[3]]
    if direct:
        reg_params.append(params[5])
    return weight_decay * sum((p * p).mean() for p in reg_params)


@torch.no_grad()
def split_loss(
    params: list[torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
    chunk_size: int = 8192,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, x.shape[0], chunk_size):
        xb = x[start : start + chunk_size]
        yb = y[start : start + chunk_size]
        loss = F.cross_entropy(forward(params, xb, cfg), yb, reduction="sum")
        total += float(loss.item())
        count += yb.numel()
    return total / count


def train_one(
    cfg: Config,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xdev: torch.Tensor,
    ydev: torch.Tensor,
    xte: torch.Tensor,
    yte: torch.Tensor,
    vocab_size: int,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    params = init_params(cfg, vocab_size)
    g = torch.Generator().manual_seed(cfg.seed + 17)
    initial_train_loss = split_loss(params, xtr, ytr, cfg)
    initial_dev_loss = split_loss(params, xdev, ydev, cfg)
    started = time.time()

    for step in range(cfg.steps):
        ix = torch.randint(0, xtr.shape[0], (cfg.batch_size,), generator=g)
        logits = forward(params, xtr[ix], cfg)
        loss = F.cross_entropy(logits, ytr[ix]) + regularization(
            params, cfg.weight_decay, cfg.direct
        )

        for p in params:
            p.grad = None
        loss.backward()

        lr = cfg.lr1 if step < cfg.lr_decay_step else cfg.lr2
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad

    train_loss = split_loss(params, xtr, ytr, cfg)
    dev_loss = split_loss(params, xdev, ydev, cfg)
    test_loss = split_loss(params, xte, yte, cfg)
    elapsed = time.time() - started

    result = {
        **asdict(cfg),
        "parameters": sum(p.numel() for p in params),
        "initial_train_loss": initial_train_loss,
        "initial_dev_loss": initial_dev_loss,
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "test_loss": test_loss,
        "elapsed_sec": elapsed,
    }
    return params, result


def build_trigram_probs(
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    vocab_size: int,
    smoothing: float,
) -> torch.Tensor:
    counts = torch.full((vocab_size, vocab_size, vocab_size), smoothing)
    left = xtr[:, -2]
    right = xtr[:, -1]
    for a, b, c in zip(left.tolist(), right.tolist(), ytr.tolist()):
        counts[a, b, c] += 1.0
    return counts / counts.sum(dim=2, keepdim=True)


@torch.no_grad()
def mixture_loss(
    params: list[torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
    trigram_probs: torch.Tensor,
    alpha: float,
    chunk_size: int = 8192,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, x.shape[0], chunk_size):
        xb = x[start : start + chunk_size]
        yb = y[start : start + chunk_size]
        nn_probs = torch.softmax(forward(params, xb, cfg), dim=1)
        nn_true = nn_probs[torch.arange(yb.numel()), yb]
        tri_true = trigram_probs[xb[:, -2], xb[:, -1], yb]
        p = (1.0 - alpha) * nn_true + alpha * tri_true
        total += float(-p.clamp_min(1e-12).log().sum().item())
        count += yb.numel()
    return total / count


def run_initialization_probe() -> list[dict[str, object]]:
    words = load_words()
    stoi, _ = build_vocab(words)
    train_words, dev_words, test_words = split_words(words)
    xtr, ytr = build_dataset(train_words, stoi, block_size=3)
    xdev, ydev = build_dataset(dev_words, stoi, block_size=3)
    xte, yte = build_dataset(test_words, stoi, block_size=3)
    vocab_size = len(stoi)

    rows = []
    for init in ["careless", "output_only", "careful"]:
        cfg = Config(name=f"init_{init}", init=init, steps=0)
        params = init_params(cfg, vocab_size)
        rows.append(
            {
                "name": cfg.name,
                "init": init,
                "train_loss": split_loss(params, xtr, ytr, cfg),
                "dev_loss": split_loss(params, xdev, ydev, cfg),
                "test_loss": split_loss(params, xte, yte, cfg),
                "uniform_loss": math.log(vocab_size),
            }
        )
    return rows


def experiment_configs() -> list[Config]:
    return [
        Config(
            name="baseline_careful_b3_e10_h200",
            block_size=3,
            embed_dim=10,
            hidden_dim=200,
            batch_size=64,
            steps=40_000,
            lr1=0.1,
            lr2=0.01,
            lr_decay_step=30_000,
        ),
        Config(
            name="wider_b3_e10_h300",
            block_size=3,
            embed_dim=10,
            hidden_dim=300,
            batch_size=64,
            steps=40_000,
            lr1=0.1,
            lr2=0.01,
            lr_decay_step=30_000,
        ),
        Config(
            name="larger_embedding_b3_e16_h300",
            block_size=3,
            embed_dim=16,
            hidden_dim=300,
            batch_size=64,
            steps=45_000,
            lr1=0.08,
            lr2=0.008,
            lr_decay_step=34_000,
        ),
        Config(
            name="longer_context_b4_e12_h300",
            block_size=4,
            embed_dim=12,
            hidden_dim=300,
            batch_size=64,
            steps=50_000,
            lr1=0.08,
            lr2=0.008,
            lr_decay_step=38_000,
        ),
        Config(
            name="longer_context_b5_e12_h300",
            block_size=5,
            embed_dim=12,
            hidden_dim=300,
            batch_size=64,
            steps=50_000,
            lr1=0.08,
            lr2=0.008,
            lr_decay_step=38_000,
        ),
        Config(
            name="long_context_b8_e12_h300",
            block_size=8,
            embed_dim=12,
            hidden_dim=300,
            batch_size=64,
            steps=50_000,
            lr1=0.08,
            lr2=0.008,
            lr_decay_step=38_000,
        ),
    ]


def run_config_set(configs: list[Config]) -> list[dict[str, object]]:
    words = load_words()
    stoi, _ = build_vocab(words)
    train_words, dev_words, test_words = split_words(words)
    rows = []
    for cfg in configs:
        xtr, ytr = build_dataset(train_words, stoi, cfg.block_size)
        xdev, ydev = build_dataset(dev_words, stoi, cfg.block_size)
        xte, yte = build_dataset(test_words, stoi, cfg.block_size)
        _, row = train_one(cfg, xtr, ytr, xdev, ydev, xte, yte, len(stoi))
        rows.append(row)
        print(json.dumps({"event": "finished", **row}), flush=True)
    return rows


def run_bengio_experiments() -> list[dict[str, object]]:
    words = load_words()
    stoi, _ = build_vocab(words)
    train_words, dev_words, test_words = split_words(words)

    base_cfg = Config(
        name="bengio_direct_b5_e12_h300",
        block_size=5,
        embed_dim=12,
        hidden_dim=300,
        batch_size=64,
        steps=50_000,
        lr1=0.08,
        lr2=0.008,
        lr_decay_step=38_000,
        direct=True,
    )
    xtr, ytr = build_dataset(train_words, stoi, base_cfg.block_size)
    xdev, ydev = build_dataset(dev_words, stoi, base_cfg.block_size)
    xte, yte = build_dataset(test_words, stoi, base_cfg.block_size)

    params, direct_row = train_one(base_cfg, xtr, ytr, xdev, ydev, xte, yte, len(stoi))
    direct_row["idea"] = "direct input-to-output connections"
    print(json.dumps({"event": "finished", **direct_row}), flush=True)

    trigram_probs = build_trigram_probs(xtr, ytr, len(stoi), smoothing=0.1)
    mix_rows = []
    for alpha in [i / 20 for i in range(0, 21)]:
        mix_rows.append(
            {
                "alpha_trigram": alpha,
                "dev_loss": mixture_loss(params, xdev, ydev, base_cfg, trigram_probs, alpha),
                "test_loss": mixture_loss(params, xte, yte, base_cfg, trigram_probs, alpha),
            }
        )
    best_mix = min(mix_rows, key=lambda row: row["dev_loss"])
    mix_result = {
        "name": "bengio_mixture_direct_plus_trigram",
        "idea": "mixture of neural model and interpolated trigram-style model",
        "base_dev_loss": direct_row["dev_loss"],
        "base_test_loss": direct_row["test_loss"],
        **best_mix,
        "grid": mix_rows,
    }
    print(json.dumps({"event": "finished", **mix_result}), flush=True)
    return [direct_row, mix_result]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["init", "tune", "bengio", "all"],
        default="all",
    )
    args = parser.parse_args()

    all_results: dict[str, object] = {}
    if args.mode in {"init", "all"}:
        all_results["initialization"] = run_initialization_probe()
    if args.mode in {"tune", "all"}:
        all_results["tuning"] = run_config_set(experiment_configs())
    if args.mode in {"bengio", "all"}:
        all_results["bengio"] = run_bengio_experiments()

    print(json.dumps({"event": "summary", "results": all_results}, indent=2))


if __name__ == "__main__":
    main()
