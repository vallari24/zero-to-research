"""Utilities for lecture 2 makemore-style bigram models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

BOUNDARY_TOKEN = "."


@dataclass(frozen=True)
class BigramVocab:
    """Character vocabulary with a shared start/end token."""

    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def size(self) -> int:
        return len(self.stoi)


def load_words(path: str | Path) -> list[str]:
    """Load newline-delimited words from disk."""

    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def build_vocab(words: Sequence[str]) -> BigramVocab:
    """Create a sorted character vocabulary with `.` as token 0."""

    chars = sorted({ch for word in words for ch in word})
    stoi = {BOUNDARY_TOKEN: 0}
    for idx, ch in enumerate(chars, start=1):
        stoi[ch] = idx
    itos = {idx: ch for ch, idx in stoi.items()}
    return BigramVocab(stoi=stoi, itos=itos)


def split_words(
    words: Sequence[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Shuffle and split words into train/val/test partitions."""

    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1")

    shuffled = list(words)
    if len(shuffled) < 3:
        raise ValueError("Need at least three words for train/val/test splits")

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = max(1, min(n_total - 2, int(n_total * train_frac)))
    n_val = max(1, min(n_total - n_train - 1, int(n_total * val_frac)))

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    if not test:
        raise ValueError("Split produced an empty test set")

    return {"train": train, "val": val, "test": test}


def iter_bigrams(words: Iterable[str]) -> Iterable[tuple[str, str]]:
    """Yield adjacent character pairs with boundary tokens attached."""

    for word in words:
        chars = [BOUNDARY_TOKEN, *word, BOUNDARY_TOKEN]
        yield from zip(chars, chars[1:])


def iter_trigrams(words: Iterable[str]) -> Iterable[tuple[str, str, str]]:
    """Yield trigram transitions with two boundary tokens of context."""

    for word in words:
        chars = [BOUNDARY_TOKEN, BOUNDARY_TOKEN, *word, BOUNDARY_TOKEN]
        yield from zip(chars, chars[1:], chars[2:])


def count_bigram_matrix(
    words: Sequence[str],
    vocab: BigramVocab,
    smoothing: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw bigram counts and row-normalized probabilities."""

    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    counts = torch.zeros((vocab.size, vocab.size), dtype=torch.float32)
    for ch1, ch2 in iter_bigrams(words):
        counts[vocab.stoi[ch1], vocab.stoi[ch2]] += 1

    counts = counts + smoothing
    probs = counts / counts.sum(dim=1, keepdim=True)
    return counts, probs


def count_trigram_tensor(
    words: Sequence[str],
    vocab: BigramVocab,
    smoothing: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return trigram counts and conditional probabilities."""

    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    counts = torch.zeros((vocab.size, vocab.size, vocab.size), dtype=torch.float32)
    for ch1, ch2, ch3 in iter_trigrams(words):
        counts[vocab.stoi[ch1], vocab.stoi[ch2], vocab.stoi[ch3]] += 1

    counts = counts + smoothing
    probs = counts / counts.sum(dim=2, keepdim=True)
    return counts, probs


def build_bigram_dataset(
    words: Sequence[str],
    vocab: BigramVocab,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert bigrams into integer-index training pairs."""

    xs: list[int] = []
    ys: list[int] = []
    for ch1, ch2 in iter_bigrams(words):
        xs.append(vocab.stoi[ch1])
        ys.append(vocab.stoi[ch2])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def build_trigram_dataset(
    words: Sequence[str],
    vocab: BigramVocab,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert trigram transitions into integer-index training triples."""

    x1s: list[int] = []
    x2s: list[int] = []
    ys: list[int] = []
    for ch1, ch2, ch3 in iter_trigrams(words):
        x1s.append(vocab.stoi[ch1])
        x2s.append(vocab.stoi[ch2])
        ys.append(vocab.stoi[ch3])
    return (
        torch.tensor(x1s, dtype=torch.long),
        torch.tensor(x2s, dtype=torch.long),
        torch.tensor(ys, dtype=torch.long),
    )


def average_negative_log_likelihood(
    words: Sequence[str],
    probs: torch.Tensor,
    vocab: BigramVocab,
) -> float:
    """Compute average NLL in natural-log units."""

    total_log_prob = 0.0
    total_pairs = 0

    for ch1, ch2 in iter_bigrams(words):
        prob = probs[vocab.stoi[ch1], vocab.stoi[ch2]].clamp_min(1e-12)
        total_log_prob += torch.log(prob).item()
        total_pairs += 1

    return -total_log_prob / total_pairs


def average_trigram_negative_log_likelihood(
    words: Sequence[str],
    probs: torch.Tensor,
    vocab: BigramVocab,
) -> float:
    """Compute average trigram NLL in natural-log units."""

    total_log_prob = 0.0
    total_triples = 0

    for ch1, ch2, ch3 in iter_trigrams(words):
        prob = probs[vocab.stoi[ch1], vocab.stoi[ch2], vocab.stoi[ch3]].clamp_min(1e-12)
        total_log_prob += torch.log(prob).item()
        total_triples += 1

    return -total_log_prob / total_triples


def sample_from_prob_matrix(
    probs: torch.Tensor,
    vocab: BigramVocab,
    num_samples: int = 20,
    seed: int = 2147483647,
) -> list[str]:
    """Sample words by repeatedly drawing the next character."""

    generator = torch.Generator().manual_seed(seed)
    samples: list[str] = []

    for _ in range(num_samples):
        out: list[str] = []
        idx = vocab.stoi[BOUNDARY_TOKEN]

        while True:
            idx = torch.multinomial(
                probs[idx],
                num_samples=1,
                replacement=True,
                generator=generator,
            ).item()
            if idx == vocab.stoi[BOUNDARY_TOKEN]:
                break
            out.append(vocab.itos[idx])

        samples.append("".join(out))

    return samples


def sample_from_trigram_tensor(
    probs: torch.Tensor,
    vocab: BigramVocab,
    num_samples: int = 20,
    seed: int = 2147483647,
) -> list[str]:
    """Sample words from a trigram model with two-character context."""

    generator = torch.Generator().manual_seed(seed)
    samples: list[str] = []

    for _ in range(num_samples):
        out: list[str] = []
        idx1 = vocab.stoi[BOUNDARY_TOKEN]
        idx2 = vocab.stoi[BOUNDARY_TOKEN]

        while True:
            idx3 = torch.multinomial(
                probs[idx1, idx2],
                num_samples=1,
                replacement=True,
                generator=generator,
            ).item()
            if idx3 == vocab.stoi[BOUNDARY_TOKEN]:
                break
            out.append(vocab.itos[idx3])
            idx1, idx2 = idx2, idx3

        samples.append("".join(out))

    return samples


def train_neural_bigram(
    train_words: Sequence[str],
    val_words: Sequence[str],
    vocab: BigramVocab,
    *,
    num_steps: int = 300,
    learning_rate: float = 30.0,
    weight_decay: float = 1e-2,
    seed: int = 42,
    track_every: int = 25,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]], dict[str, float]]:
    """Train a lookup-table neural bigram model."""

    xs, ys = build_bigram_dataset(train_words, vocab)
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn((vocab.size, vocab.size), generator=generator, requires_grad=True)
    history: list[dict[str, float]] = []

    for step in range(num_steps):
        loss = F.cross_entropy(logits[xs], ys) + weight_decay * (logits**2).mean()

        logits.grad = None
        loss.backward()

        with torch.no_grad():
            logits -= learning_rate * logits.grad

        if step == 0 or (step + 1) % track_every == 0 or step == num_steps - 1:
            probs = torch.softmax(logits.detach(), dim=1)
            history.append(
                {
                    "step": step + 1,
                    "train_loss": float(loss.item()),
                    "train_nll": average_negative_log_likelihood(train_words, probs, vocab),
                    "val_nll": average_negative_log_likelihood(val_words, probs, vocab),
                }
            )

    probs = torch.softmax(logits.detach(), dim=1)
    metrics = {
        "train_nll": average_negative_log_likelihood(train_words, probs, vocab),
        "val_nll": average_negative_log_likelihood(val_words, probs, vocab),
    }
    return logits.detach(), probs, history, metrics


def evaluate_data_regimes(
    train_words: Sequence[str],
    val_words: Sequence[str],
    vocab: BigramVocab,
    *,
    fractions: Sequence[float] = (0.01, 0.05, 0.25, 1.0),
    smoothing: float = 1.0,
    num_steps: int = 250,
    learning_rate: float = 30.0,
    weight_decay: float = 1e-2,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Compare count and neural bigrams as the train set shrinks or grows."""

    if not fractions:
        raise ValueError("fractions must contain at least one value")

    results: list[dict[str, float]] = []

    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fractions must be in (0, 1]")

        subset_size = max(1, int(len(train_words) * fraction))
        subset = list(train_words[:subset_size])

        _, count_probs = count_bigram_matrix(subset, vocab, smoothing=smoothing)
        _, neural_probs, _, neural_metrics = train_neural_bigram(
            subset,
            val_words,
            vocab,
            num_steps=num_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        )

        results.append(
            {
                "train_fraction": float(fraction),
                "train_examples": subset_size,
                "count_train_nll": average_negative_log_likelihood(subset, count_probs, vocab),
                "count_val_nll": average_negative_log_likelihood(val_words, count_probs, vocab),
                "neural_train_nll": neural_metrics["train_nll"],
                "neural_val_nll": neural_metrics["val_nll"],
            }
        )

    return results
