from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "names.txt"


def build_dataset(words, stoi, block_size):
    xs, ys = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            xs.append(context)
            ys.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(xs), torch.tensor(ys)


def load_data(block_size=3):
    words = DATA_PATH.read_text().splitlines()
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    xtr, ytr = build_dataset(words[:n1], stoi, block_size)
    xdev, ydev = build_dataset(words[n1:n2], stoi, block_size)
    return xtr, ytr, xdev, ydev, stoi, itos


@torch.no_grad()
def eval_one_hidden(C, W1, b1, W2, b2, X, Y):
    emb = C[X]
    embcat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()


def exercise_01_zero_init():
    Xtr, Ytr, Xdev, Ydev, stoi, itos = load_data()
    vocab_size = len(stoi)
    n_embd = 10
    n_hidden = 200
    fan_in = n_embd * Xtr.shape[1]
    g = torch.Generator().manual_seed(2147483647)

    # Keep the embedding table random. The exercise zeros the MLP weights/biases.
    C = torch.randn((vocab_size, n_embd), generator=g, requires_grad=True)
    W1 = torch.zeros((fan_in, n_hidden), requires_grad=True)
    b1 = torch.zeros(n_hidden, requires_grad=True)
    W2 = torch.zeros((n_hidden, vocab_size), requires_grad=True)
    b2 = torch.zeros(vocab_size, requires_grad=True)
    parameters = [C, W1, b1, W2, b2]
    names = ["C", "W1", "b1", "W2", "b2"]

    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss0 = F.cross_entropy(logits, Yb)
    for p in parameters:
        p.grad = None
    loss0.backward()

    grad0 = {name: p.grad.abs().max().item() for name, p in zip(names, parameters)}
    act0 = {
        "emb_abs_max": emb.abs().max().item(),
        "hpreact_abs_max": hpreact.abs().max().item(),
        "h_abs_max": h.abs().max().item(),
        "logits_abs_max": logits.abs().max().item(),
    }

    checkpoints = {}
    max_steps = 5000
    batch_size = 32
    for i in range(max_steps + 1):
        if i in {0, 1, 10, 100, 1000, 5000}:
            checkpoints[i] = eval_one_hidden(C, W1, b1, W2, b2, Xdev, Ydev)
        if i == max_steps:
            break
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        emb = C[Xb]
        embcat = emb.view(emb.shape[0], -1)
        h = torch.tanh(embcat @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Yb)
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -0.1 * p.grad

    with torch.no_grad():
        final_norms = {name: p.norm().item() for name, p in zip(names, parameters)}
        probs = F.softmax(b2, dim=0)
        top = torch.topk(probs, 8)
        top_b2_probs = [(itos[i.item()], v.item()) for v, i in zip(top.values, top.indices)]

    return {
        "initial_loss": loss0.item(),
        "initial_activation_max_abs": act0,
        "initial_gradient_max_abs": grad0,
        "dev_loss_checkpoints": checkpoints,
        "final_parameter_norms": final_norms,
        "top_output_bias_probabilities": top_b2_probs,
    }


class Linear:
    def __init__(self, fan_in, fan_out, generator, bias=True):
        self.W = torch.randn((fan_in, fan_out), generator=generator) / (fan_in**0.5)
        self.b = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        out = x @ self.W
        self.out = out + self.b if self.b is not None else out
        return self.out

    def parameters(self):
        return [self.W] + ([] if self.b is None else [self.b])


class BatchNorm1d:
    def __init__(self, dim, momentum=0.01, eps=1e-5):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones((1, dim))
        self.beta = torch.zeros((1, dim))
        self.running_mean = torch.zeros((1, dim))
        self.running_var = torch.ones((1, dim))

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
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


def forward_layers(layers, x):
    for layer in layers:
        x = layer(x)
    return x


def exercise_02_fold_batchnorm():
    Xtr, Ytr, Xdev, Ydev, stoi, _ = load_data()
    vocab_size = len(stoi)
    n_embd = 10
    n_hidden = 64
    fan_in = n_embd * Xtr.shape[1]
    g = torch.Generator().manual_seed(2147483647)

    C = torch.randn((vocab_size, n_embd), generator=g)
    layers = [
        Linear(fan_in, n_hidden, g),
        BatchNorm1d(n_hidden),
        Tanh(),
        Linear(n_hidden, n_hidden, g),
        BatchNorm1d(n_hidden),
        Tanh(),
        Linear(n_hidden, vocab_size, g),
        BatchNorm1d(vocab_size),
    ]
    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    for p in parameters:
        p.requires_grad = True

    max_steps = 3000
    batch_size = 64
    for i in range(max_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        emb = C[Xb]
        x = emb.view(emb.shape[0], -1)
        logits = forward_layers(layers, x)
        loss = F.cross_entropy(logits, Yb)
        for p in parameters:
            p.grad = None
        loss.backward()
        lr = 0.1 if i < 2000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

    for layer in layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = False

    @torch.no_grad()
    def infer_original(X):
        emb = C[X]
        x = emb.view(emb.shape[0], -1)
        return forward_layers(layers, x)

    def fold(linear, bn):
        scale = bn.gamma / torch.sqrt(bn.running_var + bn.eps)
        W_fold = linear.W * scale
        b = linear.b if linear.b is not None else torch.zeros_like(bn.running_mean).squeeze(0)
        b_fold = (b - bn.running_mean.squeeze(0)) * scale.squeeze(0) + bn.beta.squeeze(0)
        return W_fold, b_fold

    lin1, bn1, _tanh1, lin2, bn2, _tanh2, lin3, bn3 = layers
    W1f, b1f = fold(lin1, bn1)
    W2f, b2f = fold(lin2, bn2)
    W3f, b3f = fold(lin3, bn3)

    @torch.no_grad()
    def infer_folded(X):
        emb = C[X]
        x = emb.view(emb.shape[0], -1)
        x = torch.tanh(x @ W1f + b1f)
        x = torch.tanh(x @ W2f + b2f)
        return x @ W3f + b3f

    with torch.no_grad():
        sample = Xdev[:256]
        logits_original = infer_original(sample)
        logits_folded = infer_folded(sample)
        return {
            "dev_loss_original_bn": F.cross_entropy(infer_original(Xdev), Ydev).item(),
            "dev_loss_folded_no_bn": F.cross_entropy(infer_folded(Xdev), Ydev).item(),
            "max_abs_logit_diff_256": (logits_original - logits_folded).abs().max().item(),
            "mean_abs_logit_diff_256": (logits_original - logits_folded).abs().mean().item(),
            "argmax_agreement_256": (
                logits_original.argmax(1) == logits_folded.argmax(1)
            ).float().mean().item(),
            "folded_shapes": {
                "W1_fold": tuple(W1f.shape),
                "b1_fold": tuple(b1f.shape),
                "W2_fold": tuple(W2f.shape),
                "b2_fold": tuple(b2f.shape),
                "W3_fold": tuple(W3f.shape),
                "b3_fold": tuple(b3f.shape),
            },
        }


if __name__ == "__main__":
    torch.set_num_threads(4)
    print("E01")
    print(exercise_01_zero_init())
    print("E02")
    print(exercise_02_fold_batchnorm())
