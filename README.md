# zero-to-research

Building neural networks from first principles while following Andrej Karpathy's
*Neural Networks: Zero to Hero* course and adding one original experiment per
lecture.

Sources:
- GitHub lectures: https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures
- YouTube playlist: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

## Why this repo exists

This repo is not a transcript of the course. It uses the course as a backbone
for building intuition, then adds a small research delta each time so the work
shows judgment, experimentation, and writing.

For each lecture I will ship:
- a clean reproduction of the core idea
- one small extension or ablation
- a short note on what changed and what I learned

## Repo structure

- `notebooks/00_lecture_template.ipynb`: reusable template for any lecture
- `notebooks/01_micrograd_walkthrough.ipynb`: Lecture 1 starter notebook
- `src/`: helper code factored out of notebooks
- `data/`: local datasets and generated artifacts

## Quick start

### Fastest path on this machine

You already have a working Jupyter + `torch` environment in
`/Users/vallari/src/llm-from-first-principles/venv`, so you can reuse it:

```bash
cd /Users/vallari/src/llm-from-first-principles
source venv/bin/activate
jupyter lab /Users/vallari/src/zero-to-research/notebooks/01_micrograd_walkthrough.ipynb
```

If you want this repo to appear as a named kernel inside Jupyter, register that
environment once:

```bash
cd /Users/vallari/src/llm-from-first-principles
source venv/bin/activate
python -m ipykernel install --user --name zero-to-research --display-name "Python (zero-to-research)"
```

### Clean env for this repo

Your system `python3` is currently Python 3.13 on an Intel Mac. That is the
combination that breaks the `torch` install you hit.

Why: official macOS Intel (`x86_64`) PyTorch wheels stopped after `torch 2.2.x`,
while Python 3.13 support landed in later PyTorch releases. So there is no
useful overlap for "Python 3.13 + official prebuilt torch on Intel Mac".

If you want a dedicated environment for this repo on this machine, create it
with Python 3.11 or 3.12 instead.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-torch.txt
python -m ipykernel install --user --name zero-to-research --display-name "Python (zero-to-research)"
jupyter lab
```

If `python3.11` is not on your machine yet, reuse the existing
`llm-from-first-principles/venv` for now. That is the lowest-friction option.

Lecture 1 does not require `torch`, so `pip install -r requirements.txt` is
enough for the `micrograd` notebook. Later lectures will need
`requirements-torch.txt`.

On an Intel Mac, `requirements-torch.txt` pins `torch==2.2.2` and `numpy<2`.
That is intentional.

If you want computation-graph visualizations for `micrograd`, install Graphviz
on your machine as well.

## Working rule for every lecture

1. Reproduce the lecture faithfully.
2. Move reusable code into `src/` once the notebook becomes noisy.
3. Add one original delta.
4. End with a short finding: what worked, what changed, what surprised me.

## Small portfolio idea

### One Extra Experiment Per Lecture

For every lecture, change exactly one assumption and document the result in a
small table:

| lecture | baseline | delta | metric | takeaway |
| --- | --- | --- | --- | --- |
| 01 micrograd | scalar autodiff engine | add finite-difference gradient check | gradient error | validates the implementation instead of trusting it |

This is intentionally small. The consistency is the point. By the end of the
series, the repo reads like a sequence of compact research notes instead of a
set of copied notebooks.

## Good first deltas

- Lecture 1: add a numerical gradient checker for random scalar graphs
- `makemore`: compare embedding size or hidden size on validation loss
- transformer lectures: compare context length or parameter count versus loss

## What I want a team to see here

- I can rebuild core ideas from scratch
- I can design a small experiment instead of only copying code
- I can explain tradeoffs clearly and keep the work organized
