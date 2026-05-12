# zero-to-research

An open research notebook for rebuilding neural-network systems from first
principles, pressure-testing them with compact experiments, and documenting the
engineering tradeoffs clearly.

## Project thesis

This repo is meant to read like original technical work:
- core learning systems rebuilt from scratch
- one controlled experiment per milestone
- concise notes on what changed, what failed, and what mattered

The point is not to collect study notes. The point is to produce a sequence of
small, defensible artifacts that show implementation ability, experimental
judgment, and technical writing.

## Repo structure

- `notebooks/00_project_template.ipynb`: reusable template for build notes
- `notebooks/01_scalar_autodiff_engine.ipynb`: scalar autodiff engine build log
- `notebooks/02_makemore_bigrams.ipynb`: lecture 2 bigram build + count-vs-neural data-regime experiment
- `src/`: helper code factored out of notebooks
- `data/`: local datasets and generated artifacts
- `docs/02_makemore_bigrams_notes.md`: lecture 2 short bigram note
- `docs/04_makemore_mlp_notes.md`: part 2 note on why count tables do not scale to longer context
- `docs/03_pytorch_basics.md`: short PyTorch basics note
- `docs/rl_additional_resources.md`: filtered RL watch order for RLHF and LLM tuning
- `docs/state-rl.md`: shortlist of promising RL, world-model, and planning project ideas

## Quick start

### Fastest path on this machine

You already have a working Jupyter + `torch` environment in
`/Users/vallari/src/llm-from-first-principles/venv`, so you can reuse it:

```bash
cd /Users/vallari/src/llm-from-first-principles
source venv/bin/activate
jupyter lab /Users/vallari/src/zero-to-research/notebooks/01_scalar_autodiff_engine.ipynb
```

If you want this repo to appear as a named kernel inside Jupyter, register that
environment once:

```bash
cd /Users/vallari/src/llm-from-first-principles
source venv/bin/activate
python -m ipykernel install --user --name zero-to-research --display-name "Python (zero-to-research)"
```

### Dedicated environment for this repo

Your system `python3` is currently Python 3.13 on an Intel Mac. That
combination breaks the `torch` install for this setup.

Why: official macOS Intel (`x86_64`) PyTorch wheels stopped after `torch 2.2.x`,
while Python 3.13 support landed in later PyTorch releases. So there is no
useful overlap for "Python 3.13 + official prebuilt torch on Intel Mac".

Use Python 3.11 or 3.12 instead:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-torch.txt
python -m ipykernel install --user --name zero-to-research --display-name "Python (zero-to-research)"
jupyter lab
```

The first notebook does not require `torch`, so `pip install -r requirements.txt`
is enough for the scalar autodiff work. Later model-building notebooks will use
`requirements-torch.txt`.

On an Intel Mac, `requirements-torch.txt` pins `torch==2.2.2` and `numpy<2`.
That is intentional.

If you want computation-graph visualizations for the autodiff engine, install
the Graphviz system package as well. On macOS with Homebrew:

```bash
brew install graphviz
```

If Jupyter was already open when `graphviz` was installed, restart the kernel
before re-running the visualization cell so `dot` is picked up from `PATH`.

## Research loop

1. Build the smallest correct version of the system.
2. Move reusable pieces into `src/` once the notebook gets noisy.
3. Change one assumption and measure the effect.
4. End with a short finding: what worked, what changed, what surprised me.

## Portfolio pattern

### One controlled experiment per milestone

For each system, change exactly one assumption and document the result in a
small table:

| system | baseline | delta | metric | takeaway |
| --- | --- | --- | --- | --- |
| scalar autodiff engine | raw backward implementation | add finite-difference gradient checks | gradient error | validates the engine instead of trusting it |

This is intentionally small. The consistency is the point. By the end of the
repo, it should read like a sequence of compact research notes instead of a set
of walkthroughs.

## Good next experiments

- scalar autodiff: add a numerical gradient checker for random scalar graphs
- character model: compare count-based and neural bigrams across train-set size
- transformer model: compare context length or parameter count versus loss

## Signals for a team

- I can rebuild core ideas from scratch
- I can turn a reference idea into an original implementation
- I can design a small experiment instead of stopping at reproduction
- I can explain tradeoffs clearly and keep the work organized

## Inspirations

This project is informed by excellent public teaching material, especially
Andrej Karpathy's *Neural Networks: Zero to Hero*. The implementation,
experiments, framing, and writeups here are intended to stand on their own.

References:
- GitHub lectures: https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures
- YouTube playlist: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
