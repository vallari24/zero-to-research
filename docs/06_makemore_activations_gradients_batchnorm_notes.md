# Building Makemore Part 3: Activations, Gradients, and BatchNorm

## Main Idea

Training a neural network is not only about choosing the right architecture.
It is also about keeping the numbers inside the network healthy.

An MLP passes information through a chain of transformations:

```text
input tokens
-> embeddings
-> linear layers
-> nonlinearities
-> logits
-> loss
```

During the forward pass, activations can become too large, too small, or stuck
inside saturated nonlinearities. During the backward pass, gradients can vanish,
explode, or reach different layers at very different scales.

When that happens, the model may look correct on paper but train poorly in
practice.

This note is about learning to inspect that internal health:

- Are activations centered and reasonably scaled?
- Are nonlinearities saturated?
- Are gradients flowing through every layer?
- Are parameter updates large enough to matter but small enough to stay stable?

The central idea is:

```text
deep learning is controlled signal flow
```

Initialization, activation statistics, gradient diagnostics, and Batch
Normalization are all tools for the same job: keeping the forward signal and
backward learning signal in a useful range while the model trains.

![Signal flow through activations and gradients](assets/06_signal_flow.svg)

Read the picture as:

```text
forward signal -> activations -> logits -> loss
loss -> gradients -> earlier layers
```

If the forward signal gets too small or too large, the backward signal usually
gets unhealthy too.

## BatchNorm Intuition

BatchNorm is a way to normalize activations and then let the network learn how
to put the scale back.

![BatchNorm flow](assets/06_batchnorm_flow.svg)

The steps are:

```text
measure batch mean and variance
-> normalize
-> apply learned scale and shift
```

The useful mental model is:

```text
center the activations
keep their spread reasonable
let the model relearn the best scale
```

That is why BatchNorm often makes optimization easier: it keeps the internal
numbers from drifting into bad ranges.
