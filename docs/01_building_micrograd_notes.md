# Building Micrograd: Brief Notes

## Neural network intuition

A neuron is just:

\[
o = f\left(\sum_i w_i x_i + b\right)
\]

Inputs carry information, weights decide importance, bias shifts the total, and
the activation turns that total into an output.

## Chain rule intuition

Backprop is repeated chain rule.

\[
\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}
\]

If `x` affects the loss only through `y`, then its effect must flow through
`y`.

## Operator intuition

For gradients, `+` and `*` behave differently:

- plus operator: if `z = x + y`, then `dz/dx = 1` and `dz/dy = 1`
- so `+` just passes the upstream gradient unchanged to both inputs
- multiply operator: if `z = x * y`, then `dz/dx = y` and `dz/dy = x`
- so `*` sends each input the upstream gradient scaled by the other input

## Manual backprop example

Take:

```python
e = a * b
d = e + c
L = d * f
```

With:

```python
a = 2
b = -3
c = 10
f = -2
```

Forward pass:

```text
e = -6
d = 4
L = -8
```

One simple picture to keep in mind:

```text
a ----\
       (*) ---- e ----\
b ----/               (+) ---- d ---- (*) ---- L
c -------------------/                 /
f ------------------------------------/
```

Forward intuition:
- left to right computes the value
- multiply nodes combine two inputs
- plus nodes collect values
- this is the same basic pattern as a neuron: weighted inputs, a sum, then an output

Backward intuition:
- start at `L` with gradient `1`
- move right to left
- `*` sends gradient scaled by the other input
- `+` copies the gradient to both inputs

Backward pass:

- `dL/dL = 1`
- `dL/dd = f = -2`
- `dL/df = d = 4`
- `d = e + c`, so the plus operator routes `-2` to both inputs:
  `dL/dc = -2` and `dL/de = -2`
- `e = a * b`, so the multiply operator scales by the other side:
  `dL/da = (-2) * (-3) = 6`
  and `dL/db = (-2) * 2 = -4`

## Where those gradients came from

- `dL/dL = 1` because any variable differentiated with respect to itself is `1`
- `L = d * f`, so `dL/dd = f = -2` and `dL/df = d = 4`
- `d = e + c`, so `dd/de = 1` and `dd/dc = 1`
- by chain rule, `dL/de = dL/dd * dd/de = (-2) * 1 = -2`
- by chain rule, `dL/dc = dL/dd * dd/dc = (-2) * 1 = -2`
- `e = a * b`, so `de/da = b = -3` and `de/db = a = 2`
- by chain rule, `dL/da = dL/de * de/da = (-2) * (-3) = 6`
- by chain rule, `dL/db = dL/de * de/db = (-2) * 2 = -4`

That is the core idea of backprop: compute forward once, then send gradients
backward by multiplying local derivatives.

## From scalar values to a small MLP

Neural networks are mathematical expressions. They take input data, combine it
with learned weights and biases, and produce predictions through a forward pass.

The same scalar engine can build a neural network by composing simple pieces:

- `Neuron`: stores weights and a bias, computes `tanh(w * x + b)`
- `Layer`: runs several neurons on the same input, returning several outputs
- `MLP`: stacks layers so each layer's output becomes the next layer's input

For example, `MLP(3, [4, 4, 1])` means:

```text
3 inputs -> 4 neurons -> 4 neurons -> 1 output
```

Training adds one more scalar: the loss. A simple squared-error loss is:

```python
loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
```

The loss asks: how far are the predictions from the targets? It is designed to
be low when the network predicts well and high when it predicts poorly.

The training step is:

```python
for p in model.parameters():
    p.grad = 0.0

loss.backward()

for p in model.parameters():
    p.data += -0.01 * p.grad
```

Gradients are reset because each step needs the slope at the current weights.
Inside one backward pass, gradients should accumulate across graph paths. Across
training steps, old gradients describe old weights, so they must be cleared
before measuring the next direction.

The full loop is: run the forward pass, measure prediction quality with the
loss, backpropagate the loss to get gradients, update the weights, and repeat.
Repeated updates move the parameters toward values that minimize the loss.

## Tiny training experiment

Question: can the tiny MLP fit a four-example toy dataset?

Setup:

```text
model = MLP(3, [4, 4, 1])
learning_rate = 0.05
steps = 50
```

Result:

| step | loss |
| --- | ---: |
| 0 | 5.2305 |
| 10 | 0.2131 |
| 20 | 0.0845 |
| 30 | 0.0502 |
| 40 | 0.0349 |
| 50 | 0.0265 |

The loss falls quickly, which is the first useful signal that the scalar
autodiff engine, backpropagation, and parameter updates are working together.

## Gradient checker

A gradient checker verifies whether the gradients from backprop match an
independent finite-difference estimate:

```text
dL/dp ~= (L(p + eps) - L(p - eps)) / (2 * eps)
```

It is not something to run every training step. It is a debugging tool for
custom autodiff engines, custom layers, custom losses, or manual backward code.

For this tiny MLP, the checker compared all `41` parameter gradients after
training:

| check | result |
| --- | ---: |
| max absolute difference | < 4e-11 |
| mean absolute difference | ~1e-11 |

As a negative control, calling `backward()` twice without clearing `.grad`
caused the checker to fail, with max absolute difference around `8.46e-2`.
That confirms the checker can catch stale accumulated gradients, not just
validate the happy path.
