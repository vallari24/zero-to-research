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
