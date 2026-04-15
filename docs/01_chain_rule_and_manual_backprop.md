# Chain Rule and Manual Backprop

Backprop is just repeated chain rule.

The intuition is simple: if a value affects the loss only through another
intermediate value, then its influence must pass through that path.

\[
\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}
\]

So manual backprop means:
- start at the final output with gradient `1`
- move backward one node at a time
- multiply the upstream gradient by the local derivative

## Example from `01_scalar_autodiff_engine.ipynb`

In the notebook:

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

Manual backprop:

- `dL/dL = 1`
- `L = d * f`, so `dL/dd = f = -2` and `dL/df = d = 4`
- `d = e + c`, so `dd/de = 1` and `dd/dc = 1`
- therefore `dL/de = -2` and `dL/dc = -2`
- `e = a * b`, so `de/da = b = -3` and `de/db = a = 2`
- therefore `dL/da = (-2) * (-3) = 6`
- and `dL/db = (-2) * 2 = -4`

That is the whole idea: each node contributes a small local derivative, and
backprop multiplies those local pieces along the graph.
