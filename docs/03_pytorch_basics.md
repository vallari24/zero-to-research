# PyTorch Basics

## Torch

`torch` is the main PyTorch library. It gives you tensors, fast numerical
operations, autograd, and the core tools for building neural networks.

## Tensor

A tensor is a container of numbers. It can be a scalar, a vector, a matrix, or
a higher-dimensional array.

## Shape

`shape` tells you the size along each axis. For example, `(28, 28)` means a
2D tensor with 28 rows and 28 columns.

## Datatype

`dtype` tells you what kind of numbers the tensor stores, such as
`torch.float32` or `torch.int64`. Learnable weights are usually floats.

## Requires Grad

`requires_grad=True` tells PyTorch to track how the loss depends on a tensor.
Turn it on for weights you want to learn.

PyTorch does this by building a small computation graph during the forward
pass. Each new tensor remembers which operation created it, so autograd can
walk backward through that graph and apply the chain rule.

Example:

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(4.0, requires_grad=True)

y = a + b
z = x * y

print(y.grad_fn)
print(z.grad_fn)
```

Small visual:

```text
a ----\
       (+) ---- y ----\
b ----/               (*) ---- z
x --------------------/
```

Here:

- `y` remembers it came from `a + b`
- `z` remembers it came from `x * y`
- `z.grad_fn` points to the operation that created `z`

That stored graph is what lets `z.backward()` send gradients back to `x`, `a`,
and `b`.

## Autograd

Autograd is PyTorch's automatic differentiation system. It builds the forward
graph, then uses the chain rule to compute gradients during `backward()`.

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

y = a * b
y.backward()

print(a.grad)  # 3
print(b.grad)  # 2
```

Here `y = a * b`, so:

- `dy/da = b`
- `dy/db = a`

## Dim Collapse

For operations like `sum` or `mean`, `dim` tells PyTorch which axis to
collapse.

If

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
```

then:

- `x.sum(dim=0)` combines down the rows and keeps the columns: `[5, 7, 9]`
- `x.sum(dim=1)` combines across the columns and keeps the rows: `[6, 15]`

`softmax` uses `dim` too, but it does not collapse anything. It normalizes
along that axis.

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

row_probs = torch.softmax(x, dim=1)
print(row_probs)
```

Example output:

```python
tensor([[0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652]])
```

This turns each row into a probability distribution, so each row sums to `1`.

```python
col_probs = torch.softmax(x, dim=0)
print(col_probs)
```

Example output:

```python
tensor([[0.0474, 0.0474, 0.0474],
        [0.9526, 0.9526, 0.9526]])
```

This turns each column into a probability distribution, so each column sums to
`1`.

## torch.gather

`torch.gather(input, dim, index)` picks values from `input` along `dim`, using
the positions stored in `index`.

```python
x = torch.tensor([[10, 20, 30],
                  [40, 50, 60]])

index = torch.tensor([[2, 1],
                      [0, 2]])

out = torch.gather(x, dim=1, index=index)
print(out)
```

Result:

```python
tensor([[30, 20],
        [40, 60]])
```

Why:

- row 0 picks columns `2` and `1` -> `[30, 20]`
- row 1 picks columns `0` and `2` -> `[40, 60]`

So:

- `dim=1` means pick along columns
- `dim=0` would mean pick along rows
- `index` tells PyTorch which positions to pull from that axis

## y = wx + b

A linear layer starts with:

```python
y = w * x + b
```

For one input and one output, `w`, `x`, and `b` can all be scalars.

```python
x = torch.tensor(4.0)
w = torch.tensor(2.0)
b = torch.tensor(1.0)

y = w * x + b
print(y)  # 9
```

With many inputs, PyTorch does the same idea using tensor operations. The
matrix version is:

```python
y = x @ w.T + b
```

Example:

```python
x = torch.tensor([[1.0, 2.0, 3.0]])

w = torch.tensor([[0.2, -0.5, 1.0],
                  [1.5,  0.3, -0.7]])

b = torch.tensor([0.1, -0.2])

y = x @ w.T + b
print(y)
```

Here:

- `x` has shape `(1, 3)` -> one example, three input features
- `w` has shape `(2, 3)` -> two neurons, three weights each
- `b` has shape `(2,)` -> one bias per neuron
- `y` has shape `(1, 2)` -> one example, two outputs

## nn.Linear

`nn.Linear(in_features, out_features)` is PyTorch's built-in linear layer. It
stores `weight` and `bias` tensors and computes the same operation:

```python
y = x @ weight.T + bias
```

Example:

```python
import torch.nn as nn

layer = nn.Linear(3, 2)
x = torch.tensor([[1.0, 2.0, 3.0]])

y = layer(x)
print(y.shape)            # torch.Size([1, 2])
print(layer.weight.shape) # torch.Size([2, 3])
print(layer.bias.shape)   # torch.Size([2])
```

So `nn.Linear` is just a packaged version of the tensor math above.

## Activation

After a linear layer, you often apply an activation function. This adds
nonlinearity, which lets the network learn more than just one big linear
mapping.

Common idea:

```python
y = activation(x @ w.T + b)
```

Example with `ReLU`:

```python
import torch

x = torch.tensor([[-1.0, 2.0]])
relu_out = torch.relu(x)
print(relu_out)
```

Output:

```python
tensor([[0., 2.]])
```

`ReLU` keeps positive values and turns negative values into `0`.

Example with a linear layer:

```python
import torch.nn as nn

layer = nn.Linear(3, 2)
x = torch.tensor([[1.0, 2.0, 3.0]])

z = layer(x)
y = torch.relu(z)
```

So the usual pattern is:

- linear step: `z = x @ w.T + b`
- activation step: `y = relu(z)`

## nn.Softmax

`nn.Softmax(dim=...)` is the module version of `torch.softmax`. It turns values
into probabilities along a chosen axis.

```python
import torch
import torch.nn as nn

logits = torch.log(torch.tensor([[1.0, 1.0, 1.0],
                                 [1.0, 1.0, 2.0],
                                 [1.0, 3.0, 1.0],
                                 [2.0, 2.0, 1.0]]))

softmax = nn.Softmax(dim=-1)

y = softmax(logits)
print(y)
```

Output:

```python
tensor([[0.3333, 0.3333, 0.3333],
        [0.2500, 0.2500, 0.5000],
        [0.2000, 0.6000, 0.2000],
        [0.4000, 0.4000, 0.2000]])
```

This example has:

- `4` inputs -> 4 rows
- `3` classes -> 3 columns

Why this is easy to compute manually:

- `softmax` applies `exp(...)`
- here the inputs are logs, so `exp(log(k)) = k`
- each row becomes simple normalization

So the rows are:

- `[1, 1, 1] / 3 -> [1/3, 1/3, 1/3]`
- `[1, 1, 2] / 4 -> [1/4, 1/4, 1/2]`
- `[1, 3, 1] / 5 -> [1/5, 3/5, 1/5]`
- `[2, 2, 1] / 5 -> [2/5, 2/5, 1/5]`

Why `dim=-1`:

- `-1` means the last axis
- for shape `[4, 3]`, the last axis is the class axis
- so each row becomes a probability distribution over the 3 classes

For a 2D tensor shaped `[batch, classes]`, `dim=-1` and `dim=1` mean the same
thing.

## nn.Embedding

`nn.Embedding(num_embeddings, embedding_dim)` maps integer ids to learned dense
vectors.

Example:

```python
import torch
import torch.nn as nn

embed = nn.Embedding(5, 3)

idx = torch.tensor([0, 2, 4])
out = embed(idx)

print(out.shape)
```

Output shape:

```python
torch.Size([3, 3])
```

Here:

- `5` means there are 5 possible ids: `0` to `4`
- `3` means each id gets a learned vector of length 3
- input ids `[0, 2, 4]` become 3 embedding vectors

You can think of an embedding as a learnable lookup table:

```python
vector = embedding_table[id]
```

## nn.Parameter

`nn.Parameter` is a tensor that PyTorch treats as a learnable model parameter
inside an `nn.Module`.

```python
import torch.nn as nn

p = nn.Parameter(torch.randn(3, 4))
print(p.requires_grad)  # True
```

Why use it:

- a plain tensor is just data
- an `nn.Parameter` is registered as part of the model
- optimizers will update registered parameters during training

So in practice:

- use plain tensors for ordinary values
- use `nn.Parameter` for weights and biases you want the model to learn
