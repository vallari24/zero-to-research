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

## nn.LayerNorm

`nn.LayerNorm` normalizes values across the last dimension for each example. It
is often used to keep activations in a stable range.

```python
import torch
import torch.nn as nn

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

layer_norm = nn.LayerNorm(3)
y = layer_norm(x)
print(y.shape)
```

Here:

- `x.shape` is `[2, 3]`
- each row has 3 numbers
- `nn.LayerNorm(3)` means: normalize groups of 3 numbers at a time

So `layer_norm(x)` normalizes each row separately:

- `[1, 2, 3]` becomes roughly `[-1.22, 0.00, 1.22]`
- `[4, 5, 6]` becomes roughly `[-1.22, 0.00, 1.22]`

`y.shape` stays the same as `x.shape`, so:

```python
torch.Size([2, 3])
```

LayerNorm changes the values, not the shape.

## nn.Dropout

`nn.Dropout(p)` randomly zeros some values during training. It is used as a
regularizer so the model does not rely too much on any single feature.

```python
import torch
import torch.nn as nn

x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
dropout = nn.Dropout(p=0.5)

y = dropout(x)  # training mode
print(y)
```

Possible output:

```python
tensor([[0., 2., 0., 2.]])
```

With `p=0.5`:

- each value has a 50% chance of becoming `0`
- kept values are scaled up

During evaluation, dropout is turned off.

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

Important terms:

- `token`: one discrete item, such as a character or a word
- `vocab`: the full set of allowed tokens
- `vocab_size`: how many tokens are in the vocab
- `embedding_dim`: how many numbers each token vector contains

Character example:

```python
import torch
import torch.nn as nn

stoi = {"a": 0, "b": 1, "c": 2}

vocab_size = 3
embedding_dim = 2

embedding_layer = nn.Embedding(vocab_size, embedding_dim)

ids = torch.tensor([[0, 1, 2, 1]])  # "abcb"
out = embedding_layer(ids)

print(out.shape)
```

Output shape:

```python
torch.Size([1, 4, 2])
```

Here:

- tokens are the characters `a`, `b`, `c`
- `vocab_size = 3` because there are 3 possible character tokens
- `embedding_dim = 2` because each character gets 2 learned numbers
- input shape `[1, 4]` means 1 sequence with 4 character ids
- output shape `[1, 4, 2]` means each character id became a 2D vector

Example output shape:

```python
[
  [
    [0.2, -0.5],   # token 0 -> "a"
    [1.1,  0.3],   # token 1 -> "b"
    [-0.7, 2.0],   # token 2 -> "c"
    [1.1,  0.3]    # token 1 -> "b" again
  ]
]
```

The outer `1` is the batch, the `4` is the sequence length, and the final `2`
is the size of each embedding vector.

You can think of the embedding as a trainable table of shape `[3, 2]`:

```python
row 0 -> vector for "a"
row 1 -> vector for "b"
row 2 -> vector for "c"
```

So:

- id `0` picks the row for `"a"`
- id `1` picks the row for `"b"`
- id `2` picks the row for `"c"`

`embedding_layer(ids)` means: look up the row for each id and return those
vectors in the same sequence order.

For `"abcb"`:

```python
[0, 1, 2, 1]
```

becomes:

```python
[vec("a"), vec("b"), vec("c"), vec("b")]
```

Word example:

```python
stoi = {"i": 0, "love": 1, "pytorch": 2}

embedding_layer = nn.Embedding(3, 4)

sentence_ids = torch.tensor([[0, 1, 2]])
word_vectors = embedding_layer(sentence_ids)

print(word_vectors.shape)
```

Output shape:

```python
torch.Size([1, 3, 4])
```

Here:

- the token is now a word
- `vocab_size = 3` because there are 3 word tokens in this tiny vocab
- `embedding_dim = 4` because each word gets a vector of length 4
- `embedding_layer(sentence_ids)` returns one 4-number vector per word

In larger language models, a token is often a subword piece instead of a full
word, but the lookup idea is the same.

One-hot comparison:

- one-hot for `"b"` in vocab `["a", "b", "c"]` is `[0, 1, 0]`
- that vector is sparse and mostly zeros
- an embedding replaces it with a learned dense vector such as `[1.1, 0.3]`

Conceptually:

```python
one_hot(token_id) @ W
```

and

```python
embedding_layer(token_id)
```

do the same lookup. `nn.Embedding` is just the efficient learned-table version.

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

## Manual Update vs nn.Module

From scratch, you might create loose tensors and update them yourself:

```python
W = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

loss.backward()

with torch.no_grad():
    W -= learning_rate * W.grad
    b -= learning_rate * b.grad

W.grad.zero_()
b.grad.zero_()
```

This works, but it does not scale. If you have many layers, you do not want to
manually update and zero every tensor one by one.

The cleaner approach is:

- `nn.Module` organizes the model and its parameters
- `torch.optim` updates all registered parameters for you

## nn.Module

`nn.Module` is the standard PyTorch container for a model.

Think of it as:

- `__init__`: define the layers
- `forward`: define how data flows through those layers

Example:

```python
import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear_layer(x)
```

Build it:

```python
model = LinearRegressionModel(in_features=1, out_features=1)
print(model)
```

Output:

```python
LinearRegressionModel(
  (linear_layer): Linear(in_features=1, out_features=1, bias=True)
)
```

Why this is better than loose tensors:

- the weights and biases are stored inside the model
- `model.parameters()` can find them automatically
- you no longer manage each parameter by hand

## torch.optim

`torch.optim` contains optimizers such as SGD and Adam. An optimizer takes
`model.parameters()` and knows how to update all of them.

Example:

```python
import torch.optim as optim

learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
```

Here:

- `model.parameters()` gives the optimizer every learnable parameter
- `optimizer` handles the weight updates
- `loss_fn` computes the training loss

## The Standard Training Step

With `nn.Module` and `torch.optim`, the core loop becomes:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

What each line means:

- `optimizer.zero_grad()`: clear old gradients from the previous step
- `loss.backward()`: compute new gradients for all model parameters
- `optimizer.step()`: update all model parameters using those gradients

This replaces the manual version:

```python
with torch.no_grad():
    W -= learning_rate * W.grad
    b -= learning_rate * b.grad

W.grad.zero_()
b.grad.zero_()
```

Same idea, better packaging.

## Easy Memory Hook

Use this picture:

- `nn.Module` = the organized model blueprint
- `nn.Parameter` = the learnable tensors inside it
- `torch.optim` = the tool that updates those parameters

So the full story is:

1. Put layers inside an `nn.Module`.
2. PyTorch registers their parameters.
3. Pass `model.parameters()` to an optimizer.
4. Run `zero_grad()`, `backward()`, `step()`.
