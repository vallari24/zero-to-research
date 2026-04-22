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
