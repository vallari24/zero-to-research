# Neural Network and Backprop

A neuron is just:

\[
o = f\left(\sum_i w_i x_i + b\right)
\]

Intuition:
- inputs `x_i` carry information
- weights `w_i` decide how strongly each input matters
- bias `b` shifts the total
- activation `f` turns the total into an output

In `01_scalar_autodiff_engine.ipynb`, the code builds this idea from tiny
pieces. First it computes intermediate values, then it sends gradients backward
through the graph.

Backprop is the reverse story:
- forward pass computes the output
- backward pass tells each weight and input how much it affected that output
- each node passes gradient backward using the chain rule

That is why learning works: once you have those gradients, you can nudge the
parameters a little and get a better output on the next forward pass.
