# Building Makemore Part 2: MLP

This note starts with the motivation for moving past count-based character
models.

## Why We Need Something Beyond Count Tables

In the bigram model, we used one character of context:

```text
current character -> next character
```

That gives a count matrix and a probability matrix with shape:

```text
[27, 27]
```

That means:

- `27` possible current-character contexts
- `27` possible next characters

So each row represents one current character, and each row stores next-character
counts or probabilities.

For example:

```text
row "." -> probabilities for what starts a name
row "e" -> probabilities for what comes after e
row "m" -> probabilities for what comes after m
```

This is manageable.

But suppose we want more context.

### Two Characters of Context

If we move from a bigram model to a trigram model, one example becomes:

```text
current 2 characters -> next character
```

Now the model must represent all possible 2-character contexts:

```text
27 * 27 = 729
```

So we can think about the count/probability table in two equivalent ways:

```text
[27, 27, 27]
```

or flattened as:

```text
[729, 27]
```

The meaning is the same:

- `729` possible two-character contexts
- `27` possible next characters for each context

For example, one row might correspond to the context:

```text
e m
```

and that row would store probabilities for:

```text
what comes after "em"?
```

### Three Characters of Context

If we use three characters of context, then the number of possible contexts
becomes:

```text
27 * 27 * 27 = 19683
```

Now the count/probability table can be viewed as:

```text
[27, 27, 27, 27]
```

or flattened as:

```text
[19683, 27]
```

That is already much larger, and it is clear that this approach is not
scalable.

## Why Move to an MLP

This is the reason to move to a neural model.

Instead of storing a separate probability row for every possible context, we
will learn a function:

```text
context -> next-character probabilities
```

In Part 2, that function will be a small multilayer perceptron (MLP).

The MLP will let us:

- take more context into account
- avoid building an enormous explicit count table
- learn shared structure across similar contexts
- predict the next character from learned parameters instead of raw counts

So the transition is:

```text
count table lookup
```

to:

```text
embed the context
-> feed it through an MLP
-> predict the next character
```

That is the setup for the rest of Part 2.

## Intuition Behind the Modeling Approach

The basic intuition comes from Bengio et al. (2003), *A Neural Probabilistic
Language Model*.

Instead of treating each word as an isolated symbol, we will give each word a
learned vector.

So if the vocabulary has around `17000` words, and each word gets a
`30`-dimensional embedding, then we can think of the model as learning:

```text
17000 points in a 30-dimensional space
```

At the beginning, these vectors are random. During training, backprop updates
them, so the points move around in the space.

The hope is that words playing similar roles end up near each other. For
example:

- `dog` and `cat`
- `the` and `a`
- `room` and `bedroom`

If two words are nearby in embedding space, then the model can treat them as
similar in useful ways.

That matters because the model will often see a phrase at test time that never
appeared exactly in training.

For example, maybe the exact phrase:

```text
a dog was running in a room
```

never appeared in the training set.

But maybe the model did see related phrases such as:

```text
the cat is walking in the bedroom
```

If training has pushed:

- `dog` near `cat`
- `a` near `the`
- `room` near `bedroom`

then the model can transfer what it learned from one phrase to another similar
phrase.

This is the key advantage over a raw count table.

A count table only knows exact contexts it has seen before.

An embedding-based model can generalize across similar contexts, because the
neural network is learning a smooth function over these vectors rather than
memorizing every context literally.

Another way to say it is:

```text
by learning and manipulating the embedding space, the model can transfer
knowledge and make reasonable predictions for novel inputs it never saw exactly
during training
```

So the core picture is:

```text
word -> vector
similar words -> nearby vectors
nearby vectors -> similar predictions
```

That is the intuition for why embeddings plus an MLP can handle longer context
better than explicit count tables.

## Bengio Paper Setup

This Part 2 modeling idea follows the basic setup from:

Bengio et al. (2003), *A Neural Probabilistic Language Model*  
https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

The original paper uses word-level language modeling. So instead of a small
character vocabulary, imagine a vocabulary of about `17000` words.

The task is:

```text
given the previous 3 words, predict the next word
```

So if the previous words are:

```text
w1, w2, w3
```

the model tries to predict:

```text
the 4th word
```

## Simplified Diagram

Below is the same core idea as the paper's diagram, but in simpler form:

```text
word id w1      word id w2      word id w3
    |               |               |
    v               v               v
 lookup in C     lookup in C     lookup in C
    |               |               |
    v               v               v
 30-d vector     30-d vector     30-d vector
      \             |             /
       \            |            /
        +------ concatenate ------+
                   |
                   v
             90 input numbers
                   |
                   v
              hidden layer
                   |
                   v
                  tanh
                   |
                   v
      output layer: 17000 scores/logits
                   |
                   v
                 softmax
                   |
                   v
 probabilities over 17000 possible next words
```

## Reading the Diagram Step by Step

### 1. The inputs are word ids

Each of the three previous words is first just an integer id.

Since the vocabulary has `17000` words, each input is an integer between:

```text
0 and 16999
```

By themselves, these integers do not carry meaning. They are just word labels.

### 2. Each word id looks up a row in the embedding matrix `C`

`C` is a learned embedding table with shape:

```text
[17000, 30]
```

That means:

- `17000` rows: one row per word
- `30` columns: the 30 learned features of that word

So each input word id is converted into a `30`-dimensional vector:

```text
w1 -> C(w1)
w2 -> C(w2)
w3 -> C(w3)
```

This is just a table lookup:

```text
word id -> embedding vector
```

The same matrix `C` is shared across all words. We are not learning a separate
embedding table for each input position.

### 3. The three embeddings are joined together

Each word gives `30` numbers.

So three words give:

```text
3 * 30 = 90
```

That is why the input to the neural network is a vector of `90` numbers.

### 4. The 90 numbers go into a hidden layer

The hidden layer size is a design choice.

For example, it might have `100` neurons.

This layer is fully connected to the `90` input numbers. Then we apply a
`tanh` nonlinearity.

So at this point the model has turned:

```text
3 previous words
```

into:

```text
a learned hidden representation of the context
```

### 5. The output layer produces 17000 scores

Next, the hidden layer connects to the output layer.

Because the model must choose among `17000` possible next words, the output
layer has:

```text
17000 neurons
```

Each one produces one score, often called a logit.

So the model is effectively producing:

```text
one score for each possible next word in the vocabulary
```

This is also why most of the computation happens here: the output layer is
very large.

### 6. Softmax turns scores into probabilities

The `17000` scores are passed through softmax.

Softmax turns them into a probability distribution over all `17000` candidate
next words:

```text
probability of word 0
probability of word 1
...
probability of word 16999
```

All of these probabilities add up to `1`.

So now the model can say:

```text
given w1, w2, w3,
how likely is each possible next word?
```

### 7. Training pushes up the probability of the correct next word

During training, we know the true next word.

So we look at the probability the model assigned to that correct word.

If the model assigned a low probability, the loss is high.

Backprop then updates:

- the embedding matrix `C`
- the hidden-layer weights
- the output-layer weights

so that the correct next word gets a higher probability next time.

## The Same Idea in Makemore

The Bengio paper is word-level with a vocabulary of about `17000` words.

`makemore` uses the same idea, but at the character level.

So here:

- vocabulary size = `27` characters
- context length = `3` characters
- embedding size in the toy example = `2`

That makes the tensors much smaller and easier to inspect by hand.

### 1. Building `X` and `Y`

The dataset is built as:

```text
3 previous characters -> next character
```

For the word `emma`, we pad on the left with dots and generate:

```text
... -> e
..e -> m
.em -> m
emm -> a
mma -> .
```

If:

```text
. = 0
a = 1
e = 5
m = 13
```

then these become:

```text
X row             Y
[0, 0, 0]   ->    5
[0, 0, 5]   ->    13
[0, 5, 13]  ->    13
[5, 13, 13] ->    1
[13, 13, 1] ->    0
```

So:

- `X` stores the input contexts
- `Y` stores the target next-character indices

If we build the dataset from the first 5 names and get `32` training examples,
then:

```text
X.shape = [32, 3]
Y.shape = [32]
```

This means:

- `32` total examples
- each input has `3` character ids
- each label has `1` character id

### 2. One-hot encoding vs lookup table

Suppose we want the embedding for character index `5`.

One way is direct lookup:

```python
C[5]
```

Another way is to one-hot encode `5` and multiply by `C`:

```python
F.one_hot(torch.tensor(5), num_classes=27).float() @ C
```

These are the same operation.

Why?

Because the one-hot vector has zeros everywhere except at position `5`, so the
matrix multiply simply selects row `5` of `C`.

So the useful mental model is:

```text
one-hot + matrix multiply = row lookup
```

In practice we use direct indexing because it is simpler and faster.

### 3. What the embedding table `C` looks like

If:

```python
C = torch.randn((27, 2))
```

then:

```text
C.shape = [27, 2]
```

That means:

- `27` rows: one row per character
- `2` columns: two embedding numbers per character

So every character id gets mapped to a 2-dimensional vector.

### 4. What `C[X]` does

PyTorch lets us index tensors in a very convenient way.

For a single character index:

```python
C[5]
```

returns one embedding vector:

```text
[-0.4713, 0.7868]
```

But PyTorch also lets us index a tensor with a list or another tensor.

So:

```python
C[[5, 6, 7]]
```

returns:

```text
[
  C[5],
  C[6],
  C[7]
]
```

stacked together as one tensor.

This is the important bridge to the full dataset.

If `X` is itself a tensor of indices, then:

```python
C[X]
```

means:

```text
take every integer inside X
and replace it with the corresponding row of C
```

If:

```text
X.shape = [32, 3]
```

then:

```python
emb = C[X]
```

replaces every integer in `X` by its 2D embedding vector from `C`.

So the shape becomes:

```text
emb.shape = [32, 3, 2]
```

Read this as:

```text
[number of examples, context length, embedding size]
```

So in this case:

- `32` examples
- `3` characters per example
- `2` numbers per character embedding

### 5. A concrete example of `[32, 3, 2]`

Take one input row:

```text
[0, 0, 5]
```

Suppose for illustration that:

```text
C[0] = [1.6, -0.2]
C[5] = [-0.5, 0.8]
```

Then this one input row becomes:

```text
[
  [ 1.6, -0.2 ],
  [ 1.6, -0.2 ],
  [ -0.5,  0.8 ]
]
```

This has shape:

```text
[3, 2]
```

because:

- 3 input characters
- each one becomes a vector of length 2

If we do this for all `32` rows in `X`, then stacking all of them gives:

```text
[32, 3, 2]
```

### 6. Why `[32, 3, 2]` becomes `[32, 6]`

The MLP wants one flat feature vector per example.

For one example, we currently have:

```text
[3, 2]
```

That means:

```text
3 characters * 2 numbers each = 6 total numbers
```

So the example:

```text
[
  [ 1.6, -0.2 ],
  [ 1.6, -0.2 ],
  [ -0.5,  0.8 ]
]
```

gets flattened into:

```text
[ 1.6, -0.2, 1.6, -0.2, -0.5, 0.8 ]
```

This is now a vector of length `6`.

So one row goes from:

```text
[3, 2]
```

to:

```text
[6]
```

and the full batch goes from:

```text
[32, 3, 2]
```

to:

```text
[32, 6]
```

That flattened `[32, 6]` tensor is what gets fed into the next linear layer.

### 7. Shape Summary

| tensor | meaning | shape |
| --- | --- | --- |
| `X` | input character-index contexts | `[32, 3]` |
| `Y` | target next-character indices | `[32]` |
| `C` | embedding lookup table | `[27, 2]` |
| `emb = C[X]` | embedded input contexts | `[32, 3, 2]` |
| `emb_flat` | flattened embeddings for the MLP | `[32, 6]` |

### 8. A Tiny Tensor Example for Indexing, `unbind`, and `view`

To make the tensor operations concrete, take a very small example:

```python
emb = torch.tensor([
    [[1, 10], [2, 20], [3, 30]],
    [[4, 40], [5, 50], [6, 60]],
])
```

Its shape is:

```text
[2, 3, 2]
```

Read this as:

- `2` examples
- `3` positions in the context
- `2` embedding numbers per position

So the dimensions mean:

- `dim 0` = which example
- `dim 1` = which character position
- `dim 2` = which embedding coordinate

#### What does `emb[:, 0, :]` mean?

```python
emb[:, 0, :]
```

means:

- `:` on `dim 0` -> take all examples
- `0` on `dim 1` -> take the first character position
- `:` on `dim 2` -> take both embedding numbers

Result:

```python
tensor([
    [1, 10],
    [4, 40],
])
```

Shape:

```text
[2, 2]
```

A few similar cases:

```python
emb[:, 1, :]
```

gives the second character position from every example:

```python
tensor([
    [2, 20],
    [5, 50],
])
```

```python
emb[0, :, :]
```

gives the first full example:

```python
tensor([
    [1, 10],
    [2, 20],
    [3, 30],
])
```

```python
emb[:, :, 0]
```

gives the first embedding coordinate from every position:

```python
tensor([
    [1, 2, 3],
    [4, 5, 6],
])
```

#### What does `torch.unbind` do?

`torch.unbind` splits a tensor along one chosen dimension.

```python
torch.unbind(emb, dim=0)
```

splits by example, so we get 2 tensors of shape `[3, 2]`.

```python
torch.unbind(emb, dim=1)
```

splits by character position, so we get 3 tensors:

```python
(
  tensor([[1, 10], [4, 40]]),
  tensor([[2, 20], [5, 50]]),
  tensor([[3, 30], [6, 60]])
)
```

Each has shape:

```text
[2, 2]
```

```python
torch.unbind(emb, dim=2)
```

splits by embedding coordinate, so we get:

```python
(
  tensor([[1, 2, 3],
          [4, 5, 6]]),
  tensor([[10, 20, 30],
          [40, 50, 60]])
)
```

Each has shape:

```text
[2, 3]
```

#### Why does `torch.cat(torch.unbind(emb, 1), 1)` flatten it?

First:

```python
torch.unbind(emb, 1)
```

returns the 3 character positions separately:

```python
(
  tensor([[1, 10], [4, 40]]),
  tensor([[2, 20], [5, 50]]),
  tensor([[3, 30], [6, 60]])
)
```

Then:

```python
torch.cat(torch.unbind(emb, 1), 1)
```

concatenates them along dimension `1`, producing:

```python
tensor([
    [1, 10, 2, 20, 3, 30],
    [4, 40, 5, 50, 6, 60],
])
```

Shape:

```text
[2, 6]
```

#### What does `emb.view(emb.shape[0], -1)` mean?

Since:

```text
emb.shape = [2, 3, 2]
```

we have:

```python
emb.shape[0] = 2
```

So:

```python
emb.view(emb.shape[0], -1)
```

means:

```python
emb.view(2, -1)
```

The `-1` tells PyTorch:

```text
infer this dimension for me
```

Because there are:

```text
2 * 3 * 2 = 12
```

numbers total, PyTorch infers:

```text
12 / 2 = 6
```

So the result is:

```python
emb.view(2, 6)
```

which gives:

```python
tensor([
    [1, 10, 2, 20, 3, 30],
    [4, 40, 5, 50, 6, 60],
])
```

So `view` is just another way to turn:

```text
[2, 3, 2]
```

into:

```text
[2, 6]
```

by flattening each example.

## Main Intuition

A very compact summary of the whole model is:

```text
3 word ids
-> 3 embedding lookups
-> concatenate into one vector
-> hidden layer + tanh
-> 17000 output scores
-> softmax
-> probabilities for the next word
```

The paper's diagram may look complicated at first, but the actual idea is
simple:

```text
turn words into vectors
combine the vectors
use a neural net to predict the next word
```

## Training the Model

For the small character-level `makemore` setup, one simple parameterization is:

- `C`: embedding table of shape `[27, 2]`
- `W1`: first-layer weights of shape `[6, 100]`
- `b1`: first-layer bias of shape `[100]`
- `W2`: second-layer weights of shape `[100, 27]`
- `b2`: second-layer bias of shape `[27]`

Here `6` comes from:

```text
3 context characters * 2 embedding numbers each = 6
```

So after:

```python
emb = C[X]
```

each example has shape:

```text
[3, 2]
```

and after flattening:

```text
[6]
```

If the batch has `32` examples, then:

```text
emb.view(-1, 6) -> [32, 6]
```

So `W1` must start with `6` rows, because the first linear layer is doing:

```text
[32, 6] @ [6, 100] = [32, 100]
```

That is why:

```python
W1 = torch.randn((6, 100), generator=g)
```

The `100` is the hidden-layer width: the number of hidden neurons we want.

### Forward pass

The forward pass is:

```python
emb = C[X]                        # [32, 3, 2]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)   # [32, 100]
logits = h @ W2 + b2             # [32, 27]
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
```

Why `emb.view(-1, 6)`?

- `emb` starts as `[32, 3, 2]`
- each example is `3 x 2 = 6` numbers
- `view(-1, 6)` flattens each example and lets PyTorch infer the batch size

So:

```text
[32, 3, 2] -> [32, 6]
```

Broadcasting also happens in:

```python
emb.view(-1, 6) @ W1 + b1
```

because:

- `emb.view(-1, 6) @ W1` has shape `[32, 100]`
- `b1` has shape `[100]`

PyTorch automatically adds the same `b1` row to every example in the batch.

### From logits to probabilities

`logits` are just raw scores for the `27` possible next characters.

After softmax-style normalization, `probs` becomes a probability distribution
for each row.

To see how much probability the model gave to the correct next character, we
index like this:

```python
probs[torch.arange(X.shape[0]), Y]
```

This means:

- take row `0`, column `Y[0]`
- take row `1`, column `Y[1]`
- ...

So we are extracting the probability assigned to the correct label for each
training example.

The manual negative log-likelihood loss is:

```python
loss = -probs[torch.arange(X.shape[0]), Y].log().mean()
```

### Backward pass and update

The trainable parameters are:

```python
parameters = [C, W1, b1, W2, b2]
```

Then the training step is:

```python
for p in parameters:
    p.grad = None

loss.backward()

for p in parameters:
    p.data += -lr * p.grad
```

This is standard gradient descent:

- forward pass computes the loss
- backward pass computes gradients
- update step moves parameters in the direction that lowers the loss

### Minibatching

Using the full dataset for every update is expensive.

So instead, we sample a minibatch:

```python
batchsize = 32
batchix = torch.randint(0, X_all.shape[0], (batchsize,))
bx, by = X_all[batchix], Y_all[batchix]
```

Then we run forward, backward, and update only on that subset.

This makes training much faster.

The gradient is noisier because it is estimated from only part of the data, but
in practice that is usually a very good tradeoff.

### Learning rate intuition

The learning rate controls step size.

- too small: training barely moves
- too large: training becomes unstable and loss can blow up

One practical way to search is to sweep values on a log scale:

```python
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
```

Then train briefly with each candidate and plot loss against learning rate.

The goal is not to find a magical exact number. The goal is to find a sensible
range where loss drops well and stays stable.

### Picking the learning rate from a plot

During minibatch training, the loss often fluctuates. That is normal, because
each update only sees a small random subset of the data.

But the fluctuations also raise a real question:

```text
are we stepping too slowly or too aggressively?
```

If the learning rate is too small, training barely moves.

If it is too large, the updates overshoot and training becomes unstable.

A practical way to build confidence is:

1. choose a range of candidate learning rates
2. space them exponentially
3. train briefly with each one
4. plot loss versus learning rate

For example:

```python
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
```

This sweeps learning rates from:

```text
0.001 up to 1
```

on a log scale.

Then for each candidate learning rate, run a short minibatch training loop,
record the loss, and plot:

```python
plt.plot(lrei, lossi)
```

The useful pattern is:

- at very small learning rates, loss improves too slowly
- as learning rate increases, loss drops faster
- after some point, loss starts rising again because the steps are too large

So the rule of thumb is:

```text
pick a learning rate near the lowest, most stable part of the curve,
before the loss starts trending upward
```

In this run, the selected learning rate was:

```python
lr = 10**lrei[lossi.index(min(lossi))]
```

which gave:

```text
lr = 0.2309
```

That does not mean `0.2309` is universally best. It only means that for this
model and this setup, it looked like a strong choice from the sweep.

Using that learning rate and training longer gave a final loss around:

```text
2.5242
```

### What the training-loss plot usually looks like

A typical minibatch training curve looks like this:

```text
loss
^
|\
| \
|  \__
|     \___ noisy plateau
+----------------------------> steps
```

At the beginning, the loss usually drops quickly because the model is learning
the most obvious structure first.

After that, the curve becomes noisy and flatter:

- noisy, because each step uses a different random minibatch
- flatter, because improvements become smaller once the easy gains are gone

So a plot like this is normal:

- a sharp early decrease
- then a long, jittery band of losses

The important question is not whether every step goes down. In minibatch
training it will not. The important question is whether the overall trend is
going down over time.

## Train, Validation, and Test Splits

A lower training loss is not enough to claim that we have a better model.

Why?

Because a model with enough capacity can simply memorize the training set.

In that case:

- training loss becomes very low
- but the model does not generalize well to new examples

This is the basic overfitting problem.

So the standard practice is to split the dataset into three parts:

- train: about `80%`
- validation (or dev): about `10%`
- test: about `10%`

Each split has a different role:

- the **training split** is used to fit the model parameters
- the **validation split** is used to compare model choices and tune hyperparameters
- the **test split** is used only at the end for a final evaluation

The key idea is:

```text
train on train
choose settings on validation
report final performance on test
```

This matters because every time we look at the test set and learn from it, we
start adapting our choices to that test set too. So the test split should be
used very sparingly.

### A typical split

For the names dataset:

```python
random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

xtrain, ytrain = build_dataset(words[:n1])
xval, yval = build_dataset(words[n1:n2])
xtest, ytest = build_dataset(words[n2:])
```

In the original notebook this gives roughly:

- `25626` words in train
- `3203` words in validation
- `3204` words in test

After converting words into `(context -> next character)` examples, the tensor
sizes become much larger because each word contributes multiple training rows.

### How to read train vs validation loss

Suppose after training we get:

```text
train loss = 2.08
validation loss = 2.43
```

The validation loss is higher, which is expected, because the model has already
seen the training data.

What matters is the size of the gap.

- if train loss is much lower than validation loss, the model may be overfitting
- if train and validation loss are both high and fairly close, the model may be underfitting

In the notebook, train and validation losses stay fairly close. That suggests
the model is not severely overfitting yet. In fact, the small network is closer
to underfitting: it still has room to improve by becoming a bit larger.

### Scaling the model

One simple way to increase model capacity is to widen the hidden layer.

For example:

```python
parameters = define_nn(l1out=300)
```

This gives the model more hidden units and therefore more capacity to model the
training data.

The usual workflow is:

1. train on the training split
2. evaluate on the validation split
3. change hyperparameters such as hidden width, embedding size, learning rate, or regularization
4. repeat until validation performance stops improving
5. run the final model on the test split once

## How Sampling Works

After training, we can use the model to generate new names one character at a
time.

The idea is:

```text
start with ...
-> predict next-character probabilities
-> sample one character
-> shift the context window
-> repeat until .
```

### Step 1. Start with an empty context

If `block_size = 3`, sampling starts with:

```python
context = [0, 0, 0]
```

Since `0` is the `.` token, this means the starting context is:

```text
...
```

### Step 2. Run one forward pass

For the current context, we do:

```python
emb = C[torch.tensor([context])]
h = torch.tanh(emb.view(1, -1) @ W1 + b1)
logits = h @ W2 + b2
probs = F.softmax(logits, dim=1)
```

This is the same forward pass as training, except now we only have one example,
so the batch size is `1`.

The result is:

```text
probabilities for the next character
```

### Step 3. Sample the next character

```python
ix = torch.multinomial(probs, num_samples=1, generator=g).item()
```

This does not always pick the single most likely character. Instead, it samples
according to the probability distribution.

That is important, because deterministic argmax decoding would produce much
less varied names.

### Step 4. Update the context

After sampling the next character index `ix`, we shift the context window:

```python
context = context[1:] + [ix]
```

So if:

```python
context = [0, 0, 0]
ix = 3
```

then the new context becomes:

```python
[0, 0, 3]
```

We also append `ix` to the output sequence we are building.

### Step 5. Stop at the end token

If the model samples:

```python
ix == 0
```

then it has predicted the `.` token, which means the name is finished.

So generation stops there.

### Tiny example

One possible generation might look like:

```text
... -> c
..c -> a
.ca -> r
car -> .
```

which gives:

```text
car
```

So sampling is just the training-time prediction rule used repeatedly in a
loop, feeding each new sampled character back into the context.
