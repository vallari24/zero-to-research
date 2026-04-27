# Lecture 2: Bigram Model

A bigram model is a character model that looks at the current character and
predicts the next one.

For a name like `emma`, we add a boundary token so the model can learn where a
name starts and ends:

```text
. e m m a .
```

That creates the training pairs:

```text
. -> e
e -> m
m -> m
m -> a
a -> .
```

Each pair says: when the character on the left appeared, the character on the
right came next.

## What Gets Learned

The model learns a table of next-character probabilities.

After seeing many names, it may learn patterns like:

```text
after q: u is very likely
after .: a, e, j, m are common starts
after a: n is common, . is possible
```

At first, this is not a neural net. It is just a probability table learned
from counts.

## Tokens and Vocabulary

A token is one item the model can read or predict. In this model, each token is
a character.

The vocabulary is:

```text
.abcdefghijklmnopqrstuvwxyz
```

That gives `27` tokens:

- 26 letters
- 1 boundary token `.`

The boundary token matters because without it the model would not know how to
start or stop a generated name.

Because tensors use numbers, each character gets an integer id:

```text
. -> 0
a -> 1
b -> 2
...
```

`stoi` means string-to-integer. `itos` means integer-to-string.

## The Training Dataset

The training dataset is a long list of input-target pairs:

```text
input  = current character
target = next character
```

For `.emma.`, the pairs are:

```text
. -> e
e -> m
m -> m
m -> a
a -> .
```

This is the basic NLP habit to build:

1. Decide what one example is.
2. Decide what the target should be.
3. Decide how to represent the example as numbers.

For the bigram model:

- one example = one current character
- target = the next character

## Counting Transitions

The simplest way to train a bigram model is to count adjacent character pairs.

Conceptually, for each name:

1. Add a boundary token to the beginning and end.
2. Walk through the name one character at a time.
3. Count every current-character to next-character transition.

For `emma`, the transition `m -> a` appears once. Across the whole dataset,
that same transition may appear many times. The count tells us how common it
is.

A dictionary can store this directly:

```text
("m", "a") -> 2590
("a", "n") -> 5438
(".", "e") -> 1531
```

That already gives a count-based bigram model. It answers:

```text
How often did this next character follow this current character?
```

## The 27 x 27 Count Matrix

A dictionary is easy to understand, but a matrix is better for computation.

The count matrix `N` has shape `[27, 27]`:

```text
rows    = current character
columns = next character
```

So this cell:

```text
N[stoi["a"], stoi["n"]]
```

means:

```text
How many times did n follow a?
```

Tiny example with only three tokens, `.`, `a`, and `b`:

```text
          next char
          .    a    b
current
.         0    5    2
a         3    1    4
b         2    6    0
```

Read one cell like this:

```text
row a, column b = 4
```

That means `a -> b` appeared 4 times.

The heatmap is just this matrix drawn as an image. Each square is one possible
bigram. Darker squares mean larger counts.

As a tiny heatmap:

```text
          .    a    b
.         .    █    ▒
a         ▓    ░    ▓
b         ▒    █    .
```

The full model uses the same idea, just with 27 rows and 27 columns instead of
3 rows and 3 columns.

A row answers:

```text
Given this current character, what next characters did we see?
```

That is the important direction for generation.

## From Counts to Probabilities

Counts tell us frequency, but sampling needs probabilities.

Suppose the row for current character `a` contains:

```text
a -> n: 3
a -> l: 1
a -> .: 1
```

The row total is `5`, so the probabilities are:

```text
P(n after a) = 3 / 5
P(l after a) = 1 / 5
P(. after a) = 1 / 5
```

Now the row sums to `1`, so it is a valid probability distribution.

This is the count-based bigram model:

```text
counts -> row totals -> row probabilities
```

## Row Normalization

Conceptually, row normalization means:

- take one row of counts
- divide each entry by the total of that row

That turns a row of frequencies into a row of probabilities.

In PyTorch, if `P` holds the counts, row normalization means:

```python
row_sums = P.sum(dim=1, keepdim=True)
P = P / row_sums
```

Why `dim=1`?

For a matrix shaped `[rows, columns]`, `dim=1` means sum across the columns.
That gives one total for each row.

Small example:

```text
[[1, 2, 3],
 [4, 5, 6]]
```

Then:

```text
sum(dim=1) -> [6, 15]
```

Those are row sums.

After normalization:

```text
[[1/6,  2/6,  3/6],
 [4/15, 5/15, 6/15]]
```

Each row now sums to `1`.

## Why keepdim=True Matters

`keepdim=True` keeps the reduced dimension as size `1`.

Without it:

```text
P.sum(dim=1).shape -> [27]
```

With it:

```text
P.sum(dim=1, keepdim=True).shape -> [27, 1]
```

That `[27, 1]` shape means:

```text
one row sum for each of the 27 rows
```

This shape broadcasts correctly against `[27, 27]`.

Conceptually, the row sums look like:

```text
[[s0],
 [s1],
 [s2]]
```

Broadcasting makes them behave like:

```text
[[s0, s0, s0, ...],
 [s1, s1, s1, ...],
 [s2, s2, s2, ...]]
```

So each row is divided by its own sum.

If you accidentally use shape `[27]`, PyTorch aligns that vector with the last
dimension. In a square `[27, 27]` matrix, the code may still run, but the
denominators line up like column values instead of row values.

Small example:

```text
P =
[[1, 2],
 [3, 4]]

row_sums without keepdim = [3, 7]

P / row_sums =
[[1/3, 2/7],
 [3/3, 4/7]]
```

That is wrong for row normalization. The desired result was:

```text
[[1/3, 2/3],
 [3/7, 4/7]]
```

The sanity check is:

```python
P.sum(dim=1)
```

After correct normalization, every row sum should be close to `1`.

## Sampling From the Count-Based Model

After normalization, each row of `P` is a probability distribution over the
next character.

Generation starts at the boundary token:

```text
current = "."
```

Then repeat:

1. Look up the row for the current character.
2. Sample one next character from that row.
3. Append the sampled character to the output.
4. Move to the sampled character.
5. Stop if the sampled character is `.`.

In PyTorch, one way to sample is:

```python
ix = torch.multinomial(p, num_samples=1, replacement=True).item()
```

Here `p` is one row of `P`: a probability distribution over the 27 possible
next characters.

So generation is just a walk through the probability table:

```text
. -> j -> u -> l -> i -> a -> .
```

The model does not remember the full prefix. It only remembers the current
character. That is the limitation of a bigram model.

## Measuring Model Quality

Once we have a model, we need one number that tells us how well it explains the
data.

That is the role of likelihood.

For one training word, the model assigns a probability to every transition. For
`.emma.` the model would assign:

```text
P(e after .)
P(m after e)
P(m after m)
P(a after m)
P(. after a)
```

The probability of the whole word is the product of those transition
probabilities:

```text
P(.emma.) =
P(e after .)
* P(m after e)
* P(m after m)
* P(a after m)
* P(. after a)
```

For the whole dataset, likelihood means:

```text
multiply the probabilities of all observed transitions
```

Maximum likelihood estimation means:

```text
choose model parameters so the observed data gets the highest possible likelihood
```

For the count-based bigram model, the parameters are simply the probabilities
in the table `P`.

## Why Use Log Likelihood

Raw likelihood has two practical problems:

- it is a product of many numbers between `0` and `1`
- products of many small numbers become extremely tiny

So we take logs:

```text
log(a * b * c) = log(a) + log(b) + log(c)
```

A huge product becomes a sum, which is much easier to work with.

`log` is also monotonic: if one probability is bigger than another, its log is
also bigger. So maximizing likelihood and maximizing log likelihood are
equivalent.

For probabilities between `0` and `1`:

- `log(1) = 0`
- `log(0.5)` is negative
- `log(0.01)` is very negative

If you look at the plot of `log(x)` from `0` to `1`, it keeps increasing, so
it is a monotonic transformation: bigger probabilities stay bigger after taking
the log.

The important intuition is:

- near `x = 1`, `log(x)` is close to `0`
- as `x` gets smaller, `log(x)` gets more negative
- as `x -> 0`, `log(x) -> -infinity`

So assigning tiny probability to the true next character gets punished very
strongly.

## Negative Log Likelihood

Training code is usually written as minimizing a loss, not maximizing a score.

So instead of maximizing log likelihood, we minimize negative log likelihood:

```text
NLL = -log likelihood
```

That flips the sign:

- high likelihood -> low NLL
- low likelihood -> high NLL

So lower NLL means a better model.

For one transition:

```text
NLL = -log(p)
```

Examples:

- if `p = 1`, then `NLL = 0`
- if `p = 0.5`, then `NLL ≈ 0.69`
- if `p = 0.01`, then `NLL ≈ 4.61`

This is why NLL is a useful quality number: good probabilities give small
loss, bad probabilities give large loss.

## Average Negative Log Likelihood

For a dataset, we usually average the loss across all transitions:

```text
average NLL = mean of -log(probability of each observed next character)
```

This gives one scalar quality metric for the whole model.

Why average:

- datasets can have different numbers of transitions
- averaging makes scores easier to compare across runs

For a completely uniform model over 27 characters:

```text
p = 1 / 27 ≈ 0.037
average NLL = -log(1/27) = log(27) ≈ 3.30
```

So `3.30` is the rough baseline for a uniform 27-way random guesser. Better
models should get lower average NLL than that.

## Why Smoothing Matters

There is one more problem with raw counts: some bigrams may never appear in the
training set.

That would give:

```text
count = 0
probability = 0
log(0) = -infinity
NLL = infinity
```

That is too brittle. A single unseen transition would make the loss blow up.

So we smooth the counts before normalizing, for example with add-one smoothing:

```python
P = (N + 1).float()
```

This means:

- every bigram gets at least a count of `1`
- no probability is exactly `0`
- log likelihood and NLL stay finite

Smoothing does not mean the bigram is common. It just means the model does not
treat unseen transitions as impossible with absolute certainty.

## Transition to a Neural Bigram Model

The neural version uses the same dataset:

```text
current character -> next character
```

What changes is not the task. What changes is how we store and learn the
transition table.

Instead of storing counts directly, we learn a matrix of scores, then turn
those scores into probabilities.

That is why the full model uses a `[27, 27]` weight matrix:

- `27` possible current characters
- `27` possible next characters

Rows correspond to current characters. Columns correspond to next characters.

## Why One-Hot Encoding Appears

A neural net works with numbers, but raw integer ids like:

```text
. -> 0
e -> 5
m -> 13
a -> 1
```

are only labels. They do not mean one character is numerically "more" than
another in a useful way.

So we turn the current character into a one-hot vector.

Now the representation has one feature per possible token. That is the key
bridge from NLP objects like characters into the numeric world that a neural
network can operate on.

For the full character model:

- vocabulary size = `27`
- one-hot vector length = `27`
- input features = `27`

## The Core Intuition: One-Hot + Matrix Multiply

Suppose we represent a batch of current characters with a tensor `xenc`.

Its shape is:

```text
[batch, 27]
```

because:

- each row is one training example
- each example is one current character
- each character is represented by a one-hot vector of length `27`

Now let the weight matrix `W` have shape:

```text
[27, 27]
```

Then:

```text
xenc @ W
```

has shape:

```text
[batch, 27]
```

So each example gets `27` output scores.

That is exactly what we want: one score for each possible next character.

The most useful intuition is:

```text
one-hot + matrix multiply = row lookup
```

Why? Because a one-hot vector has only one active position.

Take the first pair in `emma`:

```text
. -> e
```

The current character is `.`. Its one-hot input vector has a `1` in the `.`
position and `0` everywhere else.

When we multiply that by `W`, we pick out the row of scores associated with
the current character `.`.

For the second pair:

```text
e -> m
```

the one-hot input picks out the row associated with `e`.

For the third pair:

```text
m -> m
```

the one-hot input picks out the row associated with `m`.

So the model is effectively doing:

```text
current character = current row selector
-> look at the matching row of W
-> use those values as scores for all possible next characters
```

This is the cleanest way to think about the neural bigram model: the count
table and the neural weight matrix are two versions of the same idea.

## Connecting It to y = w1x1 + w2x2 + w3x3

If you already understand:

```text
y = w1x1 + w2x2 + w3x3
```

then you already understand the core operation. That is one neuron:

- `3` input features
- `1` output

If instead you wanted `3` different outputs, you would need `3` different sets
of weights:

```text
y1 = w11x1 + w21x2 + w31x3
y2 = w12x1 + w22x2 + w32x3
y3 = w13x1 + w23x2 + w33x3
```

That can be written as one matrix multiply. The weight matrix would have shape
`[3, 3]`. So the bigram neural net is not a different kind of math. It is the
same weighted-sum idea applied to more inputs and more outputs.

The bigram neural net is the same pattern at a larger size:

- input features = `27`
- output features = `27`

So the weight matrix has shape:

```text
[27, 27]
```

## A Reusable Shape Recipe

When shapes feel confusing, use this checklist:

1. What is one input example?
2. How is that example represented as numbers?
3. How many output numbers do I want?

Then the tensor shapes follow:

- input: `[batch, in_features]`
- weights: `[in_features, out_features]`
- bias: `[out_features]`
- output: `[batch, out_features]`

For the bigram model:

- one example = one current character
- representation = one-hot vector of size `27`
- desired output = scores for `27` possible next characters

So:

- input = `[batch, 27]`
- weights = `[27, 27]`
- bias = `[27]`
- output = `[batch, 27]`

## From Scores to Probabilities

The row from `W` is not a probability distribution yet. It is just a vector of
raw scores, often called logits.

To turn those scores into probabilities:

1. exponentiate them so they become positive
2. divide by the row sum so the row adds up to `1`

Conceptually:

```text
scores -> positive values -> normalized probabilities
```

So the neural model still produces the same kind of object as the count-based
model: a probability distribution over the next character.

The exponentiation step matters because logits can be positive or negative, but
probabilities cannot. `exp` turns every score into a positive number. Larger
logits become larger positive values. Smaller logits become smaller positive
values. After row normalization, those positive values become probabilities.

In many explanations, the last two steps together are called softmax:

```text
logits -> exp -> normalize
```

## Putting the Neural Model Together

Take the word `emma` again. Its bigram training pairs are:

```text
. -> e
e -> m
m -> m
m -> a
a -> .
```

Using the character ids from the vocabulary, that becomes:

```text
xs = [0, 5, 13, 13, 1]
ys = [5, 13, 13, 1, 0]
```

Read this as:

- `xs` = current character ids
- `ys` = correct next-character ids

So the matrix `W` can be thought of like this:

```text
one row per possible current character
one column per possible next character
```

Each row holds all the next-character scores for one current character.

The model pipeline is:

```text
xs
-> one-hot encoding
-> xenc
-> xenc @ W
-> logits
-> exp
-> positive values
-> row normalization
-> probs
```

If there are `5` training pairs in this small example, then:

- `xenc` has shape `[5, 27]`
- `W` has shape `[27, 27]`
- `logits` has shape `[5, 27]`
- `probs` has shape `[5, 27]`

Each row of `probs` is the model's predicted probability distribution for one
training example. Each row should sum to `1`.

For `emma`, the five rows mean:

- row `0` = probabilities for what comes after `.`
- row `1` = probabilities for what comes after `e`
- row `2` = probabilities for what comes after the first `m`
- row `3` = probabilities for what comes after the second `m`
- row `4` = probabilities for what comes after `a`

More generally, let `i` be the row index. For each row `i`, there are two
different questions we can ask.

### 1. What does the model predict?

For row `i`, the model's prediction is:

```text
argmax(probs[i])
```

This means:

```text
which next character got the highest predicted probability in row i?
```

### 2. What probability did the model assign to the correct answer?

For `emma`, the correct next characters are:

```text
row 0 -> 5
row 1 -> 13
row 2 -> 13
row 3 -> 1
row 4 -> 0
```

In general, for row `i`, the probability used by the loss is:

```text
probs[i, ys[i]]
```

This means:

```text
for row i, take the probability assigned to the true next character
```

For `emma`, that becomes:

```text
probs[0, 5]
probs[1, 13]
probs[2, 13]
probs[3, 1]
probs[4, 0]
```

These are not the largest numbers in each row by definition. They are the
probabilities assigned to the true next characters for the five `emma`
training examples.

So the distinction is:

- `argmax(probs[i])` = the model's prediction for row `i`
- `probs[i, ys[i]]` = the probability the model assigned to the correct next character for row `i`

For example, when `i = 3`, row `3` corresponds to the pair:

```text
m -> a
```

So the correct column for the loss is `1`, not whichever column happens to be
largest in that row. If the model gives the biggest probability to some other
column, then the model predicted the wrong next character. The loss still uses
column `1`, because training cares about the true label, not just the model's
favorite guess.

### Reading One Output Entry

An expression like:

```text
(xenc @ W)[3, 13]
```

means:

- take the `4th` training pair from `emma`
- look at the score from the `14th` output neuron

That single value is one scalar logit: the score the model assigns to next
character `13` for row `i = 3`.

Mathematically, it is a dot product between:

- row `3` of `xenc`
- column `13` of `W`

Because the input row is one-hot, that dot product simplifies to a row lookup.
In the `emma` example, row `3` is the second `m`, so this value means:

```text
when the current character is m, what score did the model give to next character 13?
```

## From Probabilities to Loss

Once we have the correct probabilities:

- take `log` to get log likelihood
- add a minus sign to get negative log likelihood

So each training pair produces one NLL value, and the average of those values
is the loss.

That gives one scalar number answering:

```text
How good were the model's predicted next-character probabilities on this dataset?
```

In vectorized form, the probabilities used by the loss are:

```text
probs[0, ys[0]], probs[1, ys[1]], ..., probs[n-1, ys[n-1]]
```

In PyTorch this is written as:

```python
probs[torch.arange(n), ys]
```

and the average loss is:

```python
-probs[torch.arange(n), ys].log().mean()
```

## Forward Pass, Backward Pass, and Update

Training a neural network repeats three steps.

### Forward pass

The forward pass means:

```text
take the current parameters
-> run the input through the model
-> compute the loss
```

For this model, the forward pass is:

```text
xs
-> one-hot
-> xenc @ W
-> logits
-> softmax
-> probs
-> NLL loss
```

For `emma`, the five training pairs are:

```text
. -> e
e -> m
m -> m
m -> a
a -> .
```

So one forward pass takes all five current-character inputs at once and produces
five rows of probabilities:

```text
row 0 = probabilities for what comes after .
row 1 = probabilities for what comes after e
row 2 = probabilities for what comes after m
row 3 = probabilities for what comes after m
row 4 = probabilities for what comes after a
```

Then the loss looks at the probabilities of the correct next characters:

```text
probs[0, 5]
probs[1, 13]
probs[2, 13]
probs[3, 1]
probs[4, 0]
```

If those five numbers are small, the loss is high. If those five numbers get
larger, the loss goes down.

### Backward pass

The backward pass means:

```text
compute how the loss changes when each parameter changes
```

For `emma`, suppose the model currently assigns low probability to the correct
next characters. Then the backward pass tells us which entries of `W` should
move so that:

- the `.` row gives more probability to `e`
- the `e` row gives more probability to `m`
- the `m` row gives more probability to `m` and `a` in the right contexts
- the `a` row gives more probability to `.`

When we call:

```python
loss.backward()
```

PyTorch traces all the operations that created `loss` and uses the chain rule
to fill in gradients such as:

```text
W.grad
```

Each entry of `W.grad` tells us:

```text
if this weight changes a little, how will the loss change?
```

That is why `backward()` feels magical. PyTorch has remembered the dependency
graph from the forward pass and can differentiate all the way back to the
parameters.

The key idea is not that `backward()` guesses a new answer. It measures
sensitivity:

```text
which weights most need to change to make the correct next-character
probabilities larger and the loss smaller?
```

### Why the loss is differentiable

The scalar loss in this model is:

```python
loss = -probs[torch.arange(n), ys].log().mean()
```

So:

- `logits = xenc @ W` are not the loss
- `counts = logits.exp()` are not the loss
- `probs = counts / counts.sum(1, keepdim=True)` are not the loss
- `loss` is the final average negative log likelihood computed from the
  probabilities assigned to the correct next characters

This works because each step from `W` to `loss` is differentiable:

- matrix multiply is differentiable
- `exp` is differentiable
- division is differentiable
- selecting `probs[torch.arange(n), ys]` is differentiable with respect to the
  selected probability values
- `log` is differentiable for positive inputs
- `mean` is differentiable

So the whole pipeline:

```text
W
-> logits
-> counts
-> probs
-> probs[torch.arange(n), ys]
-> log
-> mean
-> loss
```

is differentiable.

### Update step

Once gradients are in `W.grad`, we move the weights a small step in the
direction that lowers the loss:

```python
W.data += -lr * W.grad
```

This is gradient descent.

If one entry of `W.grad` is positive, increasing that weight would increase the
loss, so gradient descent moves the weight downward.

If one entry of `W.grad` is negative, increasing that weight would decrease the
loss, so gradient descent moves the weight upward.

Then we run another forward pass and check the new loss. If learning is working
properly, the loss should go down over time.

For `emma`, the intuition after one update step is:

- the row for `.` should become a little more favorable to `e`
- the row for `e` should become a little more favorable to `m`
- the row for `m` should become a little more favorable to `m` and `a`
- the row for `a` should become a little more favorable to `.`

So after the update, a new forward pass should ideally produce slightly higher
values for:

```text
probs[0, 5], probs[1, 13], probs[2, 13], probs[3, 1], probs[4, 0]
```

and therefore a slightly lower average negative log likelihood.

## Count Table vs Neural Net

At this stage, the count-based bigram model and the neural bigram model often
end up with very similar loss.

That is not surprising. In this lecture, both models are solving essentially
the same problem:

- look at one current character
- produce a distribution over the next character

The count-based version stores that directly as a probability table.

The neural version stores it as:

- one-hot input
- one linear layer
- softmax output

So the neural model here is still extremely simple. It is basically another
way to parameterize the same bigram transition table.

That is why the neural model does not suddenly become much better just because
we used gradients. The context is still only one character, and the model is
still only one layer deep.

The reason to care about the neural approach is not that it wins immediately in
this tiny setting. The reason is that it scales to richer models.

For example, a count table becomes awkward very quickly if we want more
context:

- bigram table: current 1 character -> next character
- trigram table: current 2 characters -> next character
- 10-character context: current 10 characters -> next character

The number of possible contexts grows explosively. A literal lookup table for
long contexts becomes huge, sparse, and hard to estimate well.

A neural net gives a more flexible path:

- replace one-hot inputs with embeddings
- look at multiple previous characters, not just one
- add hidden layers
- learn shared structure instead of memorizing each context separately

So in this lecture, the neural bigram model is important mainly as a bridge:

```text
same task
-> same basic loss
-> same next-character prediction goal
-> but now in a differentiable form that can be extended
```

That differentiable form is what makes later models possible, including
multi-character MLPs, recurrent models, and transformers.

## Two Notes from the Lecture

### 1. One-hot encoding is really a row lookup

Once the current character is one-hot encoded, multiplying by the weight matrix
does something very simple:

```text
xenc @ W
```

selects the row of `W` that corresponds to the current character.

This is exactly the same role that the count matrix played earlier:

- current character index
- look up the corresponding row
- get scores or probabilities for the next character

So the neural version is not doing something conceptually different here. It is
recreating the same lookup-table behavior in a differentiable form.

The difference is:

- in the count model, the table was filled by counting
- in the neural model, `W` starts random and gradients push it toward useful
  values

### 2. Smoothing in the count model matches regularization in the neural model

In the count model, we smoothed the probabilities by adding fake counts:

```text
larger fake counts -> smoother distribution
very large fake counts -> nearly uniform distribution
```

In the neural model, something similar happens if we encourage the weights in
`W` to stay near zero.

If all entries of `W` were exactly `0`, then:

- all logits would be `0`
- `exp(0) = 1`, so all rows would become all ones
- after normalization, every row would become a uniform distribution

So pushing `W` toward zero makes the neural model smoother, just like adding
fake counts makes the count-based model smoother.

This is why adding a regularization term such as:

```text
(W**2).mean()
```

to the loss plays a role similar to smoothing. It discourages very large
weights, keeps the model less peaky, and pushes it slightly toward more uniform
probabilities.

## What Was Built

There are two views of the same bigram idea.

The count-based view is:

```text
data -> counts -> probabilities -> samples
```

The neural view is:

```text
current character
-> numeric representation
-> score table
-> probabilities for next character
```

In both cases, the model only uses one character of context. That is why it
can learn local patterns like `an`, `el`, `ia`, and `mar`, but not deeper
structure over long prefixes.
