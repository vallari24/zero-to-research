# Lecture 2: Bigram Model

A bigram model is a character model that only looks at the current character
and predicts the next character.

For a name like `emma`, we add a boundary token so the model can learn where a
name starts and ends:

```text
. e m m a .
```

The training signal is every neighboring pair:

```text
. -> e
e -> m
m -> m
m -> a
a -> .
```

Each pair says: when the current character is on the left, the next character
was the one on the right.

## What Gets Learned

The model learns a table of next-character probabilities.

For example, after seeing many names, the model might learn:

```text
after a: n is likely, l is possible, . is possible
after q: u is very likely
after .: e, a, m, j are common starting letters
```

This is not a neural net yet. It is a probability table learned from counts.
Training means counting how often each transition appears, then converting
those counts into probabilities.

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

`stoi` maps string to integer. `itos` maps integer back to string.

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

That is already a trained count-based model. It answers:

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

The heatmap is just this matrix drawn as an image. Each square is one possible
bigram. Darker squares mean larger counts.

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

As a heatmap, bigger numbers would be darker:

```text
          .    a    b
.         .    █    ▒
a         ▓    ░    ▓
b         ▒    █    .
```

The real model uses the same idea, just with 27 rows and 27 columns instead of
3 rows and 3 columns.

A row answers:

```text
Given this current character, what next characters did we see?
```

A column answers:

```text
How often did this character appear as the next character?
```

For generation, rows are the important part because we repeatedly ask: given
the current character, what should come next?

## Counts Are Not Probabilities

Counts tell us frequency, but sampling needs probabilities.

Suppose the row for current character `a` contains:

```text
a -> n: 3
a -> l: 1
a -> .: 1
```

The total count for that row is `5`, so the probabilities are:

```text
P(n after a) = 3 / 5
P(l after a) = 1 / 5
P(. after a) = 1 / 5
```

Now the row sums to `1`. That means it is a valid probability distribution.

Training the count-based bigram model is therefore:

```text
counts -> row totals -> row probabilities
```

## Maximum Likelihood Estimation

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

The probability of the whole word under the bigram model is the product of
those transition probabilities:

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

In this count-based model, the parameters are the probabilities in the table
`P`.

## Why Use Log Likelihood

There are two practical problems with raw likelihood:

- it is a product of many numbers between `0` and `1`
- products of many small numbers become extremely tiny

So we take logs.

This helps because:

```text
log(a * b * c) = log(a) + log(b) + log(c)
```

A huge product becomes a sum, which is much easier to work with.

Also, `log` is monotonic: if one probability is bigger than another, its log is
also bigger. So maximizing likelihood and maximizing log likelihood are
equivalent.

For probabilities between `0` and `1`:

- `log(1) = 0`
- `log(0.5)` is negative
- `log(0.01)` is very negative

So worse probabilities produce more negative log values.

If you look at the plot of `log(x)` from `0` to `1`, it keeps increasing, so it
is a monotonic transformation: bigger probabilities stay bigger after taking
the log.

The key intuition from that curve is:

- near `x = 1`, `log(x)` is close to `0`
- as `x` gets smaller, `log(x)` gets more negative
- as `x -> 0`, `log(x) -> -infinity`

Why this matters for probabilities:

- if the model assigns high probability to the true next character, the log is
  close to `0`
- if the model assigns low probability, the log becomes very negative

So `log` is a monotonic transformation that preserves ordering, while also
making tiny probabilities much easier to work with mathematically.

## Negative Log Likelihood

Training code is usually written as minimizing a loss, not maximizing a score.

So instead of maximizing log likelihood, we minimize negative log likelihood:

```text
NLL = -log likelihood
```

That flips the sign:

- high likelihood -> high log likelihood -> low negative log likelihood
- low likelihood -> very negative log likelihood -> high negative log likelihood

So lower NLL means a better model.

For one transition:

```text
NLL = -log(p)
```

Examples:

- if `p = 1`, then `NLL = -log(1) = 0`
- if `p = 0.5`, then `NLL ≈ 0.69`
- if `p = 0.01`, then `NLL ≈ 4.61`

This is why NLL is a useful quality number: good probabilities give small loss,
bad probabilities give large loss.

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

On the training set, the model should assign relatively high probability to the
observed transitions because that is exactly the data it was fit on. But for
model comparison, validation NLL matters more than training NLL because it
shows whether the model generalizes.

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

## Row Normalization in Torch

In Torch, the count matrix has shape `[27, 27]`.

To normalize each row, we need one sum per row:

```python
row_sums = P.sum(dim=1, keepdim=True)
```

Why `dim=1`?

For a matrix shaped `[rows, columns]`, `dim=1` means operate across columns.
That gives one total for each row.

This is the wording that can be confusing: we are doing row normalization, but
Torch uses `dim=1` because it must collapse the columns to get each row total.

For a small matrix:

```text
[[1, 2, 3],
 [4, 5, 6]]
```

`sum(dim=1)` gives:

```text
[6, 15]
```

Those are row sums.

Then normalization is:

```python
P = P / row_sums
```

This divides every count in a row by that row's total. After that, each row of
`P` is a probability distribution.

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

So every row is divided by its own sum.

If you accidentally use shape `[27]`, Torch aligns that vector with the last
dimension. In a square `[27, 27]` matrix, the code may still run, but the
denominators line up like column values instead of row values. The result is
not the row-normalized probability matrix we wanted.

Small example:

```text
P =
[[1, 2, 3],
 [4, 5, 6]]
```

The row sums are:

```text
[6, 15]
```

For correct row normalization, we want this shape:

```text
[[6],
 [15]]
```

Then broadcasting divides each row by its own sum:

```text
[[1/6,  2/6,  3/6],
 [4/15, 5/15, 6/15]]
```

But if the denominator is shaped like a flat vector:

```text
[6, 15]
```

Torch tries to align it with the last dimension, like column values. For a
compatible square example:

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

## Sampling From the Model

After normalization, `P` is the model.

Each row of `P` is a probability distribution over the next character.

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

In Torch, the sampling operation is:

```python
ix = torch.multinomial(p, num_samples=1, replacement=True).item()
```

Here `p` is one row of `P`: a probability distribution over the 27 possible
next characters.

The generated name is a walk through the probability table:

```text
. -> j -> u -> l -> i -> a -> .
```

The model does not remember the full prefix. It only remembers the current
character. That is the limitation of a bigram model.

## What Was Built

Conceptually, the final model is a small Markov chain over characters.

It contains:

- a vocabulary of 27 character tokens
- a count table of shape `[27, 27]`
- a probability table of shape `[27, 27]`
- one probability row for each possible current character
- a sampler that walks row by row until it reaches `.`

The important idea is not the code. The important idea is this:

```text
data -> counts -> probabilities -> samples
```

A bigram model learns local character statistics. It can produce name-like
strings because many names share local patterns like `an`, `el`, `ia`, and
`mar`. But it cannot understand longer structure because it only conditions on
one character at a time.
