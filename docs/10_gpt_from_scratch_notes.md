# Building GPT From Scratch: Pretraining A Tiny Transformer

This post is about teaching a small model to continue text.

## What To Expect

This blog follows the whole path from a text file to a small GPT-like model.

The full story is:

```text
1. Prepare the text
2. Train a first simple model
3. Validate it on text it did not train on
4. Add self-attention so the model can use earlier context
5. Build the Transformer by stacking attention blocks
6. Generate new text from the trained model
```

A GPT-like model is complicated, but the training idea is simple:
<span style="color:#8aff8a"><strong>use previous tokens to predict the next
token</strong></span>.

The first part prepares the data:

```text
text file
-> token ids
-> train/validation split
```

Then we train a tiny baseline model. It will not be impressive, but it gives us
the core training loop:

```text
input tokens
-> predict the next token
-> measure the mistake
-> update the model
```

Then we validate. Validation means we test the model on text it did not train
on. This tells us whether the model is learning patterns, not just memorizing.

After the training loop works, we make the model stronger. Self-attention lets
each token look back at earlier tokens and decide which ones matter.

The Transformer is the final shape: repeated attention blocks plus a few
supporting layers that make the model train well.

Memory hook:

```text
given previous tokens, predict the next token
```

## Tokenization

We give the model a text file, show it many examples of "what comes next?", and
train it to get better at guessing the next piece of text.

Skim version: before we build the model, we need to
<span style="color:#ffff99"><strong>turn text into numbers</strong></span>,
split the numbers into train and validation data, and decide how we will measure
whether the model is improving.

A computer cannot train directly on this:

```text
hello
```

It needs numbers.

Tokenization means <span style="color:#ffff99"><strong>choosing text pieces and
assigning each piece an integer id</strong></span>.

Tokenization is the step where we choose the pieces of text and give each piece
a number.

The pieces can be:

```text
characters: h, e, l, l, o
word chunks: hello
subword chunks: he, llo
```

For the first version, we use characters because they are easiest to see.

Suppose the whole text is:

```python
text = "hello"
```

First, collect every unique character:

```python
chars = sorted(list(set(text)))
print(chars)
```

This gives:

```text
['e', 'h', 'l', 'o']
```

Now give each character an id:

```python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print(stoi)
```

One possible mapping is:

```text
{
  'e': 0,
  'h': 1,
  'l': 2,
  'o': 3,
}
```

Now `hello` can become numbers:

```python
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)

ids = encode("hello")
print(ids)
print(decode(ids))
```

Output:

```text
[1, 0, 2, 2, 3]
hello
```

That is the whole concept:

```text
text -> ids -> text
```

Sanity check: if <span style="color:#8aff8a"><strong>`decode(encode(text))`
gives back the original text</strong></span>, the basic tokenizer loop is
working.

The model trains on the ids:

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

`encode(text)` makes a Python list of numbers. `torch.tensor(...)` turns that
list into the PyTorch format the model can use.

### Our Tokenizer vs tiktoken vs SentencePiece

All tokenizers do the same basic job:

```text
text <-> token ids
```

They differ in what they choose as a "piece" of text.

Our first tokenizer is character-level:

```text
hello -> h e l l o
```

This is simple and transparent, but it makes long sequences.

The GPT-2 tokenizer in `tiktoken` is already trained. It uses common chunks of
text, so it may represent `hello world` with far fewer tokens than a
character-level tokenizer.

For example:

```python
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("hello world")
print(ids)
print(enc.decode(ids))
```

Output:

```text
[31373, 995]
hello world
```

The GPT-2 tokenizer represents `hello world` with only two tokens. Our
character-level tokenizer represents the same text with eleven tokens:

```text
[55, 52, 59, 59, 62, 1, 70, 62, 65, 59, 51]
hello world
```

The key tradeoff is <span style="color:#93c5fd"><strong>codebook size versus
sequence length</strong></span>.

The codebook is the list of possible tokens. It is also called the vocabulary.

```text
small codebook:
few possible tokens
very long sequence of integers

large codebook:
many possible tokens
shorter sequence of integers
```

Our first tokenizer has a very small codebook: just the unique characters in the
text. In our setup, that is about `83` characters. This keeps the tokenizer easy
to understand, but it means the text becomes a very long sequence of integers.

The GPT-2 tokenizer uses a much larger codebook of subword pieces. Because
common chunks like `hello` or ` world` can become single tokens, the same text
can be represented with a shorter sequence of integers.

This matters because Transformers train on sequences of token ids. Longer
sequences mean:

```text
more positions to process
more memory use
more attention computation
less real text fits into the same context window
```

This is why GPT-like models typically use subword tokenization. It is a
practical middle ground:

```text
not as tiny as characters
not as huge and brittle as full words
usually much shorter than character sequences
```

```text
tiktoken GPT-2 = load existing tokenizer rules
```

SentencePiece is usually used to train or load tokenizer rules.

```text
SentencePiece library = tool
tokenizer.model = trained rules
```

So SentencePiece does not work just because the package is installed. It needs a
trained tokenizer model file, or you have to train one.

We start with character tokenization because it is
<span style="color:#8aff8a"><strong>intentionally simple</strong></span>. It
makes the data pipeline easy to inspect before we introduce more powerful
tokenizers.

## Train/Validation Split

If we train and test on the exact same text, we cannot tell whether the model is
learning or just memorizing.

Train data is for practice. Validation data checks whether the model learned
something useful <span style="color:#93c5fd"><strong>beyond the exact practice
text</strong></span>.

So we split the data into two parts:

```text
training data:
the part the model studies from

validation data:
the part we keep hidden while training, then use to check progress
```

In code:

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

This means:

```text
first 90%  -> train
last 10%   -> validate
```

An intuition:

```text
training data = practice problems
validation data = quiz problems the model has not seen
```

If the model improves on training data but not validation data, it may be
memorizing the practice problems instead of learning the pattern.

For this language model, the main number we watch is usually loss.

```text
lower training loss:
the model is getting better on the text it studies

lower validation loss:
the model is also getting better on held-out text
```

### Metrics For LLM Pretraining

The main metrics are <span style="color:#ffff99"><strong>cross-entropy
loss</strong></span> and <span style="color:#ffff99"><strong>perplexity</strong></span>.

For next-token prediction, the main metric is cross-entropy loss.

The model looks at a context and produces a score for every possible next token.
After softmax, those scores become probabilities.

For example, after seeing:

```text
hello
```

the model might assign probabilities like:

```text
space: 0.40
!:     0.10
s:     0.05
other tokens: ...
```

If the true next token is `space`, the model did well because it gave the
correct token high probability.

If the true next token is `space` but the model gave it probability `0.001`,
the model was very surprised. Cross-entropy loss measures that surprise.

```text
high probability on the correct next token -> low loss
low probability on the correct next token  -> high loss
```

So during pretraining, we usually track:

```text
training loss:
surprise on the text the model trains on

validation loss:
surprise on held-out text
```

Lower validation loss means the model is getting better at predicting new text
from the same kind of data.

Another common metric is perplexity:

```text
perplexity = exp(loss)
```

Perplexity is another way to read the same signal. Intuitively:

```text
perplexity ~= how many likely next-token choices the model feels confused among
```

Lower perplexity is better.

```text
perplexity 100:
the model is very unsure

perplexity 10:
the model has narrowed the next token down much more

perplexity 2:
the model is very confident among a small number of likely choices
```

Loss and perplexity are useful because they match the actual pretraining task:

```text
given context, predict the next token
```

### Precision And Recall Intuition

Precision and recall are useful for many machine learning problems, but they are
<span style="color:#ff8a8a"><strong>not the main scorecard for next-token
pretraining</strong></span>.

Imagine a model that marks emails as spam.

Precision asks:

```text
When the model says "spam", how often is it right?
```

High precision means few false alarms.

Recall asks:

```text
Of all the real spam emails, how many did the model catch?
```

High recall means it misses very little spam.

The tradeoff:

```text
high precision:
be careful before saying yes

high recall:
try hard to catch every yes
```

Validation data is where these kinds of numbers matter. We want to measure the
model on examples it did not train on, because that is closer to how it will
behave on new data.

## Feeding Data In Chunks And Batches

We do <span style="color:#ff8a8a"><strong>not</strong></span> feed the whole
dataset into the model at once. The dataset can be millions of characters long,
which will never fit. Instead we feed small pieces.

There are two ideas here, and they are easy to mix up:

```text
block_size = how long ONE chunk is   (the time dimension)
batch_size = how many chunks at once (the batch dimension)
```

### Chunks And Block Size

We cut the data into chunks. The maximum length of a chunk is the
<span style="color:#ffff99"><strong>block size</strong></span>.

For our setup the block size is `8`. A chunk that trains on a block of `8`
actually needs `9` characters: `8` inputs plus the one extra character that is
the target for the last input.

```python
block_size = 8
x = train_data[:block_size]      # the 8 inputs
y = train_data[1:block_size+1]   # the same window, shifted by one = targets
```

One chunk of `9` characters is not one example. It is packed with
<span style="color:#8aff8a"><strong>8 individual examples</strong></span>:

```python
for t in range(block_size):
    context = x[:t+1]
    label = y[t]
    print(f"when input is {context} the target: {label}")
```

```text
when input is tensor([1])                       the target: 41
when input is tensor([ 1, 41])                  the target: 63
when input is tensor([ 1, 41, 63])              the target: 62
when input is tensor([ 1, 41, 63, 62])          the target: 67
when input is tensor([ 1, 41, 63, 62, 67])      the target: 7
when input is tensor([ 1, 41, 63, 62, 67,  7])  the target: 1
when input is tensor([ 1, 41, 63, 62, 67,  7,  1])      the target: 41
when input is tensor([ 1, 41, 63, 62, 67,  7,  1, 41])  the target: 63
```

The context grows one character at a time:

```text
chunk = 9 characters, block_size = 8

 chars:  1   41   63   62   67   7   1   41 | 63
         └─ x: the 8 inputs ──────────────┘   └ last target

 1                              -> 41
 1 41                           -> 63
 1 41 63                        -> 62
 1 41 63 62                     -> 67
 1 41 63 62 67                  -> 7
 1 41 63 62 67 7                -> 1
 1 41 63 62 67 7 1              -> 41
 1 41 63 62 67 7 1 41           -> 63
```

This is on purpose. We want the Transformer to learn to predict with
<span style="color:#93c5fd"><strong>as little as 1 character of context, and as
much as `block_size` characters</strong></span>. By the end of the chunk it has
seen every context length from `1` up to `block_size`.

The Transformer will <span style="color:#ff8a8a"><strong>never</strong></span>
see more than `block_size` characters at a time as input. Anything longer has to
be truncated down to the last `block_size` characters.

```text
context shorter than block_size -> fine, the model has seen these
context longer  than block_size -> truncate to the last block_size chars
```

### Batches And Batch Size

One chunk at a time would leave the GPU mostly idle. A GPU is happiest when it
has many independent pieces of work to crunch at once.

So we stack several chunks side by side and process them together. The number of
chunks we process in parallel is the
<span style="color:#ffff99"><strong>batch size</strong></span>.

```python
batch_size = 4   # how many independent sequences we process in parallel
block_size = 8   # the maximum context length for predictions
```

The chunks are <span style="color:#8aff8a"><strong>independent</strong></span>:
they do not talk to each other, and we pull them from
<span style="color:#93c5fd"><strong>random locations</strong></span> in the
training data so each batch is a fresh mix.

```python
torch.manual_seed(1337)

ix = torch.randint(len(train_data) - block_size, (batch_size,))
x = torch.stack([train_data[i:i+block_size]   for i in ix])
y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
```

Reading it line by line:

```text
ix : batch_size random offsets, each between 0 and len(data) - block_size
x  : block_size characters starting at each offset, stacked as rows
y  : the same windows shifted right by one, stacked as rows
```

`ix` is just `4` random starting points:

```text
ix = tensor([636549, 429903, 270558, 12140])   # shape [4]
```

Picture each offset landing somewhere random in the data, then grabbing the next
`8` characters:

```text
train_data: ........[chunk]...............[chunk]......[chunk]...[chunk].....
                     ^636549              ^429903     ^270558   ^12140
```

`x` stacks those four chunks into rows. `y` is the same four chunks shifted by
one, so every position in `x` has its next-character answer in `y`:

```text
x  (inputs)  shape [4, 8]        y  (targets) shape [4, 8]
[[62 61  4 67  1 70 48 66]       [[61  4 67  1 70 48 66 55]
 [70  1 66 55 56 65 67  1]        [ 1 66 55 56 65 67  1 48]
 [ 1 67 55 48 67  1 66 62]        [67 55 48 67  1 66 62 60]
 [66  1 69 52 65 72  1 55]]       [ 1 69 52 65 72  1 55 48]]
```

Look at the top-left: in `x` the value `62` sits at position `(0,0)`, and the
target at `y[0,0]` is `61` — the very next character. That is true for
every cell.

### What We Actually Feed The Transformer

Put the two ideas together and the input is a single `[batch_size, block_size]`
block of integers:

```text
x : shape [4, 8]   the inputs
y : shape [4, 8]   the targets, one for every position in x
```

That little `4 x 8` block is doing a lot of work. Each row contributes
`block_size` examples (context lengths `1` to `8`), and there are `batch_size`
rows, so one batch holds:

```text
batch_size * block_size = 4 * 8 = 32 independent examples
```

We can spell all `32` out by walking both dimensions:

```python
for b in range(batch_size):   # batch dimension: which chunk
    for t in range(block_size):   # time dimension: how far into the chunk
        context = x[b, :t+1]
        target = y[b, t]
        print(f"when input is {context.tolist()} the target: {target}")
```

```text
when input is [62]                       the target: 61
when input is [62, 61]                   the target: 4
...
when input is [62, 61, 4, 67, 1, 70, 48, 66]   the target: 55   <- end of row 0
when input is [70]                       the target: 1          <- row 1 starts
...
when input is [66, 1, 69, 52, 65, 72, 1, 55]   the target: 48   <- end of row 3
```

`32` separate "given this context, predict the next character" problems, all
packed into one tidy tensor and solved in a single forward/backward pass.

Memory hook:

```text
block_size = how far back one example can look
batch_size = how many examples we crunch at once
one batch = batch_size x block_size training examples
```

## The Bigram Model: Our First Baseline

Now we build the simplest possible model that can predict the next character: a
<span style="color:#ffff99"><strong>bigram language model</strong></span>. It will
not be good, but it gives us the full loop — a model, a loss, and text
generation — that the Transformer will later slot into.

The idea is almost too simple: <span style="color:#93c5fd"><strong>each token
directly looks up the scores for the next token</strong></span>. If I am
character `5`, I go to row `5` of a table and read off "given that I am `5`, here
is how likely every character is to come next." It only ever looks at the single
previous character — hence *bigram*.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx)   # (B, T, C)
        return logits
```

`nn.Embedding(vocab_size, vocab_size)` is just a lookup table with one row per
character, and each row has `vocab_size` numbers in it. Passing `idx` in returns
the matching rows.

### Logits Are Scores For The Next Character

The output is called <span style="color:#ffff99"><strong>logits</strong></span> —
a raw score for every possible next character. For each position in the input,
the model emits one score per character in the vocabulary.

```python
m = BigramLanguageModel(vocab_size)
logits = m(xb, yb)
print(logits.shape)
```

```text
torch.Size([4, 8, 83])
```

That shape is `(B, T, C)`, and these three letters show up everywhere from here
on:

```text
B = batch   : how many independent sequences   (4)
T = time    : position in the sequence, up to block_size   (8)
C = channels: one score per vocabulary character   (vocab_size = 83)
```

So for every one of the `4 x 8 = 32` positions, the model produces `83` scores.
A good model puts a <span style="color:#8aff8a"><strong>high number on the
correct next character</strong></span> and low numbers on the rest.

### Measuring The Loss

To score how wrong we are, we use
<span style="color:#ffff99"><strong>cross-entropy</strong></span> loss (the same
"negative log likelihood" idea from the metrics section): the logit at the
target character should be large, everything else small.

The catch is the shape. PyTorch's `F.cross_entropy` does not want
`(B, T, C)` — it wants the channels in the **second** dimension, i.e.
`(N, C)` for the logits and `(N,)` for the targets. Calling it directly fails:

```python
loss = F.cross_entropy(logits, targets)   # error: wrong shape
```

**Why does cross-entropy not just accept `(B, T, C)`?** Because to it, a
"position" is one prediction problem: a row of `C` scores plus the one correct
answer. It does not care that our positions happen to be arranged in a `B x T`
grid (4 sequences, 8 characters each). It only wants a flat list of prediction
problems.

We have `B x T` positions, and each is an independent "predict the next
character" problem. So we flatten the batch and time grid into one long
list — every position becomes just another row — and flatten the targets the
same way so they still line up:

```python
def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)   # (B, T, C)

    if targets is None:
        loss = None
    else:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)   # (B*T, C) -> stack all positions
        targets = targets.view(B*T)    # (B*T,)
        loss = F.cross_entropy(logits, targets)

    return logits, loss
```

Picture it with our real numbers (`B = 4`, `T = 8`, `C = 83`). The `view`
flattens the first two axes into one, leaving the row of `83` scores untouched:

```text
logits BEFORE: (B, T, C) = (4, 8, 83)        a 4x8 GRID of positions,
                                             each position = 83 scores

      seq 0:  [pos0][pos1]...[pos7]          each [posX] is a length-83 row
      seq 1:  [pos0][pos1]...[pos7]
      seq 2:  [pos0][pos1]...[pos7]
      seq 3:  [pos0][pos1]...[pos7]

                    | logits.view(B*T, C)
                    | stack every row into one tall list
                    v

logits AFTER:  (B*T, C) = (32, 83)           a flat LIST of 32 positions,
                                             each still 83 scores
      row 0:  [83 scores]   <- was seq0/pos0
      row 1:  [83 scores]   <- was seq0/pos1
      ...
      row 31: [83 scores]   <- was seq3/pos7
```

The targets ride along the same way — one correct character id per position, so
`(B, T)` becomes a flat `(B*T,)` in the exact same order:

```text
targets BEFORE: (B, T) = (4, 8)     [[c, c, ..., c],   <- 8 answers for seq 0
                                     [c, c, ..., c],
                                     [c, c, ..., c],
                                     [c, c, ..., c]]

                    | targets.view(B*T)
                    v

targets AFTER:  (B*T,) = (32,)      [c, c, c, ... , c]  <- 32 answers in a row
```

Order is what makes this safe: `view` does not shuffle anything, so `logits` row
`i` and `targets` entry `i` are still the same position. Now the shapes match
what `F.cross_entropy` wants — `(N, C)` scores and `(N,)` answers, with
`N = B*T = 32` — and it can score all `32` predictions at once.

```text
F.cross_entropy( (32, 83) logits , (32,) targets )  ->  one average loss number
```

We can sanity-check the number before any training. With `83` characters and no
learning at all, the best a model can do is guess uniformly — probability
`1/83` on the right answer. That gives an expected loss of:

```text
-ln(1 / 83) = ln(83) ~= 4.41
```

Our untrained model prints about `4.9` — a little worse than `4.41`, because the
random initial weights are not even uniform yet. Close to the expected value is
the sign we wanted: nothing is badly broken.

### Generating Text

`generate` makes the model write new text. You hand it some starting text, and
it repeatedly does one thing: <span style="color:#8aff8a"><strong>predict one
character, stick it on the end</strong></span>. Each loop makes the text one
character longer.

The text-so-far lives in `idx`, a tensor of character ids with shape `(B, T)`:

```text
B = how many sequences we are writing at once
T = how many characters are in each sequence RIGHT NOW
```

`T` is just a length counter. Every loop appends one character, so `T` ticks up
by one: `T -> T+1 -> T+2 -> ...`. That is the whole "grows one character at a
time" idea — the tensor gains one column each pass:

```text
idx shape over time (with B = 1):

  start        T = 1     [[0]]
  after pass 1 T = 2     [[0, 41]]
  after pass 2 T = 3     [[0, 41, 12]]
  after pass 3 T = 4     [[0, 41, 12, 7]]
                                     ^ newest character, appended each loop
```

And "for every sequence in the batch at once" just means: if `B = 3`, it writes
three separate texts in parallel, adding one character to *all three* each loop:

```text
            after pass 1      after pass 2
  text 0:   [0, 41]           [0, 41, 12]
  text 1:   [0,  7]           [0,  7, 53]
  text 2:   [0, 22]           [0, 22,  9]
            \____ one column added to every row, every pass ____/
```

In our run we only generate one sequence (`B = 1`), so you can ignore the batch
part for now — it is there only because the same code handles many at once.

```python
def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        logits, loss = self(idx)            # targets is None -> loss is None
        logits = logits[:, -1, :]           # focus on the last step: (B, C)
        probs = F.softmax(logits, dim=-1)   # scores -> probabilities (B, C)
        idx_next = torch.multinomial(probs, num_samples=1)   # sample (B, 1)
        idx = torch.cat((idx, idx_next), dim=1)   # append: (B, T+1)
    return idx
```

Walking the loop:

```text
1. self(idx)        -> run the model. No targets here, so loss is None;
                       we do not have answers when generating.
2. logits[:, -1, :] -> keep only the LAST position's scores. That is the
                       prediction for the next character.
3. softmax          -> turn raw scores into a probability distribution.
4. multinomial      -> sample one character from that distribution
                       (not always the top one — sampling adds variety).
5. torch.cat        -> glue the new character onto the running sequence.
```

The tricky part is how the **shape changes** at each step. Here is one pass
through the loop, following the tensor (using `B = 1`, current length `T = 3`,
vocabulary `C = 83`):

```text
  idx                       (B, T)     = (1, 3)     the text so far: [[0, 41, 12]]
   |
   |  self(idx)  -> score every position
   v
  logits                    (B, T, C)  = (1, 3, 83) a row of 83 scores per position
   |
   |  logits[:, -1, :]  -> throw away all but the LAST position
   v
  logits                    (B, C)     = (1, 83)    83 scores for the next char only
   |
   |  softmax  -> normalize scores into probabilities (still 83 numbers)
   v
  probs                     (B, C)     = (1, 83)    e.g. [0.01, 0.00, 0.07, ...]
   |
   |  multinomial  -> draw ONE character id from those probabilities
   v
  idx_next                  (B, 1)     = (1, 1)     e.g. [[7]]
   |
   |  torch.cat((idx, idx_next), dim=1)  -> append to the end
   v
  idx                       (B, T+1)   = (1, 4)     [[0, 41, 12, 7]]
```

Read top to bottom: we go from the whole sequence, to scores for *every*
position, down to scores for just the *last* position, to probabilities, to a
single sampled character, and finally back to a sequence that is one character
longer. That last `idx` becomes the input to the next pass, and the cycle
repeats.

The two shape moves that trip people up:

```text
logits[:, -1, :] : (B, T, C) -> (B, C)   drop the time axis, keep last step
torch.cat(...)   : (B, T)    -> (B, T+1) add one column back on
```

One quirk worth noticing: we pass the
<span style="color:#93c5fd"><strong>entire</strong></span> growing `idx` into the
model every step, but the bigram only ever uses the
<span style="color:#8aff8a"><strong>last character</strong></span> — it ignores
everything before it. That is wasteful here, but we write it this general way on
purpose, because the Transformer *will* use the whole history.

To kick generation off, we feed a single starting character — index `0`, which
is the newline character:

```python
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long),
                        max_new_tokens=100)[0].tolist()))
```

```text
cLEvM9zu|Kx-bb!FKG52'G g8,?J'C?MNatIZlDZnrjxzu0:SH4u'Ye a|5"|ohDrH' ,-o8to ap"N;FlR'Ykv1Q"m0?j b|!
```

Pure gibberish — which is exactly right for an untrained model. The plumbing
works end to end: model -> logits -> loss -> sampled text. All that is left is to
actually train it, and later to replace the lookup table with attention.

Memory hook:

```text
logits  = raw scores for the next character, shape (B, T, C)
loss    = cross-entropy, after reshaping to (B*T, C) and (B*T)
generate = predict last step -> softmax -> sample -> append -> repeat
```

## Training The Model

So far the model has random weights, which is why it produces gibberish.
Training means <span style="color:#8aff8a"><strong>nudging those weights, over
and over, so the loss goes down</strong></span> — so the model puts more
probability on the character that actually comes next.

### The Optimizer

First we create an optimizer. Its whole job is to take the weights and update
them using the gradients we compute.

```python
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
```

We use <span style="color:#ffff99"><strong>Adam</strong></span> (here the `AdamW`
variant). You can think of it as a smarter version of plain gradient descent: it
keeps a little memory of past gradients and adapts the step size per weight, so
it trains faster and more reliably. For a tiny model like this we can also afford
a higher learning rate, `1e-3`.

```text
lr (learning rate) = how big a step we take each update
   too small -> learns very slowly
   too big   -> overshoots and the loss bounces around
```

Because this model is so small, we also bump the batch up to `32`, so each step
learns from more examples at once:

```python
batch_size = 32   # more sequences per step than the 4 we used to demo
```

### The Loop

Now we repeat the same four moves thousands of times. Each pass is one *step*:

```python
batch_size = 32
for steps in range(10000):   # increase number of steps for good results...

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
```

What each line means:

```text
get_batch('train')  -> grab a fresh random batch of chunks (the data feeder
                       from earlier). New examples every step = no memorizing
                       one fixed batch.

m(xb, yb)           -> forward pass. Run the model and measure the loss:
                       how wrong are these predictions?

optimizer.zero_grad -> reset gradients to zero. PyTorch ADDS up gradients by
                       default, so we must clear last step's before computing
                       this step's, or they pile up.

loss.backward()     -> backward pass. Work out, for every weight, which
                       direction would lower the loss (the gradients).

optimizer.step()    -> actually move the weights a small step in that
                       direction. This is the moment learning happens.
```

A clean mental model of one step:

```text
  sample batch  ->  forward (get loss)  ->  zero_grad  ->  backward  ->  step
       ^                                                                  |
       |__________________ repeat 10,000 times ___________________________|

  each lap: how wrong are we?  ->  which way is better?  ->  nudge that way
```

The order of `zero_grad -> backward -> step` matters: clear the old gradients,
compute fresh ones, then take the step. After `10000` steps our loss drops from
the starting `~4.9` to about:

```text
2.286759376525879
```

Lower loss means less surprise — the model is genuinely predicting better now,
not guessing uniformly.

### Generating Again

The real payoff is sampling from the *trained* model. Same `generate` call, now
asking for `300` characters:

```python
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long),
                        max_new_tokens=300)[0].tolist()))
```

```text
Thitheap ry dmey whd s y Thecomoy Lisorioupt imy's s hien. bund ppen actuathexcan t e a licarid cifurch
To Her an wed le sid wapus s and ly mext|>
Heno aill mm, he he. ga, tened waind s ued wake ft't Mait deakny brarote a goowied." tatodlerket g e s hed tit be, npy Sherfled. mbee bawobon he tonthew
```

Still not English — but compare it to the earlier gibberish. There are now real
word shapes, spaces in sensible places, capital letters after line breaks. A
bigram can only ever look one character back, so this is about as good as it
gets. To do better, the model needs to look at *more* of the context — which is
exactly what self-attention and the Transformer will add next.

Memory hook:

```text
optimizer = the thing that updates weights from gradients (we use AdamW)
one step  = sample batch -> forward (loss) -> zero_grad -> backward -> step
training  = repeat the step thousands of times until loss stops dropping
```

## The Math Trick At The Heart Of Self-Attention

Everything so far has each token predicting on its own. The bigram never lets
tokens *talk to each other* — token 5 has no idea what tokens 1–4 were. Real
language needs context, so we need a way for tokens to <span style="color:#8aff8a"><strong>communicate</strong></span>.
Before the full self-attention machinery, there is a small mathematical trick
that makes that communication cheap. This section builds the intuition; the next
ones will show progressively faster ways to do the same thing.

### The Setup: B, T, C

We work with a tensor of shape `(B, T, C)`:

```text
B  = batch        -> how many independent sequences we process at once
T  = time         -> positions in the sequence (our 8 tokens, in order)
C  = channels     -> the vector of information stored at each position
```

So for one sequence we have `T = 8` tokens lined up in time, and each token
carries a `C`-dimensional vector describing it.

### The Rule: Information Only Flows Forward

The 8 tokens are currently <span style="color:#ff8a8a"><strong>not talking to each
other</strong></span>. We want to couple them — but in a very specific way.

Token at position 5 may look at positions 1, 2, 3, 4 and itself. It must
<span style="color:#ff8a8a"><strong>never</strong></span> look at positions 6, 7, 8.
Those are *future* tokens, and the whole game is to predict the future — if we
let a token peek ahead, we'd be cheating.

```text
position:     1     2     3     4     5     6     7     8
                                      ^
                              "I am token 5"

  can see:  [ 1 ]-[ 2 ]-[ 3 ]-[ 4 ]-[ 5 ]   X     X     X
            \________ past + present ______/  \__ future __/
                  information flows ->            (off-limits)
```

Information only flows from <span style="color:#93c5fd"><strong>previous context to
the current timestep</strong></span>. Never backward from the future.

### The Easiest Way To Communicate: Average The Past

What is the simplest possible way for token 5 to gather its history? Just
<span style="color:#ffff99"><strong>take the average of itself and all previous
tokens</strong></span>.

If I am token 5, I grab the channel vectors at steps 1, 2, 3, 4, 5, add them up,
and divide by 5. That average becomes a new feature vector that summarizes
<span style="color:#8aff8a"><strong>"me, in the context of my history."</strong></span>

```text
token 1:  x[1]                                  -> avg of {1}
token 2:  x[1], x[2]                            -> avg of {1,2}
token 3:  x[1], x[2], x[3]                      -> avg of {1,2,3}
token 4:  x[1], x[2], x[3], x[4]                -> avg of {1,2,3,4}
token 5:  x[1], x[2], x[3], x[4], x[5]          -> avg of {1,2,3,4,5}
                                                       ^
                            each token's new value = mean of itself + its past
```

Every position ends up holding the running average of everything up to and
including it — a tiny summary of its own backstory.

### Why Averaging Is Weak (But Fine For Now)

An average is <span style="color:#ff8a8a"><strong>lossy</strong></span>. It throws
away *which* token came where — the order and spacing of the past all collapse
into one blurred mean. It is an extremely weak form of communication.

That's okay. We start with the crudest version to get the mechanics right, then
upgrade it. Coming up next: the same "average the past" idea written several
different ways — first with loops, then with matrix multiplication, and finally
in the form that turns into real self-attention.

Memory hook:

```text
(B, T, C)   = batch, time (positions), channels (info per token)
the rule    = a token may only see itself + earlier tokens, never the future
the trick   = each token = average of itself and all previous tokens
why average = simplest way to mix the past; lossy, but we fix that later
```

## Three Ways To Average The Past

We'll now write that "average my history" idea **three different ways**. They
all produce the *exact same numbers* — the journey from version 1 to version 3 is
the journey from a slow loop to the operation that becomes real self-attention.

### Version 1: The Obvious For-Loop ("bag of words")

The direct way: literally loop over every batch and every time step, grab the
tokens up to now, and average them.

```python
# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B, T, C))
for b in range(B):              # each sequence in the batch, independently
    for t in range(T):          # each position in time
        xprev = x[b, :t+1]      # (t+1, C) -> this token + everything before it
        xbow[b, t] = torch.mean(xprev, 0)   # average over the time dimension
```

The name <span style="color:#ffff99"><strong>`xbow`</strong></span> stands for
**"bag of words"** — when you average up a set of word/token embeddings and
forget their order, you're treating them like an unordered *bag*. That's exactly
what this does at every position.

Reading the loop:

```text
b loops over batches         -> handle each sequence on its own
t loops over time            -> handle each position on its own
xprev = x[b, :t+1]           -> shape (t+1, C): how many tokens are "in the past"
                                grows as t grows -> 1, then 2, then 3, ...
torch.mean(xprev, 0)         -> average DOWN the time axis (dim 0),
                                leaving one (C,) vector: the running summary
```

What comes out — position 1 is unchanged (it only has itself), position 2 is the
average of rows 1–2, position 3 the average of rows 1–3, and so on:

```text
        x  (one sequence, T rows of C channels)        xbow (the averages)

  t=1   [ a ........... ]                               [ a ]               <- just itself
  t=2   [ b ........... ]                               [ (a+b)/2 ]         <- avg of 1..2
  t=3   [ c ........... ]            ===>               [ (a+b+c)/3 ]       <- avg of 1..3
  t=4   [ d ........... ]                               [ (a+b+c+d)/4 ]     <- avg of 1..4
   :                                                     :
```

It works, but those Python loops are slow. We'd love to do this with **one**
fast operation. That operation is matrix multiplication.

### A Quick Detour: Matrix Multiply As Weighted Sums

Before version 2, here's the key insight that makes it click. A matrix multiply
`a @ b` produces, for each output row, a **weighted sum of the rows of `b`** —
and the weights are the numbers in the corresponding row of `a`.

Start with `a` as all ones:

```python
a = torch.ones(3, 3)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
```

```text
   a (3x3)          b (3x2)            c = a @ b  (3x2)
 [1 1 1]          [2  7]             [14  16]   <- 2+6+6 , 7+4+5  (sum of ALL rows)
 [1 1 1]    @     [6  4]      =      [14  16]   <- same
 [1 1 1]          [6  5]             [14  16]   <- same
```

Every row of `a` is `[1 1 1]`, so every output row is the **sum of all rows of
`b`**. Now the magic — make `a` lower-triangular with `torch.tril`:

```python
a = torch.tril(torch.ones(3, 3))
```

```text
torch.tril keeps the lower-left triangle, zeros above the diagonal:

 [1 0 0]
 [1 1 0]
 [1 1 1]
```

Multiply again and watch what the zeros do — they **switch off** the future rows:

```text
   a (tril)         b                 c = a @ b
 [1 0 0]          [2  7]             [ 2   7]   <- 1*row1                = row1 only
 [1 1 0]    @     [6  4]      =      [ 8  11]   <- 1*row1 + 1*row2       = rows 1..2
 [1 1 1]          [6  5]             [14  16]   <- row1 + row2 + row3    = rows 1..3
```

The zeros in `a` mean "ignore that row." Row 1 of the answer sees only the first
row of `b`; row 2 sees the first two; row 3 sees all three. **This is exactly the
"only look at the past" rule, expressed as a matrix.** Right now it gives *sums*,
though — we want *averages*. So normalize each row of `a` to sum to 1:

```python
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)   # divide each row by its own sum
```

```text
 [1.0000  0       0     ]      row sums to 1   -> weights are all 1/(count so far)
 [0.5000  0.5000  0     ]
 [0.3333  0.3333  0.3333]

   a (normalized)    b                 c = a @ b
 [1.00 0    0   ]   [2  7]            [2.0000  7.0000]   <- just row1
 [0.50 0.50 0   ] @ [6  4]     =      [4.0000  5.5000]   <- average of rows 1..2
 [0.33 0.33 0.33]   [6  5]            [4.6667  5.3333]   <- average of rows 1..3
```

Now each output row is a **weighted average of the past** — and the lower-
triangular shape guarantees we never reach into the future. (`keepdim=True` keeps
the sum as a column `(3,1)` so broadcasting divides each row correctly.)

### Version 2: The Same Average, As One Matrix Multiply

Apply that detour to the real tensor. Build a `(T, T)` weight matrix `wei`, then
multiply it by `x`:

```python
# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x       # (T, T) @ (B, T, C) ---> (B, T, C)
torch.allclose(xbow, xbow2)   # True
```

The `wei` matrix is just the normalized triangle, now `8x8`:

```text
wei  (each row = "how much of each past token to blend in")
[1.000 0     0     0     0     0     0     0    ]   pos 1: 100% itself
[0.500 0.500 0     0     0     0     0     0    ]   pos 2: half + half
[0.333 0.333 0.333 0     0     0     0     0    ]   pos 3: thirds
[0.250 0.250 0.250 0.250 0     0     0     0    ]   pos 4: quarters
[0.200 0.200 0.200 0.200 0.200 0     0     0    ]   ...
[0.167 0.167 0.167 0.167 0.167 0.167 0     0    ]
[0.143 0.143 0.143 0.143 0.143 0.143 0.143 0    ]
[0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125]   pos 8: averages all 8
```

About the shapes — `wei` is `(T, T)` and `x` is `(B, T, C)`. They don't match, so
PyTorch <span style="color:#93c5fd"><strong>broadcasts</strong></span> `wei`
across the batch and runs a **batched matrix multiply** in parallel:

```text
        (T, T)  @  (B, T, C)
   PyTorch adds a batch dim ->  (B, T, T) @ (B, T, C)
   per batch element:           (T, T) @ (T, C)  =  (T, C)
   stacked back up:             (B, T, C)
```

`torch.allclose(xbow, xbow2)` returns `True` — the slow loop and the one-line
matmul give identical results. We just replaced two Python loops with a single
GPU-friendly operation.

### Version 3: Softmax — The Version That Becomes Attention

Version 3 produces the *same* averages again, but assembles `wei` in a way that
sets us up for real attention. Watch the four steps:

```python
# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)   # True
```

Step by step, watching `wei` transform:

```text
(1) tril = lower-triangular ones        (2) wei = zeros, then masked_fill
    -- the "who may talk" mask --           -- future positions -> -inf --

  [1 0 0 0 0 0 0 0]                        [0  -inf -inf -inf -inf -inf -inf -inf]
  [1 1 0 0 0 0 0 0]                        [0   0   -inf -inf -inf -inf -inf -inf]
  [1 1 1 0 0 0 0 0]                        [0   0    0   -inf -inf -inf -inf -inf]
  [1 1 1 1 0 0 0 0]            ===>        [0   0    0    0   -inf -inf -inf -inf]
  [1 1 1 1 1 0 0 0]                        [0   0    0    0    0   -inf -inf -inf]
   ...                                      ...
```

```text
(3) softmax along each row (dim=-1)     -> exponentiate, then divide by row sum

  exp(0) = 1   and   exp(-inf) = 0

  so each row of -inf/0 values turns straight back into the averaging weights:

  [1.000 0     0     0    ...]
  [0.500 0.500 0     0    ...]
  [0.333 0.333 0.333 0    ...]
   ...                              <- IDENTICAL to wei from version 2
```

`-inf` becomes `0` after softmax (a future token gets **zero weight** — it is not
aggregated), and the remaining `0`s become equal positive weights that sum to 1.
`torch.allclose(xbow, xbow3)` is `True` yet again.

### Why Bother With Version 3?

Versions 1 and 2 hard-code a flat average — every past token counts equally. But
look at what `wei` actually *is* in version 3:

```text
wei = torch.zeros((T,T))                          <- start: all affinities 0
wei = wei.masked_fill(tril == 0, float('-inf'))   <- future: forbidden (-inf)
wei = F.softmax(wei, dim=-1)                       <- turn into weights that sum to 1
```

Read `wei` as a table of <span style="color:#8aff8a"><strong>affinities</strong></span>
— *how interesting does each past token find every other past token?*

```text
        wei[t, i]  =  "how much should position t pull in info from position i?"

        right now we START these affinities at zero (a flat average),
        but they DO NOT have to stay flat.
```

Here's the whole point:

```text
  zeros        ->  affinities begin neutral; nothing is special yet
  masked_fill  ->  the future is set to -inf: those tokens CAN'T communicate
  softmax      ->  -inf -> 0 weight (blocked), the rest -> data-dependent weights

  In real attention these affinities are NOT fixed. They are computed from the
  tokens themselves, so a token can decide some past tokens are far more
  interesting than others -- and weight them more heavily.
```

So the long story, in one line:

> You can do a **weighted aggregation of your past** with a matrix multiply, where
> a **lower-triangular** matrix says *which* tokens may contribute, and the values
> in it say <span style="color:#ffff99"><strong>how much of each past token fuses
> into the current position</strong></span>.

A flat average was just the special case where all the weights are equal.
Self-attention is what happens when we let the data *learn* those weights — that
is the very next step.

Memory hook:

```text
v1 (loop)    = for each t, average x[:t+1]   -- correct but slow ("bag of words")
v2 (matmul)  = wei @ x, wei = normalized tril -- same numbers, one fast op
v3 (softmax) = zeros -> mask future to -inf -> softmax -> wei @ x  -- same again
tril         = the "only see the past" rule, written as a matrix
wei          = affinities: HOW MUCH of each past token fuses into position t
the punchline= let wei be LEARNED from the data -> that is self-attention
```
