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

---

## Self-Attention: Letting Tokens Decide Who To Listen To

Everything so far used `wei = torch.zeros((T, T))` — a **flat average**. Every past
token counts exactly the same:

```text
   wei (the flat average)              what it means

  [1.00  0     0    ...]               token 4 listens to tokens 1,2,3,4
  [0.25  0.25  0.25  0.25 ...]         ...with EQUAL weight to each
        ^^^^^^^^^^^^^^^^^^
        uniform -- boring -- token 4 has no opinion about WHO matters
```

But that is not how language works. Different tokens care about different things:

```text
   a vowel scanning the past for a consonant:

   "...   t   h   e        c   a [t]"            <- I am 'a', a vowel
                                  ^
          I don't care equally about every past token.
          I'm specifically HUNTING for consonants near me.

          I want to pull in a LOT from 'c' and 't', a little from 'the'.
```

We want each token to gather from the past in a
<span style="color:#8aff8a"><strong>data-dependent</strong></span> way — weights that
*depend on the actual tokens*, not a fixed flat average. **Self-attention** is the
mechanism that makes this happen.

### The Big Idea: Query and Key

Every token emits **two little vectors**:

```text
   ┌─────────────────────────────────────────────────────────────┐
   │  QUERY  (q)  =  "what am I looking for?"                      │
   │                  e.g. 'a' the vowel: "I want a consonant"     │
   │                                                               │
   │  KEY    (k)  =  "what do I contain / what am I?"              │
   │                  e.g. 'c': "I am a consonant"                 │
   └─────────────────────────────────────────────────────────────┘
```

How do we measure whether token A's *query* matches token B's *key*? **Dot product.**

```text
   affinity(A, B)  =  q_A · k_B      (a single number)

   q_A and k_B point the SAME way   ->   big dot product   ->  HIGH affinity
   q_A and k_B unrelated            ->   ~zero dot product  ->  low affinity

   "the vowel's query lines up with the consonant's key -> they bond"
```

So the `wei` table — which used to be hard-coded zeros — is now **built from the
tokens themselves**: every entry `wei[t, i]` is the dot product of token `t`'s query
with token `i`'s key.

### A Single Head Of Self-Attention (the code)

```python
# let's see a single Head perform self-attention
head_size = 16
key   = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)     # (B, T, 16)
q = query(x)   # (B, T, 16)
wei = q @ k.transpose(-2, -1)   # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T, T))     # <- the OLD flat version, now replaced
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v   # instead of  wei @ x

out.shape   # torch.Size([4, 8, 16])
```

### What `nn.Linear` Actually Does

`nn.Linear(in_features, out_features, bias=False)` is just a matrix multiply. It
holds a weight matrix and maps each token's `C`-dim vector down to `head_size`:

```text
   nn.Linear(C, head_size, bias=False)

      weight  W : shape (head_size, C)   <- learned, init'd random (Kaiming uniform)
      bias      : NONE here (bias=False), otherwise a (head_size,) vector added on

   for one token vector  x_t  of shape (C,):

      k_t = W @ x_t      ->  shape (head_size,) = (16,)

   "take this token's 32 numbers, mix them into 16 numbers that ARE its key"
```

Three separate Linears = three different learned projections of the same token:

```text
        x_t (C=32)
          │
     ┌────┼────┬─────────┐
     ▼    ▼              ▼
   key  query         value          each is its own (head_size, C) matrix
     │    │              │
     ▼    ▼              ▼
   k_t  q_t            v_t   (each 16-dim)
  "what  "what          "what I'll
   I am"  I want"        actually share"
```

### Walking The Dimensions

```text
   x       : (B, T, C)  = (4, 8, 32)   <- batch of 4, 8 tokens each, 32 channels

   k = key(x)     -> (B, T, 16)        every token gets a 16-dim KEY
   q = query(x)   -> (B, T, 16)        every token gets a 16-dim QUERY

   wei = q @ k.transpose(-2, -1)

         q                k.transpose(-2,-1)         wei
     (B, T, 16)    @       (B, 16, T)        --->   (B, T, T)
     (4, 8, 16)    @       (4, 16, 8)        --->   (4, 8, 8)
```

Why does that shape work out to `(T, T)`? Because matrix-multiplying *all queries*
against *all keys* gives **every (query, key) pair** at once:

```text
              k1   k2   k3  ...  k8        (keys along the top)
            ┌────────────────────────┐
        q1  │ q1·k1 q1·k2 ...        │     row t = token t's QUERY
        q2  │ q2·k1 q2·k2 ...        │     dotted against EVERY key
        q3  │ q3·k1 q3·k2 ...        │
        ... │                        │     entry (t, i) = q_t · k_i
        q8  │ ...              q8·k8  │                 = affinity of t for i
            └────────────────────────┘
                                          shape: (T, T) = (8, 8)
```

The key change from before:

```text
   OLD:  wei = zeros   ->  SAME flat table reused for every batch element
   NEW:  wei = q @ kᵀ  ->  EACH batch element gets its OWN table,
                            because each has its own tokens -> own q, k

   wei is no longer a constant. It is DATA-DEPENDENT.
```

### Mask + Softmax (same two steps as before)

The raw affinities `wei[0]` (one batch element, **before** masking) are just numbers
— positive where query/key align, negative where they don't:

```text
   wei[0]  (raw scores, q·k)              after masked_fill(tril==0, -inf)

  [-1.76  0.55 -0.83  ...]               [-1.76  -inf  -inf  -inf ...]
  [-3.33 -1.66  2.04  ...]    mask the   [-3.33 -1.66  -inf  -inf ...]
  [-1.02 -1.26  0.08  ...]    future     [-1.02 -1.26  0.08  -inf ...]
   ...                        ───────>    ...
```

The mask is the **same `tril` trick** — a token still may not see the future. Then
softmax turns each row into weights that sum to 1:

```text
   wei[0]  after  F.softmax(dim=-1)   -- now DATA-DEPENDENT weights:

   row 6 (token 6 deciding who to listen to):
   [0.0176, 0.2689, 0.0215, 0.0089, 0.6812, 0.0019,  0,  0]
              ▲▲▲▲                   ▲▲▲▲▲▲
            position 2             position 5
          "kind of interesting"  "VERY interesting"

   token 6's query lined up strongly with position 5's key (0.68),
   somewhat with position 2's key (0.27), and basically ignored the rest.

   Compare to the OLD flat average, which would have given every
   past token 1/6 = 0.167.  Self-attention has OPINIONS.
```

### The Value Vector: What A Token Shares

One last piece. We do **not** aggregate the raw token `x` anymore. Each token also
emits a **value** `v = value(x)`, and *that* is what gets aggregated:

```text
   out = wei @ v          (not  wei @ x)

      x_t  =  the token's PRIVATE identity (all 32 channels of raw info)
      v_t  =  "if you decide to listen to me, THIS is what I'll tell you"

   think of it as:
      q  =  what I'm looking for
      k  =  what I am (for matching purposes)
      v  =  what I actually deliver if matched
```

```text
   out = wei @ v

     (B, T, T)   @   (B, T, 16)   --->   (B, T, 16)
     (4, 8, 8)   @   (4, 8, 16)   --->   (4, 8, 16)

   each output token = weighted sum of the VALUES of the tokens it chose to attend to
```

So `out` comes out `head_size`-dim (16), not `C`-dim — the head has gathered the
past into a fresh 16-dim summary, weighted by genuine, learned interest.

### Memory hook

```text
problem  = flat average treats every past token equally -- no opinions
fix      = self-attention: gather from the past in a DATA-DEPENDENT way

each token emits 3 vectors (via nn.Linear projections of x):
   query (q) = what am I looking for?
   key   (k) = what do I contain?
   value (v) = what will I share if you listen to me?

wei = q @ kᵀ          -> affinity of every token for every other (B,T,T)
   (replaces the old hard-coded zeros -- now built FROM the tokens)
mask -inf + softmax   -> still no peeking at the future; rows sum to 1
out = wei @ v         -> weighted sum of VALUES, not raw x   (B,T,head_size)

x = private identity   |   v = what I broadcast   |   wei = how much you tune in
```

---

## Six Things To Really Understand About Attention

Now that a single head works, here are the ideas that make attention click — the
ones easy to miss the first time.

### 1) Attention Is A Communication Mechanism (it's a graph)

Forget words for a second. Attention is just **nodes in a directed graph** passing
messages. Each node holds a vector of info; a node updates itself by taking a
**weighted sum of every node that points to it**.

```text
   a GENERAL attention graph                our LANGUAGE-MODEL graph
   (any edges allowed)                      (autoregressive: only point backward)

        ┌──►(1)◄──┐                          (0)   (1)   (2)  ...  (7)
       (0)        (2)                         ▲     ▲     ▲          ▲
        ▲  ╲    ╱  │                          │     │     │          │
        │   ╳     ▼                       node 0: just itself
       (5)─►(4)─►(3)                       node 1: 0, 1
            edges point                    node 2: 0, 1, 2
            every which way                ...
                                           node 7: 0,1,2,3,4,5,6,7  (all the past)
```

For an 8-token block (`block_size = 8`) the graph is fixed and triangular:

```text
   node 0  ◄── {0}                     "I can only see myself"
   node 1  ◄── {0,1}                   "myself + 1 token back"
   node 2  ◄── {0,1,2}
   node 3  ◄── {0,1,2,3}
   node 4  ◄── {0,1,2,3,4}
   node 5  ◄── {0,1,2,3,4,5}
   node 6  ◄── {0,1,2,3,4,5,6}
   node 7  ◄── {0,1,2,3,4,5,6,7}       "myself + all 7 tokens of history"

   This triangular pattern IS the tril mask. The autoregressive rule = the graph.
```

### 2) Attention Has No Sense Of Space

Attention acts over a **set** of vectors. The nodes have **no idea where they sit**
in the sequence — there are no arrows that say "I'm 3rd." That's why we had to add
**positional embeddings**: we *anchor* each token with its position so it knows where
it is.

```text
   attention sees:   { •  •  •  •  •  •  •  • }   <- an unordered SET
                       no built-in "1st, 2nd, 3rd..."

   so we inject it:  token_emb + position_emb     <- now each node knows its slot

   (contrast: a CONVOLUTION has a fixed spatial layout — the filter slides over
    a KNOWN grid of positions. Attention does not. It must be TOLD positions.)
```

### 3) Batch Elements Never Talk To Each Other

The `B` dimension is fully independent. With `B=4, T=8`, you have **4 separate pools
of 8 nodes**. Nodes only mix *within* their own pool.

```text
   batch 0:  (•••••••• )   <- 8 nodes, fully connected (causally) to each other
   batch 1:  (•••••••• )   <- separate pool, never sees batch 0
   batch 2:  (•••••••• )
   batch 3:  (•••••••• )

   32 nodes processed in total, but as 4 isolated groups of 8. No cross-pool edges.
```

### 4) Encoder vs Decoder: The Mask Is Optional

The `masked_fill(tril == 0, -inf)` line is the *only* thing forbidding the future.
That's a **decoder** (autoregressive, for generation). Delete that one line and every
node talks to every other node — that's an **encoder** block.

```text
   DECODER (what we built)            ENCODER (delete the mask line)
   wei.masked_fill(tril==0, -inf)     # no mask at all

   only see the past                  everyone sees everyone
   needed for GENERATING text         great for e.g. sentiment analysis,
                                       where the whole sentence is allowed
                                       to talk to itself

   Attention itself does NOT care. Both are valid; the mask is a choice.
```

### 5) Self-Attention vs Cross-Attention: Where Q, K, V Come From

We built **self-attention**: query, key, and value all come from the **same** `x`.
The nodes are attending to themselves.

```text
   SELF-ATTENTION                       CROSS-ATTENTION
   q, k, v  all from  x                 q from x  |  k, v from ANOTHER source

   x ──► query                          x ──► query
   x ──► key                            (encoder output) ──► key
   x ──► value                          (encoder output) ──► value

   "nodes talk among themselves"        "MY nodes pull info from an external
                                         set of nodes (e.g. an encoder that
                                         holds the context to condition on)"
```

Cross-attention is how you *condition* on outside context: you produce queries from
your own tokens, but read keys/values off a separate stack of nodes.

### 6) Scaled Attention: The `1/√(head_size)` You're Missing

Look at the paper's formula — there's a divisor we haven't used yet:

```text
                          ⎛  Q · Kᵀ  ⎞
   Attention(Q,K,V) = softmax⎜ ─────── ⎟ V          d_k = head_size
                          ⎝  √d_k   ⎠
```

**Why divide by √(head_size)?** If `q` and `k` are unit-Gaussian (variance ≈ 1), then
`wei = q @ kᵀ` has variance on the order of `head_size` — the numbers get **big**.

```text
   WITHOUT scaling:                       WITH  * head_size**-0.5 :
   k.var() ≈ 1.04                         k.var() ≈ 0.90
   q.var() ≈ 1.07                         q.var() ≈ 1.00
   wei.var() ≈ 17.5   <- blown up         wei.var() ≈ 1.00   <- tamed
```

Why does a blown-up `wei` matter? It feeds straight into **softmax**, and softmax on
large-magnitude inputs **sharpens toward a one-hot vector**:

```text
   softmax([0.1, -0.2, 0.3, -0.2, 0.5])      -> [0.19, 0.14, 0.24, 0.14, 0.29]
                                                  nicely DIFFUSE -- spreads attention

   softmax([0.1, -0.2, 0.3, -0.2, 0.5] * 8)  -> [0.03, 0.00, 0.16, 0.00, 0.80]
                                                  PEAKY -- collapses onto one node
```

A too-peaky softmax means each token basically copies a **single** other node instead
of *blending* the past. Why is that bad? Two reasons:

**(a) You throw away context.** The whole point of attention is that a token's meaning
comes from *several* past tokens together. Collapse to one-hot and you grab info from
exactly one position, ignoring the rest:

```text
   blended (wei.var ≈ 1):   out = 0.3·v2 + 0.4·v5 + 0.3·v6   <- mixes 3 contexts
   peaky   (wei.var ≈ 17):  out = 1.0·v5                     <- just a COPY of v5

   all that q/k/v machinery, and the aggregation became a no-op.
```

**(b) The real killer — it breaks learning at initialization.** At the start of
training `q` and `k` are random, so the affinities are essentially **noise**. There is
no real signal yet about which past token matters.

```text
   diffuse softmax  ->  noisy guess -> gentle near-uniform blend
                        gradient flows to ALL past tokens -> model gently learns

   peaky softmax    ->  random noise amplified into a CONFIDENT one-hot
                        model commits hard to whatever the random init favored...
                        ...a decision based on nothing
```

And softmax is nearly flat (derivative → 0) once it saturates, so a peaky softmax also
**starves the other nodes of gradient** — it makes a bad random choice *and* makes that
choice hard to correct later. Learning stalls.

```text
   softmax saturated near one-hot  ->  gradient ≈ 0 for the non-selected nodes
                                       (the vanishing-gradient region of softmax)
```

Dividing by `√(head_size)` keeps `wei.var ≈ 1`, so at init the softmax stays diffuse:
attention starts as a soft average over the past (much like the flat-average baseline
from earlier) and *learns* to sharpen onto the right tokens as training reveals real
structure. You want sharpening to be **earned from data**, not handed out for free by
an unlucky random seed.

> One-line version: a peaky-at-init softmax makes **confident decisions from noise**,
> and then can't backpropagate its way out of them.

### Memory hook

```text
1. graph     = attention is nodes aggregating from whoever points to them;
               our LM graph is triangular (node t sees 0..t) = the tril mask
2. no space  = attention sees an unordered SET -> must ADD positional embeddings
3. batches   = B pools are independent; nodes never cross batch elements
4. mask      = keep it -> DECODER (causal, generation);
               drop it -> ENCODER (all-to-all, e.g. sentiment)
5. self/cross= self: q,k,v from x  |  cross: q from x, k,v from another source
6. scale     = divide q·kᵀ by √(head_size) so wei.var ≈ 1, keeping softmax
               diffuse instead of collapsing to one-hot
```

---

## Wiring A Self-Attention Head Into The Network

Time to package everything into a reusable module and actually plug it into the model.

```python
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5   # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)              # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v      # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out
```

Two small-but-important details that turn the loose code into a real module:

```text
   register_buffer('tril', ...)   <- tril is NOT a parameter (nothing to learn).
                                     register_buffer parks it on the module so it
                                     moves to GPU with .to(device) and is saved,
                                     but the optimizer never touches it.

   self.tril[:T, :T]              <- crop to the ACTUAL sequence length T, so the
                                     same head works whether T is 8 (training) or
                                     1,2,3... (early steps of generation).
```

Now wire one head into the model: embed tokens, add positions, **let them
communicate**, then project to vocab logits.

```python
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)                       # <-- NEW: self-attention
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                       # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb                                           # (B,T,C)
        x = self.sa_head(x)        # apply one head of self-attention.  # (B,T,C)
        logits = self.lm_head(x)                                        # (B,T,vocab_size)
        ...
```

### One Gotcha In `generate`: Crop The Context

The old bigram model could take any-length context. But now we have a
`position_embedding_table` that only knows positions `0 .. block_size-1`. Feed it a
sequence longer than `block_size` and it has no embedding for position 9, 10, ... and
crashes. So in `generate` we **crop to the last `block_size` tokens**:

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -block_size:]     # <-- crop! never feed more than block_size
    logits, loss = self(idx_cond)
    logits = logits[:, -1, :]           # focus on the last time step -> (B, C)
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=1)
```

### Also: Drop The Learning Rate

```python
learning_rate = 1e-3    # was higher before
```

Self-attention is a more delicate beast than a plain lookup table — it **can't
tolerate a very high learning rate**. Lower it so training stays stable. The payoff:

```text
   plain bigram (no attention):   loss ≈ 2.5
   + one self-attention head:     loss ≈ 2.4   <- the head is clearly DOING something

   tokens are now communicating, and it shows up in the loss.
```

---

## Multi-Head Attention: Several Conversations At Once

One head gives each token **one** way to look at the past — a single query/key
pattern. But a token often needs to ask **several different questions at the same
time**:

```text
   the token 'a' (a vowel) might simultaneously want to know:

     head 1:  "where's the nearest consonant?"        (phonetics)
     head 2:  "what word am I inside of?"             (boundaries)
     head 3:  "is there a vowel I should agree with?" (harmony)
     head 4:  "how far back is the sentence start?"   (position-ish)

   one head can only chase ONE of these. We want all four AT ONCE.
```

> *"With a single attention head, averaging inhibits the ability to attend to
> information from different representation subspaces at different positions."*
> — Attention Is All You Need

So we run **several heads in parallel**, each with its own `q/k/v` projections (its own
"representation subspace"), and **concatenate** their outputs.

### How It's Done: Split, Attend, Concatenate

The trick is that the heads are **smaller**. Instead of one head of size 32, we use
**4 heads of size 8**, run them independently, and glue the results back together along
the channel dimension to recover the original 32:

```text
                       x  (B, T, 32)
                       │
        ┌──────────┬───┴───┬──────────┐         4 heads, each sees the SAME x
        ▼          ▼       ▼          ▼          but with its OWN q/k/v weights
     head 1     head 2  head 3     head 4
   (B,T,8)     (B,T,8) (B,T,8)    (B,T,8)        each outputs an 8-dim summary
        │          │       │          │
        └──────────┴───┬───┴──────────┘
                       ▼
              concat over channels
                  (B, T, 32)                     8 + 8 + 8 + 8 = 32 = n_embd

   four parallel "communication channels", each smaller, recombined into the full width
```

```python
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)   # concat over channels
```

```python
# in the model:
self.sa_heads = MultiHeadAttention(4, n_embd // 4)   # 4 heads of 8-dim self-attention
#                                   ^         ^
#                              num_heads   head_size = 32 // 4 = 8
```

```text
   WHY split instead of just adding more big heads?

   - each head specializes in a DIFFERENT kind of relationship (its own subspace)
   - running them in parallel is cheap and keeps total width = n_embd
   - concatenating lets the next layer mix all the specialists' findings

   it's like a GROUP CONVOLUTION: instead of one big conv over everything, you do
   several smaller convolutions in separate groups, then combine.
```

This is exactly the paper's `MultiHead(Q,K,V) = Concat(head₁,...,head_h) Wᴼ` — each
`headᵢ = Attention(QWᵢ, KWᵢ, VWᵢ)` is one of our small heads with its own projections.

---

## The Transformer Block: Communication + Computation

Here's the famous diagram from *Attention Is All You Need*. We can already spot the
pieces we've built:

<div align="center">
<img src="assets/10_transformer_architecture.png" alt="Figure 1 of Attention Is All You Need: the Transformer model architecture. Left column is the encoder (input embedding + positional encoding, then N times: multi-head attention with add and norm, feed forward with add and norm). Right column is the decoder (output embedding + positional encoding, then N times: masked multi-head attention, cross multi-head attention fed by the encoder, feed forward, each with add and norm), ending in linear and softmax to produce output probabilities" width="460">
</div>

<p align="center"><sub><em>From <a href="https://arxiv.org/pdf/1706.03762">Attention Is All You Need</a> (Vaswani et al., 2017) — left column is the encoder, right column is the decoder.</em></sub></p>

What we already have: **token + positional embedding** (the pink boxes `⊕` the
sinusoid), the **masked multi-head attention** (orange, decoder side), the **feed
forward** (blue), and the final **linear + softmax** (purple → green). We are building
only the **decoder** column (causal). We skip the middle orange box — the
**cross-attention to an encoder** — because a plain language model has no encoder to
condition on.

The one piece we haven't built yet sits right after attention: **Feed Forward**.

---

## Feed-Forward: Give Each Token Time To Think

After attention, the tokens have **looked at each other** and gathered information — but
they haven't had a moment to **process** what they gathered. Attention is *communication*;
it does not, by itself, do much *computation per token*.

```text
   attention   = the tokens TALK    (gather info from the past)
   feed-forward = each token THINKS  (digest what it just gathered)

   without the feed-forward, a token collects neighbors' values and immediately has to
   produce logits -- no chance to transform/combine that info first.
```

The feed-forward is just a tiny **MLP** applied to each token: a linear layer, then a
ReLU non-linearity.

```python
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
```

This matches the paper's `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂` — two linear layers with a
ReLU in between (`max(0, ·)` *is* ReLU).

**Why a non-linearity?** Without ReLU, stacking linear layers collapses into a single
linear layer — no extra power. The ReLU is what lets the token compute genuinely new,
non-linear features from what attention gathered.

```text
   Linear -> ReLU -> ... :  can represent curves, gates, "if this then that"
   Linear -> Linear     :  collapses to one Linear -- no thinking, just re-projection
```

**Crucially, this is per-token and independent.** Every token runs through the *same*
MLP on its *own* vector — no mixing across positions here (the mixing already happened
in attention).

```text
   token 0 ─► [MLP] ─► token 0'      same weights for every token,
   token 1 ─► [MLP] ─► token 1'      applied position-by-position,
   token 2 ─► [MLP] ─► token 2'      completely independently.
     ...                             (it's "position-wise")
```

```python
# in the model:
self.sa_heads = MultiHeadAttention(4, n_embd // 4)   # the tokens talk
self.ffwd     = FeedForward(n_embd)                  # then each token thinks
...
x = self.sa_heads(x)   # (B,T,C)  communication
x = self.ffwd(x)       # (B,T,C)  computation
logits = self.lm_head(x)
```

---

## The Block: Bundling Communication + Computation

Attention (talk) followed by feed-forward (think) is the repeating unit of a
Transformer. We wrap the pair into a single **`Block`** so we can stack it as many
times as we like:

```python
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)   # communication
        self.ffwd = FeedForward(n_embd)                     # computation

    def forward(self, x):
        x = self.sa(x)      # tokens talk to each other
        x = self.ffwd(x)    # each token thinks on what it heard
        return x
```

```text
   ┌──────────────── Block ────────────────┐
   │                                        │
   │   x ─► Multi-Head Attention ─► Feed-   │
   │         (communication)       Forward  │ ─► x'
   │                               (compute)│
   │                                        │
   └────────────────────────────────────────┘
        head_size = n_embd // n_head  -> heads stay small, concat back to n_embd

   stack N of these  ->  Block -> Block -> Block -> ...  ->  a deep Transformer
```

`head_size = n_embd // n_head` keeps the bookkeeping automatic: however many heads you
ask for, each is sized so their concatenation comes back out to `n_embd`. Now the whole
model is just *embed → stack of Blocks → final linear → logits*.

### Memory hook

```text
Head            = one self-attention head; register_buffer('tril') (not learned),
                  crop tril[:T,:T]; out = softmax(q·kᵀ/√d, masked) @ v
generate crop   = idx[:, -block_size:]  -- pos-emb only knows 0..block_size-1
lower LR (1e-3) = attention is delicate; loss 2.5 -> 2.4 with one head
multi-head      = run h SMALL heads in parallel (own q/k/v each), CONCAT over
                  channels; 4 heads x 8 dims = 32 = n_embd  (like group conv)
                  -> different heads learn different relationships (subspaces)
feed-forward    = per-token MLP: Linear -> ReLU; tokens TALK in attention,
                  then THINK here; ReLU = the non-linearity that enables real compute
Block           = MultiHeadAttention (communicate) + FeedForward (compute);
                  head_size = n_embd//n_head; stack N blocks -> deep Transformer
the mantra      = "communication followed by computation"
```

### Stacking The Blocks Sequentially

A single `Block` is one round of "talk, then think." A real Transformer is just **that
round repeated**. We drop a few `Block`s into an `nn.Sequential`, which simply pipes the
output of one straight into the next:

```python
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )                                    # <-- 3 blocks, applied in order
        self.lm_head = nn.Linear(n_embd, vocab_size)
```

`nn.Sequential` is just shorthand for "run these in order, feeding each one's output
into the next":

```text
   x ─► Block ─► Block ─► Block ─► x'
        talk/      talk/    talk/
        think      think    think

   round 1: tokens gather + digest a first pass of context
   round 2: ...gather + digest again, now over RICHER representations
   round 3: ...and again -> deeper, more abstract features each time
```

In `forward`, the whole stack is now a single call — replace the one-off attention/MLP
lines with `self.blocks(x)`:

```python
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                              # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb     # (B,T,C)
        x = self.blocks(x)        # (B,T,C)  <- talk+think, three times over
        logits = self.lm_head(x)  # (B,T,vocab_size)
        ...
```

```text
   the full forward, end to end:

   idx ─► token_emb + position_emb ─► [Block ×3] ─► lm_head ─► logits
           "what + where"              "talk/think    "project to
                                        repeatedly"     vocab scores"
```

Stacking deepens the model — but as we'll see next, stacking *too* naively makes deep
nets hard to train, which is exactly what residual connections and layer-norm are about
to fix.
