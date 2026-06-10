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
