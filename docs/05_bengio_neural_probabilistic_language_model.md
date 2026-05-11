# Bengio Paper: A Neural Probabilistic Language Model

This blog explains the core ideas from:

Yoshua Bengio, Rejean Ducharme, Pascal Vincent, and Christian Jauvin,
*A Neural Probabilistic Language Model*, JMLR 2003.

Paper: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

The goal is not to memorize the paper. The goal is to build a durable mental
model for why embeddings and multilayer perceptrons are useful for language.

## The One-Sentence Idea

Instead of memorizing a giant table of word sequences, learn a vector for each
word and train a neural network to predict the next word from the vectors of
previous words.

In table form:

```text
old view:
previous words -> look up a probability table row -> next-word probabilities

new view:
previous words -> word vectors -> MLP -> next-word probabilities
```

This is the key shift:

```text
from discrete table lookup
to continuous learned geometry
```

That shift is one of the foundations of modern deep learning for language.

## The Problem: Language Has Too Many Possible Sentences

A language model assigns probabilities to sequences of words.

For a sentence:

```text
w1 w2 w3 ... wT
```

we can write its probability as:

```text
P(w1, w2, ..., wT)
  = P(w1) * P(w2 | w1) * P(w3 | w1, w2) * ...
```

So the repeated task is:

```text
given previous words, predict the next word
```

The hard part is that the number of possible contexts explodes.

Suppose:

```text
vocabulary size = 100,000 words
context length  = 10 words
```

Then the number of possible 10-word combinations is roughly:

```text
100,000^10 = 10^50
```

That is not just large. It is impossible to cover with data.

So the model will constantly see new word sequences at test time:

```text
seen during training:
the old doctor opened the door

new at test time:
the young nurse opened the window
```

A good model should understand that these sentences are related. A plain table
does not naturally know that.

## The Old Solution: N-Gram Tables

Before neural language models, a common approach was the n-gram model.

A trigram model predicts the next word from the previous two words:

```text
P(next word | previous two words)
```

Example:

```text
context: "opened the"

possible next words:
door      0.32
window    0.18
box       0.06
...
```

The model is basically a table:

```text
context                    probability row
------------------------------------------------
"opened the"               door: .32, window: .18, ...
"walked into"              room: .24, house: .12, ...
"sat on"                   chair: .20, bed: .15, ...
```

This works well when the exact context appeared many times.

But it struggles when the exact context is rare or unseen:

```text
"the old doctor"     seen
"the young nurse"    not seen
```

Even if the two contexts are meaningfully similar, the table treats them as
mostly unrelated rows.

## The Core Weakness Of Tables

A table has no built-in idea that:

```text
doctor is related to nurse
door is related to window
old is related to young
```

Every context is a separate address.

Visualize it like this:

```text
table world

"the old doctor"   -> row 8172
"the young nurse"  -> row 9914
"a tired surgeon"  -> row 1208

Each row lives separately.
Similarity has to be manually engineered.
```

This creates data sparsity.

Even if the training corpus is huge, most possible sequences never appear.
The model needs a way to transfer knowledge from seen sequences to similar
unseen sequences.

## The New Solution: Learn A Space

Bengio et al. propose that every word should have a learned vector.

Instead of:

```text
doctor -> id 3942
nurse  -> id 8129
door   -> id 1170
```

use:

```text
doctor -> [ 0.12, -0.44,  0.87, ...]
nurse  -> [ 0.10, -0.39,  0.80, ...]
door   -> [-0.51,  0.22,  0.15, ...]
```

The ID only says "which word is this?"

The vector can learn "how does this word behave?"

That is the embedding idea.

## Visualizing Embeddings

Real embeddings may have 30, 60, 100, or many more dimensions. But imagine a
2D version:

```text
                medical / people
                       ^
                       |
              doctor   nurse
                       |
                       |
   object <------------+------------> action
                       |
              door     window
                       |
                       v
                    places / things
```

The exact axes are not hand-labeled in real life. The model discovers useful
directions during training.

The important thing is distance:

```text
nearby vectors -> words that behave similarly in context
far vectors    -> words that behave differently
```

This lets one training sentence influence many similar sentences.

## The Career-Level Intuition

This is the idea that should stick:

```text
Neural networks generalize better when discrete symbols are represented as
continuous vectors and processed by smooth functions.
```

Why?

Because small changes in vector space tend to produce small changes in output.

If:

```text
doctor and nurse have similar vectors
```

then:

```text
"the doctor opened the ..."
```

can help the model with:

```text
"the nurse opened the ..."
```

even if the second context was rare.

## The Model In One Diagram

The model takes previous words, looks up their embeddings, concatenates them,
feeds them through an MLP, and predicts the next word.

```text
previous word ids

  w(t-3)       w(t-2)       w(t-1)
    |            |            |
    v            v            v
 embedding    embedding    embedding
 lookup C     lookup C     lookup C
    |            |            |
    v            v            v
  C(w1)        C(w2)        C(w3)
    \            |            /
     \           |           /
      v          v          v
       concatenate into one vector x
                    |
                    v
              hidden layer
             tanh(d + Hx)
                    |
                    v
       output scores for all words
             b + Wx + U hidden
                    |
                    v
                 softmax
                    |
                    v
       P(next word | previous words)
```

The MLP is not reading words directly. It reads vectors.

That matters.

## The Architecture As Layers

A simplified view:

```text
word ids
  |
  v
embedding table C
  |
  v
concatenated context vector x
  |
  v
linear layer: Hx + d
  |
  v
nonlinearity: tanh
  |
  v
linear output layer: U h + b
  |
  v
softmax probabilities
```

That is an MLP language model.

The embedding table is also learned by backpropagation. It is not a fixed
preprocessing step.

## The Formula

The paper writes the model's output scores like this:

```text
y = b + W x + U tanh(d + H x)
```

Read it piece by piece.

### x

`x` is the concatenation of the context word vectors.

If the model uses 3 previous words and each word vector has 30 numbers:

```text
x has 3 * 30 = 90 numbers
```

So:

```text
x = [embedding(word1), embedding(word2), embedding(word3)]
```

### Hx + d

This is the hidden layer's linear transformation.

```text
hidden pre-activation = Hx + d
```

`H` mixes the input features together. `d` is the hidden bias.

### tanh(d + Hx)

The `tanh` nonlinearity lets the network model interactions.

Without a nonlinearity, stacking linear layers would still be linear. The
nonlinearity is what lets the model learn more flexible patterns.

For example, it can learn something like:

```text
if the context is about people and the previous word suggests an action,
then increase probability of certain verbs or objects
```

Not as a hand-coded rule, but as a learned pattern.

### U tanh(...)

`U` maps hidden features into one score per possible next word.

If the vocabulary has 17,000 words, the output has 17,000 scores.

```text
score for "door"
score for "window"
score for "doctor"
...
```

These scores are also called logits.

### W x

`W x` is an optional direct connection from the embeddings to the output.

It lets the model learn a more direct, almost linear mapping from context
features to next-word scores.

Visualize the two paths:

```text
path 1:
x -> hidden layer -> output

path 2:
x ----------------> output
```

The hidden path gives nonlinear modeling power. The direct path can make
simple associations easier to learn.

### b

`b` is the output bias.

It captures how common each word is before considering the context.

Common words start with higher baseline scores.

## Softmax: Turning Scores Into Probabilities

The network produces raw scores:

```text
door      5.2
window    4.7
doctor    0.1
banana   -2.4
...
```

Softmax turns these into probabilities:

```text
door      0.41
window    0.25
doctor    0.01
banana    0.00
...
```

All probabilities are positive and sum to 1.

That gives:

```text
P(next word = word_i | context)
```

## One Training Example

Imagine the training text contains:

```text
the doctor opened the door
```

At one position, the model might see:

```text
context: the doctor opened the
target:  door
```

Forward pass:

```text
1. Look up embeddings for "the", "doctor", "opened", "the"
2. Concatenate them into x
3. Compute hidden features with tanh(d + Hx)
4. Compute scores for every possible next word
5. Softmax scores into probabilities
6. Look at the probability assigned to "door"
```

If the model gives:

```text
P(door | the doctor opened the) = 0.02
```

the loss is high.

If it gives:

```text
P(door | the doctor opened the) = 0.72
```

the loss is low.

Training pushes the model toward assigning more probability to the real next
word in the data.

## What Backprop Updates

When the prediction is wrong, backprop updates:

```text
1. the output layer weights
2. the hidden layer weights
3. the embeddings of the words in the context
```

That last part is crucial.

If "doctor" appears in contexts similar to "nurse", their embeddings may move
toward similar regions of the vector space.

So the embedding space is shaped by prediction pressure:

```text
words used similarly -> vectors become useful in similar ways
```

The model is not told what "doctor" means. It learns a representation that
helps predict surrounding words.

## Why The Model Generalizes

The model generalizes through two linked ideas.

### 1. Similar words get similar vectors

If two words appear in similar contexts, the training process can give them
similar embeddings.

```text
doctor <-> nurse
city   <-> town
door   <-> window
```

### 2. The MLP is a smooth function

The MLP maps continuous inputs to continuous outputs.

So nearby inputs tend to produce nearby predictions.

Visualize it:

```text
embedding space

sentence A: [the, old, doctor, opened, door]
sentence B: [the, young, nurse, opened, window]

If the word vectors are near each other,
the sentence representations are also near each other.

Near inputs -> similar output probabilities.
```

That is how one training example can help many unseen but related examples.

## What The MLP Learns

The hidden layer learns features of the context.

These features are not named by humans, but you can imagine them as detectors:

```text
feature 17: context seems like a person doing an action
feature 42: context seems like a location phrase
feature 88: context suggests a number may come next
```

This is only an intuition. The real hidden units are distributed and mixed.

But it is a useful way to think:

```text
embedding layer: turns words into useful coordinates
hidden layer: combines coordinates into context features
output layer: turns context features into next-word scores
```

## Why Hidden Units Matter

A model with only embeddings and a linear output can learn useful statistics.
But the hidden layer gives the model a way to combine features nonlinearly.

For example:

```text
"bank" near money-related words
```

and:

```text
"river" in the context
```

should produce a different prediction than:

```text
"bank" near money-related words
```

and:

```text
"loan" in the context
```

The hidden layer gives the model a place to represent combinations.

This is a foundation of MLP intuition:

```text
linear layers mix information
nonlinearities let mixed information interact
depth builds more useful intermediate representations
```

## The Cost: Full Vocabulary Output Is Expensive

The model predicts a probability for every word in the vocabulary.

If:

```text
vocabulary size = 17,000
```

then every training example needs scores for 17,000 possible next words.

That is expensive.

The paper notes that the output layer is the main computational bottleneck.

Visualize the cost:

```text
hidden vector
     |
     v
score word 1
score word 2
score word 3
...
score word 17,000
```

Most of the time, only one word is the correct target, but softmax requires
normalizing against all words.

This is one reason later work explored faster output methods.

## The Paper's Improvement Ideas

The paper does not stop at the basic MLP. It also points toward improvements.

### 1. Break the network into smaller networks

Group words into clusters, then train smaller models.

```text
all words
  |
  +-- people words
  +-- object words
  +-- place words
  +-- function words
```

Smaller networks can be faster and easier to train.

Modern descendant:

```text
class-based models, mixture-of-experts, modular networks
```

### 2. Use a tree over words

Instead of scoring every word directly, predict a path through a tree.

```text
root
 |
 +-- noun-like
 |    |
 |    +-- person
 |    |    |
 |    |    +-- doctor
 |    |    +-- nurse
 |    |
 |    +-- object
 |         |
 |         +-- door
 |         +-- window
 |
 +-- verb-like
      |
      +-- opened
      +-- walked
```

This can reduce computation.

Modern descendant:

```text
hierarchical softmax
```

### 3. Update only a subset of output words

Instead of computing gradients for every possible word, update:

```text
the correct word
+ a small set of likely or informative wrong words
```

Modern descendants:

```text
sampled softmax
negative sampling
noise contrastive estimation
importance sampling
```

The intuition:

```text
You do not always need to compare the target against every word.
Often, comparing against useful negatives is enough.
```

### 4. Add prior knowledge

The paper suggests using external knowledge, such as:

```text
semantic resources
part-of-speech tags
grammar structure
```

This matters because data alone may be inefficient.

Modern descendant:

```text
pretraining objectives, structured supervision, retrieval augmentation,
tool use, knowledge-augmented models
```

### 5. Capture longer context

The MLP uses a fixed window of previous words.

But language often depends on longer context:

```text
sentence-level context
paragraph-level context
document-level context
```

The paper suggests time-delay and recurrent neural networks as ways to reuse
computation and model longer histories.

Modern descendants:

```text
RNN language models
LSTMs
Transformers
long-context models
```

### 6. Handle multiple meanings per word

One weakness of the model is that each word gets one vector.

But many words have multiple meanings:

```text
bank = financial institution
bank = side of a river
```

One fixed vector cannot fully represent both senses.

The paper suggests giving words multiple points in semantic space.

Modern descendant:

```text
contextual embeddings
```

In modern models, the representation of a word depends on the sentence around
it.

## What Makes This Paper Foundational

The paper is important because it combines several ideas that became central:

```text
1. represent words with learned vectors
2. learn those vectors jointly with the prediction model
3. use an MLP to map context vectors to next-word probabilities
4. train by maximizing likelihood
5. evaluate with held-out likelihood or perplexity
6. use distributed representations to generalize beyond exact examples
```

This is not just a language modeling trick. It is a deep learning pattern:

```text
discrete input
  -> learned vector representation
  -> neural network
  -> probability distribution
  -> loss
  -> backprop
```

That pattern appears everywhere.

## The MLP Foundation To Remember

An MLP is a learned function.

For this paper, the function is:

```text
context vectors -> next-word probabilities
```

The pieces have clear jobs:

```text
embedding table:
  learns a useful coordinate system for words

linear layer:
  mixes the input coordinates

nonlinearity:
  lets features interact

hidden layer:
  builds intermediate context features

output layer:
  converts features into one score per word

softmax:
  turns scores into probabilities

loss:
  tells the model how surprised it was

backprop:
  moves all parameters to reduce future surprise
```

This is the durable idea:

```text
MLPs do not memorize only examples.
They learn a smooth function over representations.
```

The quality of the representation matters as much as the network on top.

## A Compact Mental Model

If you remember only one picture, remember this:

```text
word ids
  |
  v
embedding lookup
  |
  v
points in a learned space
  |
  v
MLP combines nearby points smoothly
  |
  v
next-word probability distribution
```

And remember why it works:

```text
Similar words create similar inputs.
Similar inputs create similar predictions.
Similar predictions let the model generalize to sentences it never saw.
```

## How This Connects To Modern Models

Modern language models are much larger and use different architectures, but
many core ideas remain:

```text
Bengio et al. 2003                 Modern language models
---------------------------------------------------------------
word embeddings                    token embeddings
fixed context window               long context windows
MLP over concatenated embeddings    attention + MLP blocks
softmax next-word prediction        softmax next-token prediction
maximum likelihood training         maximum likelihood pretraining
learned distributed representation learned contextual representation
```

The biggest change is context.

The Bengio model gives each word one learned vector and uses a fixed window.
Modern transformer models compute context-dependent representations at every
position.

But the spine is still recognizable:

```text
tokens -> vectors -> neural computation -> logits -> softmax -> loss
```

## Final Takeaway

The paper's lasting contribution is not just a specific architecture.

It is the worldview:

```text
Language is too sparse for giant tables.
Represent words as learned vectors.
Use a neural network to learn a smooth probability function over those vectors.
Train the whole system together.
Let similarity in representation space drive generalization.
```

That idea is worth carrying for the rest of your machine learning career.
