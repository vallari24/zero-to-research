# Additional RL Resources for RLHF and LLM Tuning

If the goal is not general robotics or control, but instead RL for `LLM`
post-training, it helps to watch RL material selectively.

This note is a filtered watch order for the following RL lecture series:

- RL lecture playlist: https://www.youtube.com/watch?v=SupFHGbytvA&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps

The idea is simple:

- do not try to learn all of RL evenly
- prioritize the parts that map directly to `SFT`, `reward modeling`, and
  `PPO`-style tuning

## Phase 1: Must Watch

These are the highest-ROI topics if the end goal is `RLHF` or `LLM` tuning.

### Imitation Learning / Behavioral Cloning

This maps directly to:

- supervised fine-tuning (`SFT`)
- the idea of copying good behavior from demonstration data
- the first step before reward optimization

If you understand behavioral cloning, you already understand the basic shape
of:

```text
prompt -> desired response
```

as a supervised learning problem.

### Policy Gradients

This is one of the most important conceptual bridges into `RLHF`.

Why it matters:

- `PPO` is built on policy-gradient ideas
- it teaches you what it means to update a policy using reward
- it gives the basic logic behind optimizing sampled actions rather than fixed
  labels

For LLMs, the policy is the model that generates tokens. Policy-gradient
methods are the foundation for turning scalar reward into model updates.

### Actor-Critic

This is where `PPO` starts to make more sense conceptually.

Why it matters:

- the `actor` is the policy that generates actions
- the `critic` estimates how good those actions are
- modern RLHF-style methods often inherit this structure directly or
  conceptually

If policy gradients are the foundation, actor-critic is the practical bridge.

## Phase 2: Very Relevant for LLMs

These are not always the first topics people learn, but for `LLM`
post-training they are highly relevant.

### Offline RL

This is especially important.

A lot of `LLM` training is effectively:

- train on logged data
- learn from past trajectories
- improve without interacting freely with the world in the online RL sense

That makes offline RL much closer to real `LLM` training pipelines than many
robotics-style RL setups.

### Exploration vs Exploitation

This helps build intuition for:

- diversity vs greediness
- sampling behavior
- why always taking the top-probability action can be limiting

For language models, this connects naturally to:

- decoding
- response diversity
- trying good alternatives without collapsing to repetition

### Inverse RL

This is one of the closest conceptual cousins to:

- reward modeling
- preference learning

Inverse RL asks:

```text
if I observe good behavior, what reward function might explain it?
```

That is not the same as modern preference modeling, but the intuition is very
close and worth having.

## Phase 3: Optional

These are useful if you want to go deeper into RL research, but they are not
the highest-priority topics for `RLHF`.

### Model-Based RL

Useful for:

- planning
- learned dynamics
- control-heavy applications

This is good background, but less central for understanding the standard
`SFT -> reward model -> PPO` pipeline.

### Meta-Learning

Interesting, but not the first thing to study for `LLM` post-training.

This tends to be more useful if you want to go deeper into:

- adaptation
- fast transfer
- broader RL research topics

## Suggested Watch Order

If you want the shortest high-value path, use this order:

1. `Imitation Learning / Behavioral Cloning`
2. `Policy Gradients`
3. `Actor-Critic`
4. `Offline RL`
5. `Exploration vs Exploitation`
6. `Inverse RL`

Then only go into `Model-Based RL` and `Meta-Learning` if you want more depth.

## What This Gives You

After Phase 1 and Phase 2, you should have the right conceptual foundation for:

- why `SFT` looks like imitation learning
- why `RLHF` needs policy optimization rather than pure supervised labels
- why `PPO` sits naturally in the actor-critic family
- why reward models and preference learning feel close to inverse RL
- why offline data matters so much in real `LLM` pipelines
