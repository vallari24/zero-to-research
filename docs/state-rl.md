# State / RL Project Ideas

This note is a shortlist of project directions that are:
- still interesting enough to write up seriously
- not as crowded as plain math-or-code benchmark chasing
- realistic to scope into a portfolio artifact

The goal is not to pick the flashiest topic. The goal is to pick something that
can be finished well, evaluated cleanly, and defended as actual research or
engineering work.

## 1. Text2World: verifier-guided symbolic world models

**One-line idea**

Take a natural-language world description, generate a symbolic world model in
`PDDL`, run a planner on it, then use verifier failures to repair the model.

**What gets built**

- a generator that maps text to `PDDL`
- a verifier that checks syntax, executability, and planning consistency
- a repair loop that fixes only the broken parts
- a planner that measures whether the generated world actually supports solving tasks

**Why it is interesting**

- much cheaper than video or embodied world models
- cleaner than open-ended judge-based evaluation
- still close to the real question: can the model build a usable world representation?

**Why it is portfolio-worthy**

This is easy to demo:
- input a world description
- show generated symbolic rules
- show verifier failure
- show repaired model
- show final plan

**Why it is paper-worthy**

The novelty is not "use an LLM on `Text2World`". The novelty is:
- verifier-guided repair
- planner-aware decoding
- uncertainty-aware abstention
- execution-based reranking

**Best fit if**

You want the best balance of low compute, clean evaluation, and strong writeup potential.

## 2. WorldTest / AutumnBench: exploration-first world-model learning

**One-line idea**

Learn an exploration policy that builds a useful world model before the agent
knows what downstream task it will later be tested on.

**What gets built**

- an exploration policy for reward-free interaction
- a memory or latent model of discovered dynamics
- an evaluation pipeline on derived tasks like planning or causal change prediction

**What is different here**

Most RL setups ask:

```text
maximize task reward
```

This setup asks:

```text
explore first, learn the world, then solve a different but related task
```

**Why it is interesting**

- much less crowded than plain RLVR
- targets transferable understanding, not just task reward hacking
- better fit for serious "world model" claims

**Why it is portfolio-worthy**

Moderate. It is more researchy than flashy. The value is in the experimental
story, not the demo.

**Why it is paper-worthy**

Good questions include:
- which exploration strategy best improves later planning?
- does novelty help less than causal intervention?
- what kind of memory matters for downstream transfer?

**Best fit if**

You want a truer RL-style world-model project and care more about research depth than demo appeal.

## 3. Tool-use RLVR

**One-line idea**

Train an LLM with reinforcement learning from verifiable rewards in a
deterministic tool environment where task success can be checked exactly.

**What gets built**

- an agent that calls tools or acts in a sandboxed environment
- a reward function based on exact task completion
- a `GRPO` or `PPO` training loop
- evaluation on success rate, cost, and robustness

**Why it is interesting**

- teaches actual RLVR mechanics
- less crowded than plain math or code answer checking
- more agentic than one-shot reward tasks

**Why it is portfolio-worthy**

High. It looks like a modern post-training project instead of a static benchmark reproduction.

**Why it is paper-worthy**

Good angles include:
- reward shaping for tool efficiency
- local vs full retry after failure
- verifier design for multi-step tasks
- success vs token/tool budget tradeoff

**Best fit if**

You specifically want to build RLVR skill, not just planning or verification skill.

## 4. Long-horizon planning: uncertainty-aware hierarchical replanning

**One-line idea**

Build an agent that decomposes long tasks into subgoals, tracks uncertainty, and
replans only when needed instead of restarting from scratch.

**What gets built**

- a planner that proposes high-level subgoals
- an executor that carries them out
- an uncertainty or failure detector
- a local repair or replanning policy

**Good benchmark shapes**

- browser or workflow tasks like `WebArena`
- higher-level procedural planning tasks like `WorldPrediction`

**Why it is interesting**

Long tasks fail because:
- plans drift
- one early mistake poisons later steps
- agents either overcommit or replan too often

That makes this a good place to study abstraction, repair, and execution-time decision making.

**Why it is portfolio-worthy**

Very high. Long-horizon agents are easy to explain and visually compelling.

**Why it is paper-worthy**

Good angles include:
- uncertainty-triggered replanning
- local repair vs full replanning
- structured subgoals vs free-form plans
- memory-bounded planning

**Best fit if**

You want the strongest visible demo, and you are willing to take on more engineering complexity.

## 5. Other less crowded project directions

### Feedback-driven continual alignment under memory budget

**Idea**

Keep updating an LLM from a stream of user corrections while preventing forgetting under a fixed replay or parameter budget.

**Why it is interesting**

- realistic product setting
- strong continual learning question
- less crowded than standard RLVR benchmarks

### Formal theorem proving RLVR

**Idea**

Use proof-checking as the verifier and train for proof success instead of just answer matching.

**Why it is interesting**

- stronger research signal than plain math RLVR
- fully verifiable
- cleaner than preference-based RLHF

### Structured extraction or SQL RLVR

**Idea**

Reward exact schema validity or exact execution-result match instead of fuzzy natural-language quality.

**Why it is interesting**

- practical and easy to evaluate
- closer to real deployment tasks
- good entry point into verifier design

## Quick comparison

| Project | Compute | RL content | Paper potential | Portfolio value |
| --- | --- | --- | --- | --- |
| Text2World symbolic world models | Low | Low to medium | High | High |
| WorldTest exploration-first world models | Low to medium | High | High | Medium |
| Tool-use RLVR | Medium | High | High | High |
| Long-horizon replanning | Medium to high | Medium | High | Very high |
| Continual alignment under memory budget | Medium | Medium | High | Medium |

## Recommendation

If the goal is **best overall project**, pick:

**Text2World with verifier-guided symbolic world-model repair**

If the goal is **best project for learning RLVR**, pick:

**tool-use RLVR**

If the goal is **best project for a visibly strong demo**, pick:

**long-horizon replanning**

If the goal is **best research-heavy world-model question**, pick:

**WorldTest / AutumnBench exploration-first learning**

## References

- `Text2World`: https://arxiv.org/abs/2502.13092
- `WorldTest / Benchmarking World-Model Learning`: https://arxiv.org/abs/2510.19788
- `WebArena`: https://proceedings.iclr.cc/paper_files/paper/2024/hash/4410c0711e9154a7a2d26f9b3816d1ef-Abstract-Conference.html
- `Plan-and-Act`: https://proceedings.mlr.press/v267/erdogan25a.html
- `ProAct`: https://arxiv.org/abs/2602.05327
- `WorldPrediction`: https://arxiv.org/abs/2506.04363
- `WorldArena`: https://arxiv.org/abs/2602.08971
