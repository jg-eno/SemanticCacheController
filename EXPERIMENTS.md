| Stage | Experiment | What it answers |
|---|---|---|
| 1 | Sanity checks: no cache, always recompute, always reuse | Do costs, hits, and incorrect reuse behave as expected? |
| 2 | Retrieval sweep: vary embedding model and similarity threshold | How much correct reuse is possible before changing the controller? |
| 3 | Eviction comparison: FIFO vs LRU vs LFU, with the same reuse rule and capacity | Does a different replacement policy help under eviction pressure? |
| 4 | Decision rules: always reuse, always recompute, similarity threshold, current drift rule | What does PPO need to beat? |
| 5 | PPO vs rules: same trace, encoder, threshold, capacity, and eviction policy | Does learning improve the quality–cost tradeoff? |
| 6 | Feature ablations: remove workload drift, novelty, or recent history features one at a time | Which observations actually help PPO? |
| 7 | Robustness: vary capacity, workload shift size, threshold, and reward/constraint settings | Does the result hold beyond one configuration? |
| 8 | Held-out evaluation: test once on a later, distinct trace | Does the chosen policy generalize? |

Stage 4 runs through `python -m src.evaluate`. The similarity-threshold
decision rule uses a default cosine threshold of `0.90`, independent of the
cache retrieval threshold; override it with `--decision-threshold`. The
resulting per-policy reports include the same hit, correctness, reward, and
timing metrics so the four rules can be compared on the same workload and
environment configuration.
