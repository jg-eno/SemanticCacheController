# Semantic Cache Controller

A Python project for simulating semantic caching under changing query workloads. It provides a Gymnasium environment, dataset adapters, deterministic baselines, and offline evaluation.

The current implementation is a simulation. Its results do not establish production answer correctness, serving latency, or monetary savings. Custom PPO-Lagrangian training is available for offline experiments.

## Setup

Use Python 3.13 or later. From the repository root:

```sh
python -m pip install numpy gymnasium datasets sentence-transformers
```

Only NumPy and Gymnasium are needed for the default offline tests. Running the sample or evaluation with the default encoder downloads a sentence-embedding model on first use.

## Quick start

Run the bundled sample workload:

```sh
python main.py
```

Minimal environment creation:

```python
from src.datasets.workload import read
from src.semantic_cache import SemanticCacheEnv

env = SemanticCacheEnv(read("src/datasets/example_workload.jsonl"))
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(1)
env.close()
```

## Evaluation and tests

Compare the deterministic baselines on the example and held-out workloads:

```sh
python -m src.evaluate src/datasets/example_workload.jsonl --held-out src/datasets/example_held_out.jsonl
python -m unittest discover -s src/tests -v
```

The bundled workloads are synthetic examples. Default tests run without downloading datasets or embedding models. Use `python -m src.evaluate --help` and `python -m src.datasets.workload --help` for available commands and configuration.

## Repository structure

```text
.
├── main.py                     # Sample environment entry point
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Dependency lockfile
├── README.md
├── docs/                       # Supporting documents and assets
└── src/
    ├── semantic_cache.py       # Public environment imports
    ├── evaluate.py             # Baseline evaluation CLI
    ├── rl/                     # PPO-Lagrangian training and evaluation
    ├── baseline_models/        # Additional baseline files
    ├── datasets/               # Workload adapters and example data
    ├── env/                    # Gymnasium environment
    ├── models/                 # Cache, embedding, and drift components
    └── tests/                  # Offline automated tests
```

## Data sources

The primary dataset is [SemCacheLMArena](https://huggingface.co/datasets/vCache/SemBenchmarkLmArena), whose dataset card lists Apache-2.0. Optional evaluation sources are [WildChat](https://huggingface.co/datasets/allenai/WildChat) (ODC-BY) and [CacheEval](https://github.com/BudEcosystem/CacheEval) (CC BY-SA 4.0 for data and MIT for scripts, according to its repository). Consult upstream attribution and licensing before redistributing external data. No external dataset is bundled here.

## PPO-Lagrangian training

Install PyTorch in addition to the environment dependencies:

```sh
python -m pip install torch
python -m src.rl.train --env-factory your_adapter:make_env --train-settings train.json --validation-settings validation.json --output /tmp/cache-ppo-run --seed 0
```

The environment factory is supplied explicitly; no existing environment import is assumed. Each settings file is a JSON object passed as keyword arguments to that callable. Registered environments can use `--env-factory gym:YourRegisteredID`. Use separate chronological training and validation traces. Record workload paths/revisions, encoder revision, cache capacity, retrieval threshold, reward settings and other environment options in those files. Training does not read held-out data.

The adapter must preserve the six-feature float32 observation and actions 0/1. Step info must provide `effective_action`, `reuse_status` (`correct`, `incorrect`, `unevaluable`, `not_reused`), and finite `reward_components` including the signed `recompute_cost`; components must sum to the returned reward. For labeled-request coverage, provide boolean `labeled_request` in step info or pass `--train-labels` and `--validation-labels` JSON maps from request ID to source intent or null. These are offline metadata. Evaluation additionally requires `cache_hit`, fractional `cache_occupancy`, and `phase_id` for phase/transition reports.

Experimental constraint defaults are false-hit rate ≤ 0.01 and recomputation rate ≤ 0.5, configured by `--false-limit` and `--recompute-limit`. False-hit rates use evaluable labeled requests; actual recomputations include forced misses. Unevaluable reuse is reported as unknown. By default, feasibility requires at least 32 evaluable labeled requests and full evaluable coverage. When the recomputation constraint is enabled, training utility excludes the explicit logged recomputation penalty, while raw environment reward remains logged.

Use `python -m src.rl.train --help` for timestep, rollout, minibatch, optimizer, discount/GAE, clipping, entropy, multiplier, evaluation and checkpoint options. Limits are experimental, and constraints may be infeasible for a trace. Use a fresh output directory per seed/run. Outputs include `config.json`, `metrics.jsonl`, `costs.jsonl`, periodic checkpoints, `final.pt`, and `selection.json`. `best.pt` exists only if validation satisfies every enabled constraint with sufficient labeled evidence.

## Learned-policy evaluation

Provide `baselines.json` mapping `always_reuse`, `always_recompute`, and `drift_rule` to their existing `module:callable` policies. Each callable accepts only the observation and returns action 0 or 1.

```sh
python -m src.rl.evaluate --checkpoint /tmp/cache-ppo-run/best.pt --env-factory your_adapter:make_env --held-out-settings held_out.json --baselines baselines.json --seeds 0 1 2 --output /tmp/cache-ppo-held-out.json
```

Use `--held-out-labels` if your adapter does not emit labeled-request metadata. Evaluation uses deterministic actions and fresh environments with identical settings for all policies. Reports include utility, label coverage, false-hit/recomputation rates, constraint violations, cache metrics, phase/transition results, multipliers and mean/standard deviation across seeds. Do not use held-out results to select checkpoints or tune settings. Distinct settings and disjoint label maps are checked where available; the factory owner must ensure actual temporal split separation. Repeating a deterministic trace across seeds is not independent workload evidence.

The new policy tests use a small synthetic Gymnasium fixture and need no downloads:

```sh
python -m unittest src.tests.test_ppo_lagrangian -v
```

Training stability, convergence, and constraint satisfaction on real workloads remain experimental; unit tests establish implementation behavior rather than production performance.
