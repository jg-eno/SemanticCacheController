# Stage 4: decision-rule experiment

This folder is specifically for Stage 4 of the experiment plan: comparing the
four deterministic decision rules that PPO should beat. It is not a general
archive for every evaluation or test in the project. `results.json` is the
current report for always reuse, always recompute, similarity threshold, and
drift rule.

## Run the evaluation

From the project root (`SemanticCacheController`), run:

```bash
.venv/bin/python -m src.evaluate --output experimental_results/stage_4_decision_rules/results.json
```

This uses the bundled example workload by default. To evaluate another JSONL
workload, pass it explicitly:

```bash
.venv/bin/python -m src.evaluate \
  --workload path/to/workload.jsonl \
  --output experimental_results/stage_4_decision_rules/my-run.json
```

The similarity rule uses a cosine decision threshold of `0.90` by default.
Set `--decision-threshold` to change it. Use a distinct output filename to
keep previous runs.

## Reading the report

Each top-level key names a decision rule. Its report includes reward, cache
hits, correct and incorrect reuse, and shared evaluation metrics. The bundled
workload is a synthetic example; its results illustrate behavior on that
trace and should not be treated as production performance or as results on a
held-out real workload. Cost, staleness, and controller-memory fields remain
unavailable without additional instrumentation.
