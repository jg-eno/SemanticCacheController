"""CUCB-SC + Reverse Greedy on the standard benchmark dataset jg-eno/SemanticCacheWorkload.

Learning uses ONLY the training split; evaluation uses ONLY the test split.

Protocol (this is NOT the SemanticCacheEnv protocol; see README):
  1. Q = distinct training query strings. Offline dataset D = training split, one
     arrival per row, every cost observed with C_t = 0.2 (the benchmark's uniform
     recompute_cost; not a measured serving cost).
  2. e(q) = all-MiniLM-L6-v2; d = Euclidean distance divided by the largest
     pairwise TRAINING distance (min-max to [0,1] on Q, as in evaluate.py).
  3. CUCB-SC (Alg. 2) -> Reverse_Greedy (Alg. 1) -> fixed cache M, |M| = k.
  4. Replay the standard test sequence (seed-0 shuffle, origin_id -> intent_id,
     request ids s0-r{step}) against the fixed M. Nothing is admitted or evicted.
     Rule (1): reuse s(q_t, M) iff d(q_t, M) < c(q_t) = 0.2; otherwise recompute.
     A reuse is "correct" iff the cached entry's origin_id equals the request's.

Reported
  - the benchmark's five metrics (src/benchmark.py definitions on GitHub main
    cd718a6), with its default rewards: +1.0 correct, -2.0 incorrect, -0.2 recompute
  - paper loss Eq. (2) l(M; p_test, c = 0.2, d), reported separately

Run from anywhere (all paths are relative to the repository root):
    python src/baseline_models/cucb_sc_reverse_greedy/evaluate_semantic_cache_workload.py
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

# Multi-threaded torch crashed (segfault) while embedding on Windows; one thread is
# deterministic and fast enough for 351 queries. Must be set before torch is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from cache import distance_matrix, loss
from cucb_sc import cucb_sc
from embed import MODEL, embed_queries
from workload_jsonl import read_jsonl

DATASET = "jg-eno/SemanticCacheWorkload"
REVISION = "0be7e5981d640493849471113ebbf4706d1f88c2"
SPLIT_FILES = {"train": "training_workload.jsonl", "test": "test_workload.jsonl"}
RECOMPUTE_COST = 0.2                      # benchmark Rewards.recompute_cost
REWARDS = {"correct": 1.0, "incorrect": -2.0, "not_reused": -RECOMPUTE_COST}
SHUFFLE_SEED = 0                          # standard prepare_semantic_cache_workload seed
REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_relative(path):
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def download_splits(data_dir):
    """Fetch both splits at the pinned revision (skipped if already present)."""
    paths = {split: os.path.join(data_dir, f) for split, f in SPLIT_FILES.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        from huggingface_hub import hf_hub_download
        for f in SPLIT_FILES.values():
            hf_hub_download(DATASET, f, repo_type="dataset", revision=REVISION, local_dir=data_dir)
    return paths


def standard_test_sequence(rows):
    """Same mapping as src/datasets/prepare_semantic_cache_workload.py:convert(seed=0)."""
    rows = list(rows)
    random.Random(SHUFFLE_SEED).shuffle(rows)
    return [dict(request_id=f"s{SHUFFLE_SEED}-r{step}", intent_id=str(row["origin_id"]),
                 query=row["query"], topic_id=None, phase_id=0, step=step)
            for step, row in enumerate(rows)]


def replay(cache, test_dist, test_intents, cache_intents):
    """Score each test request against the fixed cache with rule (1)."""
    d_t = test_dist[:, cache].min(axis=1)                       # d(q_t, M)
    served_by = np.array(cache)[test_dist[:, cache].argmin(axis=1)]  # s(q_t, M)
    records = []
    for t, intent in enumerate(test_intents):
        if d_t[t] < RECOMPUTE_COST:
            status = "correct" if cache_intents[served_by[t]] == intent else "incorrect"
        else:
            status = "not_reused"
        records.append({"status": status, "d": float(d_t[t]), "served_by": int(served_by[t])})
    return records


def benchmark_metrics(records):
    """src/benchmark.py:summarize. Every request is labeled, so all are evaluable."""
    n = len(records)
    incorrect = sum(r["status"] == "incorrect" for r in records)
    recomputed = sum(r["status"] == "not_reused" for r in records)
    return {
        "requests": n,
        "evaluable_labeled_requests": n,
        "false_hit_rate": incorrect / n,
        "recomputation_rate": recomputed / n,
        "mean_environment_reward": sum(REWARDS[r["status"]] for r in records) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "semantic_cache_workload"))
    parser.add_argument("--k", type=int, nargs="+", default=[32], help="cache capacities (32 = benchmark default)")
    parser.add_argument("--delta", type=float, default=0.05, help="confidence parameter of Algorithm 2")
    parser.add_argument("--out", default=str(REPO_ROOT / "benchmark" / "results_cucb_sc.json"))
    args = parser.parse_args()

    os.makedirs(args.data, exist_ok=True)
    paths = download_splits(args.data)
    train_rows, test_rows = read_jsonl(paths["train"]), read_jsonl(paths["test"])

    # ---- learning: training split only ----
    texts = list(dict.fromkeys(r["query"] for r in train_rows))       # Q
    index = {t: i for i, t in enumerate(texts)}
    origin_of = {r["query"]: str(r["origin_id"]) for r in train_rows}
    cache_intents = [origin_of[t] for t in texts]
    train_ids = np.array([index[r["query"]] for r in train_rows])
    train_costs = [RECOMPUTE_COST] * len(train_ids)                   # C_t, uniform benchmark cost

    train_emb = embed_queries(texts, os.path.join(args.data, "embeddings_train.npz"))
    raw_train = distance_matrix(train_emb, normalize=False)
    scale = raw_train.max()                                           # training-only constant
    dist = raw_train / scale

    # ---- evaluation: test split only ----
    test = standard_test_sequence(test_rows)
    test_texts = [r["query"] for r in test]
    test_intents = [r["intent_id"] for r in test]
    test_emb = embed_queries(test_texts, os.path.join(args.data, "embeddings_test.npz"))
    test_dist = np.linalg.norm(test_emb[:, None, :] - train_emb[None, :, :], axis=2) / scale

    overlap_queries = len(set(test_texts) & set(texts))
    overlap_intents = len(set(test_intents) & set(cache_intents))
    print(f"train: n={len(train_ids)} requests, m={len(texts)} distinct queries, "
          f"{len(set(cache_intents))} origin_ids")
    print(f"test : {len(test)} requests, {len(set(test_intents))} origin_ids; "
          f"shared queries with train = {overlap_queries}, shared origin_ids = {overlap_intents}")
    print(f"test distance to nearest TRAIN query: min {test_dist.min(axis=1).min():.4f}, "
          f"median {np.median(test_dist.min(axis=1)):.4f}  (reuse needs < {RECOMPUTE_COST})\n")

    p_test = np.full(len(test), 1 / len(test))                        # each test query arrives once
    results = []
    for k in args.k:
        learned = cucb_sc(train_ids, train_costs, dist, k, args.delta)
        cache = learned["cache"]
        records = replay(cache, test_dist, test_intents, cache_intents)
        d_t = np.array([r["d"] for r in records])
        results.append({
            "k": k,
            **benchmark_metrics(records),
            "hits": sum(r["status"] != "not_reused" for r in records),
            "paper_loss_test": float(np.sum(p_test * np.minimum(RECOMPUTE_COST, d_t))),
            "paper_loss_train_selection_objective": loss(cache, learned["p_hat"], learned["c_bar"], dist),
            "c_bar": float(learned["c_bar"][0]),
            "cache": [{"train_query_id": q, "origin_id": cache_intents[q], "query": texts[q]} for q in cache],
            "per_request": [{"request_id": r["request_id"], "intent_id": r["intent_id"], **rec,
                             "served_by_origin_id": cache_intents[rec["served_by"]]}
                            for r, rec in zip(test, records)],
        })

    print(f"{'k':>3} {'requests':>8} {'evaluable':>9} {'false_hit':>9} {'recompute':>9} "
          f"{'mean_reward':>11} {'hits':>5} {'paper_loss':>10}")
    for r in results:
        print(f"{r['k']:>3} {r['requests']:>8} {r['evaluable_labeled_requests']:>9} {r['false_hit_rate']:>9.4f} "
              f"{r['recomputation_rate']:>9.4f} {r['mean_environment_reward']:>11.4f} {r['hits']:>5} "
              f"{r['paper_loss_test']:>10.4f}")
    print(f"\nCUCB-SC: N_c(q)=1 for every q, c_hat=0.2, c_bar={results[0]['c_bar']:.4f} (delta={args.delta})")

    report = {
        "method": "CUCB-SC (Alg. 2) + Reverse_Greedy (Alg. 1), arXiv:2508.07675; offline, fixed cache",
        "dataset": {"name": DATASET, "revision": REVISION,
                    "files": {s: {"path": repo_relative(p), "sha256": sha256(p)} for s, p in paths.items()}},
        "learning_data": "training split only", "evaluation_data": "test split only",
        "test_sequence": "random.Random(0).shuffle, origin_id -> intent_id, request_id s0-r{step}",
        "synthetic_data_used": False,
        "cost_model": "uniform C_t = 0.2 = benchmark recompute_cost (not a measured serving cost)",
        "rewards": REWARDS, "embedding_model": MODEL,
        "distance": "Euclidean / max pairwise training distance (training-only constant)",
        "distance_scale": float(scale), "delta": args.delta,
        "train_test_shared_queries": overlap_queries, "train_test_shared_origin_ids": overlap_intents,
        "protocol_difference": ("Not the SemanticCacheEnv online protocol: the cache is selected offline "
                                "from the training split and stays fixed during the test replay (no "
                                "admission/eviction); reuse uses paper rule (1) d(q,M) < 0.2 with normalised "
                                "Euclidean distance, not the env's cosine threshold."),
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {repo_relative(args.out)}")


if __name__ == "__main__":
    main()
