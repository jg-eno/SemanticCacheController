"""Load the REAL workload (data/real/) produced by prepare_real_data.py.

Paper mapping
  - Query set Q: every distinct `query` string in the train and held-out
    workloads gets one id 0..m-1 (the paper's finite query set Q).
  - Offline dataset D = {(q_t, C_t)}: the train workload rows, with C_t taken
    from costs.jsonl (`serving_cost`, the normalised recorded latency).
    A missing cost would be stored as None (C_t = empty); with this data every
    cost is observed.
"""

import json

import numpy as np


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_costs(path):
    return {row["request_id"]: row for row in read_jsonl(path)}


def build_query_set(*workloads):
    """Distinct query strings (first-appearance order) and each query's intent."""
    texts, intent_of = [], {}
    for workload in workloads:
        for row in workload:
            if row["query"] not in intent_of:
                texts.append(row["query"])
                intent_of[row["query"]] = row["intent_id"]
            elif intent_of[row["query"]] != row["intent_id"]:
                raise ValueError(f"query appears with two intents: {row['query'][:60]!r}")
    return texts, [intent_of[t] for t in texts]


def to_log(workload, costs, query_index):
    """Workload rows -> (query ids, costs C_t, intents). Fails on any missing cost record."""
    missing = [row["request_id"] for row in workload if row["request_id"] not in costs]
    if missing:
        raise ValueError(f"{len(missing)} requests have no cost record, e.g. {missing[:3]}")
    ids = np.array([query_index[row["query"]] for row in workload])
    observed = [costs[row["request_id"]]["serving_cost"] for row in workload]
    intents = [row["intent_id"] for row in workload]
    return ids, observed, intents
