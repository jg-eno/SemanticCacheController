"""Deterministic policies and simulation-only reports."""
import argparse
from collections import Counter
import json
import numpy as np
from .env.semantic_cache_env import SemanticCacheEnv, REUSE, RECOMPUTE, FEATURE_NAMES
from .models.semantic_cache import SentenceEncoder, cosine
from .datasets.workload import read, generate


def always_reuse(observation):
    return REUSE if observation[1] < 1 else RECOMPUTE


def always_recompute(observation):
    return RECOMPUTE


def drift_rule(observation, shift_threshold=0.1, novelty_threshold=0.075):
    return REUSE if observation[1] < 1 and observation[0] <= shift_threshold and observation[1] <= novelty_threshold else RECOMPUTE


POLICIES = {'always_reuse': always_reuse, 'always_recompute': always_recompute, 'drift_rule': drift_rule}


def evaluate(workload, encoder, policy, seed=0, transition_window=3, **config):
    if transition_window < 1:
        raise ValueError('transition_window must be positive')
    env = SemanticCacheEnv(workload, encoder=encoder, **config)
    obs, _ = env.reset(seed=seed)
    records, rewards = [], []
    while True:
        obs, reward, ended, truncated, info = env.step(policy(obs.copy()))
        records.append(info)
        rewards.append(reward)
        if ended or truncated:
            break
    n = len(records)
    statuses = Counter(r['reuse_status'] for r in records)
    transitions = []
    for i in range(1, n):
        if records[i]['phase_id'] != records[i - 1]['phase_id']:
            before = records[max(0, i-transition_window):i]
            after = records[i:min(n, i+transition_window)]
            transitions.append(dict(step=i, phase_id=records[i]['phase_id'],
                mean_shift_before=float(np.mean([r['query_drift_features'][FEATURE_NAMES[0]] for r in before])),
                mean_shift_after=float(np.mean([r['query_drift_features'][FEATURE_NAMES[0]] for r in after]))))
    return dict(total_reward=sum(rewards), mean_reward=sum(rewards)/n,
                cache_hit_rate=sum(r['cache_hit'] for r in records)/n,
                cache_miss_rate=sum(r['cache_miss'] for r in records)/n,
                correct_reuse_count=statuses['correct'], incorrect_reuse_count=statuses['incorrect'],
                unevaluable_reuse_count=statuses['unevaluable'], reuse_count=env.reuse_count,
                recompute_count=env.recompute_count,
                forced_recomputation_count=sum(r['forced_recomputation'] for r in records),
                average_cache_occupancy=float(np.mean([r['cache_occupancy'] for r in records])),
                phase_transitions=transitions)


def compare(workload, encoder, seed=0, **config):
    return {name: evaluate(workload, encoder, policy, seed, **config) for name, policy in POLICIES.items()}


def cacheeval(rows, encoder, threshold=0.85):
    """Isolated text pairs; score binary labels directly, never convert to intent IDs."""
    counts = Counter()
    for row in rows:
        if row.get('binary_label') not in {'HIT', 'MISS'}:
            raise ValueError('CacheEval requires public binary HIT/MISS labels')
        hit = cosine(encoder.encode(row['query_a']), encoder.encode(row['query_b'])) >= threshold
        expected = row['binary_label'] == 'HIT'
        counts['tp' if hit and expected else 'fp' if hit else 'fn' if expected else 'tn'] += 1
    negatives = counts['fp'] + counts['tn']
    return dict(counts, false_hit_rate=counts['fp']/negatives if negatives else None,
                limitation='Text-only retrieval: context, tools and system prompts are not encoded; binary labels retain those distinctions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workload')
    parser.add_argument('--held-out', help='Separate temporal workload; never used for tuning')
    parser.add_argument('--source-rows', action='store_true', help='Input is normalized labeled source rows; resample per seed')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--capacity', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--model-revision')
    parser.add_argument('--drift-window', type=int, default=16)
    parser.add_argument('--projection-count', type=int, default=8)
    parser.add_argument('--histogram-bins', type=int, default=8)
    parser.add_argument('--projection-seed', type=int, default=0)
    parser.add_argument('--cacheeval', help='Separate CacheEval JSONL pair benchmark')
    args = parser.parse_args()
    encoder = SentenceEncoder(args.model, args.model_revision)
    config = dict(capacity=args.capacity, threshold=args.threshold, drift_window=args.drift_window,
                  projection_count=args.projection_count, histogram_bins=args.histogram_bins,
                  projection_seed=args.projection_seed)
    report = {}
    for seed in args.seeds:
        rows = read(args.workload)
        if args.source_rows:
            rows = generate(rows, seed=seed)
        report[str(seed)] = compare(rows, encoder, seed, **config)
    if args.held_out:
        report['held_out'] = compare(read(args.held_out), encoder, **config)
    if args.cacheeval:
        report['cacheeval'] = cacheeval(read(args.cacheeval), encoder, args.threshold)
    print(json.dumps(report, indent=2))
