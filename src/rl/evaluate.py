"""Deterministic split evaluation through environment factories and baseline callables."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .config import make_env, load_factory, observation, seed_all, settings_file
from .policy import PPO
from .rollout import signals


def summarize(records, config):
    n = len(records)
    if not n:
        raise ValueError('Cannot evaluate an empty episode')
    evaluable = sum(r['evaluable'] for r in records)
    false_hits = sum(r['status'] == 'incorrect' for r in records)
    false_rate = false_hits/evaluable if evaluable else None
    recompute_rate = sum(r['recompute_cost'] for r in records)/n
    sufficient = evaluable >= config.min_labeled_steps and evaluable/n >= config.min_labeled_fraction
    return dict(requests=n, cumulative_task_utility=sum(r['utility'] for r in records),
                mean_reward=sum(r['utility'] for r in records)/n,
                cumulative_environment_reward=sum(r['environment_reward'] for r in records),
                false_hit_rate=false_rate, labeled_requests=sum(r['labeled_request'] for r in records),
                evaluable_labeled_requests=evaluable, unevaluable_requests=n-evaluable,
                labeled_reuse_decisions=sum(r['status'] in ('correct', 'incorrect') for r in records),
                unevaluable_reuse_decisions=sum(r['status'] == 'unevaluable' for r in records),
                correct_reuse_count=sum(r['status'] == 'correct' for r in records),
                incorrect_reuse_count=false_hits, recomputation_rate=recompute_rate,
                reuse_count=sum(r['status'] != 'not_reused' for r in records),
                recompute_count=sum(r['recompute_cost'] for r in records),
                cache_hit_rate=sum(r['cache_hit'] for r in records)/n,
                cache_miss_rate=sum(not r['cache_hit'] for r in records)/n,
                average_cache_occupancy=float(np.mean([r['occupancy'] for r in records])),
                forced_recomputation_count=sum(r['forced'] for r in records),
                false_constraint_satisfied=(false_rate <= config.false_limit if sufficient else None)
                    if config.false_constraint else None,
                recompute_constraint_satisfied=recompute_rate <= config.recompute_limit if config.recompute_constraint else None,
                false_constraint_violation=max(0, false_rate-config.false_limit) if false_rate is not None else None,
                recompute_constraint_violation=max(0, recompute_rate-config.recompute_limit),
                sufficient_labeled_data=sufficient)


def evaluate_episode(factory, settings, policy, config, seed=0, labels=None, transition_window=3):
    env = make_env(factory, settings, seed)
    records = []
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(config.max_episode_steps):
            action = policy(observation(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            observation(obs)
            signal = signals(reward, info, config, labels)
            if 'cache_hit' not in info or 'cache_occupancy' not in info:
                raise ValueError('Evaluation requires cache_hit and cache_occupancy in step info')
            occupancy = float(info['cache_occupancy'])
            if not np.isfinite(occupancy) or not 0 <= occupancy <= 1:
                raise ValueError('Cache occupancy must be a finite fraction')
            signal.update(cache_hit=bool(info['cache_hit']), occupancy=occupancy, phase=info.get('phase_id'))
            records.append(signal)
            if terminated or truncated:
                break
        else:
            raise ValueError('Evaluation episode exceeded max_episode_steps')
    finally:
        env.close()
    overall = summarize(records, config)
    if config.false_constraint and not overall['evaluable_labeled_requests']:
        raise ValueError('False-hit constraint enabled but evaluation has no valid labeled costs')
    phases = {}
    for r in records:
        if r['phase'] is not None:
            phases.setdefault(str(r['phase']), []).append(r)
    transitions = []
    for i in range(1, len(records)):
        if records[i]['phase'] is not None and records[i]['phase'] != records[i-1]['phase']:
            # During is the first post-boundary window; after is the following window.
            windows = {'before': records[max(0, i-transition_window):i],
                       'during': records[i:i+transition_window],
                       'after': records[i+transition_window:i+2*transition_window]}
            transitions.append(dict(step=i, phase_id=records[i]['phase'],
                **{name: summarize(rows, config) if rows else None for name, rows in windows.items()}))
    return dict(overall=overall, phases={p: summarize(rows, config) for p, rows in phases.items()},
                transitions=transitions, constraint_limits=dict(false_hit=config.false_limit, recomputation=config.recompute_limit))


def aggregate(reports):
    summary = {}
    for key in reports[0]['overall']:
        values = [r['overall'][key] for r in reports]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            summary[key] = dict(mean=float(np.mean(values)), std=float(np.std(values)))
    return summary


def compare(factory, settings, agent, baselines, seeds, labels=None):
    policies = {'ppo_lagrangian': lambda obs: agent.model.action(obs, deterministic=True)[0]}
    for name, spec in baselines.items():
        if name in policies:
            raise ValueError('Baseline names must not shadow ppo_lagrangian')
        policies[name] = load_factory(spec) if isinstance(spec, str) else spec
    results = {}
    for name, policy in policies.items():
        reports = []
        for seed in seeds:
            seed_all(seed)
            reports.append(evaluate_episode(factory, settings, policy, agent.config, seed, labels))
        results[name] = dict(runs=reports, summary=aggregate(reports))
    return results


def load_labels(path):
    if path is None:
        return None
    labels = json.loads(Path(path).read_text())
    if not isinstance(labels, dict):
        raise ValueError('Labels must be a JSON object: request ID -> source intent or null')
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--env-factory', required=True, help='module:callable or gym:RegisteredID')
    parser.add_argument('--held-out-settings', required=True)
    parser.add_argument('--held-out-labels')
    parser.add_argument('--baselines', required=True, help='JSON mapping exactly three baseline names to module:callable')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        raise ValueError('Evaluation seeds must be nonnegative')
    agent, metadata = PPO.load(args.checkpoint)
    baselines = settings_file(args.baselines)
    if set(baselines) != {'always_reuse', 'always_recompute', 'drift_rule'}:
        raise ValueError('Provide the three required deterministic baseline callables')
    held_out = settings_file(args.held_out_settings)
    held_out_labels = load_labels(args.held_out_labels)
    for split in ('train_labels', 'validation_labels'):
        if held_out_labels is not None and metadata.get(split) is not None and set(held_out_labels) & set(metadata[split]):
            raise ValueError('Held-out request-ID labels overlap a training or validation split')
    if held_out in [metadata.get('train_settings'), metadata.get('validation_settings')]:
        raise ValueError('Held-out settings must differ from training and validation')
    report = compare(args.env_factory, held_out, agent, baselines, args.seeds, held_out_labels)
    result = dict(results=report, seeds=args.seeds, settings=held_out, factory=args.env_factory,
                  multipliers=agent.multipliers.values.tolist(), multiplier_history=metadata.get('multiplier_history', []),
                  training_config=asdict(agent.config), checkpoint=args.checkpoint,
                  limitation='Offline simulation; seed variability on a fixed deterministic trace is not independent workload evidence.')
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
