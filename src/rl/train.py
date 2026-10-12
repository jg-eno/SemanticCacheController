"""Trace-driven PPO-Lagrangian training; held-out data is never opened here."""
import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import platform
import random
import numpy as np
import torch
import gymnasium
from .config import Config, FEATURE_NAMES, make_env, seed_all, settings_file
from .policy import PPO
from .rollout import Collector
from .evaluate import evaluate_episode, load_labels


def feasible(report, config):
    metrics = report['overall']
    return ((not config.false_constraint or metrics['false_constraint_satisfied'] is True) and
            (not config.recompute_constraint or metrics['recompute_constraint_satisfied'] is True))


def train(factory, train_settings, validation_settings, output, config=None, train_labels=None, validation_labels=None):
    config = (config or Config()).validate()
    if train_labels is not None and validation_labels is not None and set(train_labels) & set(validation_labels):
        raise ValueError('Training and validation request-ID label maps must be disjoint')
    if train_settings == validation_settings:
        raise ValueError('Use distinct chronological training and validation split settings')
    seed_all(config.seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Output directory must be empty to avoid mixing runs')
    metadata = dict(factory=factory if isinstance(factory, str) else factory.__module__ + ':' + factory.__name__,
                    train_settings=train_settings, validation_settings=validation_settings,
                    observation_features=list(FEATURE_NAMES),
                    reward_mode='environment reward minus signed recompute_cost' if config.recompute_constraint else 'environment reward',
                    versions=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__), gymnasium=gymnasium.__version__),
                    train_labels=train_labels, validation_labels=validation_labels,
                    split_note='Factory settings must preserve chronological splits; distinct settings alone cannot prove trace disjointness.',
                    multiplier_history=[], best_validation=None)
    (output/'config.json').write_text(json.dumps(dict(config=asdict(config), metadata=metadata), indent=2, allow_nan=False))
    agent = PPO(config)
    env = make_env(factory, train_settings, config.seed)
    steps, best_utility = 0, -float('inf')
    next_eval, next_checkpoint, next_log = config.eval_frequency, config.checkpoint_frequency, config.log_frequency
    logs = output/'metrics.jsonl'
    cost_log = output/'costs.jsonl'
    try:
        collector = Collector(env, agent, config, train_labels)
        while steps < config.total_steps:
            batch, records, episodes = collector.collect(min(config.rollout_length, config.total_steps-steps))
            steps += len(records)
            with cost_log.open('a') as file:
                for offset, signal in enumerate(records):
                    file.write(json.dumps(dict(timestep=steps-len(records)+offset, **signal), allow_nan=False)+'\n')
            metrics = agent.update(batch)
            known = sum(r['evaluable'] for r in records)
            false_hits = sum(r['status'] == 'incorrect' for r in records)
            recomputes = sum(r['recompute_cost'] for r in records)
            rates = agent.multipliers.update(false_hits, known, recomputes, len(records))
            dual = dict(steps=steps, false_hit_rate=rates[0], recomputation_rate=rates[1],
                        false_hit_count=false_hits, recompute_count=recomputes,
                        evaluable_labeled_requests=known, labeled_requests=sum(r['labeled_request'] for r in records),
                        unevaluable_requests=len(records)-known,
                        unevaluable_reuse_decisions=sum(r['status'] == 'unevaluable' for r in records),
                        false_multiplier=float(agent.multipliers.values[0]), recompute_multiplier=float(agent.multipliers.values[1]),
                        false_constraint_violation=max(0, rates[0]-config.false_limit) if rates[0] is not None else None,
                        recompute_constraint_violation=max(0, rates[1]-config.recompute_limit))
            metadata['multiplier_history'].append(dual)
            metadata['steps'] = steps
            # Always preserve rate/dual estimates; loss/episode reporting has its own interval.
            record = dict(type='rollout', **dual, rollout_reward=sum(r['utility'] for r in records))
            if steps >= next_log or steps == config.total_steps:
                record.update(metrics, episodes=episodes)
                next_log = (steps//config.log_frequency+1)*config.log_frequency
            else:
                record['episodes'] = episodes
            with logs.open('a') as file:
                file.write(json.dumps(record, allow_nan=False)+'\n')
            if steps >= next_eval or steps == config.total_steps:
                # Isolate evaluation RNG state so validation frequency does not change training samples.
                python_rng = random.getstate()
                numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state()
                try:
                    report = evaluate_episode(factory, validation_settings,
                        lambda obs: agent.model.action(obs, deterministic=True)[0], config, config.seed, validation_labels)
                finally:
                    random.setstate(python_rng)
                    np.random.set_state(numpy_rng)
                    torch.set_rng_state(torch_rng)
                utility = report['overall']['cumulative_task_utility']
                with logs.open('a') as file:
                    file.write(json.dumps(dict(type='validation', steps=steps, report=report), allow_nan=False)+'\n')
                if feasible(report, config) and utility > best_utility:
                    best_utility = utility
                    metadata['best_validation'] = dict(steps=steps, report=report)
                    agent.save(output/'best.pt', metadata)
                next_eval = (steps//config.eval_frequency+1)*config.eval_frequency
            if steps >= next_checkpoint:
                agent.save(output/f'step_{steps}.pt', metadata)
                next_checkpoint = (steps//config.checkpoint_frequency+1)*config.checkpoint_frequency
        agent.save(output/'final.pt', metadata)
        (output/'selection.json').write_text(json.dumps(dict(best_validation=metadata['best_validation'],
            rule='Highest validation task utility among checkpoints satisfying all enabled constraints with sufficient labeled evidence.',
            status='feasible checkpoint selected' if metadata['best_validation'] else 'No feasible validation checkpoint; final.pt is not certified feasible.'), indent=2))
    finally:
        env.close()
    return agent, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-factory', required=True, help='module:callable or gym:RegisteredID')
    parser.add_argument('--train-settings', required=True)
    parser.add_argument('--validation-settings', required=True)
    parser.add_argument('--train-labels')
    parser.add_argument('--validation-labels')
    parser.add_argument('--output', required=True)
    for field in fields(Config):
        name = '--'+field.name.replace('_', '-')
        if isinstance(field.default, bool):
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=field.default)
        else:
            parser.add_argument(name, type=type(field.default), default=field.default)
    args = parser.parse_args()
    config = Config(**{field.name: getattr(args, field.name) for field in fields(Config)}).validate()
    train(args.env_factory, settings_file(args.train_settings), settings_file(args.validation_settings), args.output,
          config, load_labels(args.train_labels), load_labels(args.validation_labels))


if __name__ == '__main__':
    main()
