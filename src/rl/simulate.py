"""Offline, reproducible PPO simulation on the repository's cache environment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..datasets.workload import generate, write
from ..env.semantic_cache_env import SemanticCacheEnv
from ..evaluate import always_recompute, always_reuse, drift_rule
from .config import Config
from .evaluate import compare
from .train import train


class SyntheticEncoder:
    """Stable local embeddings with two intentionally similar intent pairs."""

    _centers = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.96, 0.28, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.96, 0.28],
    ])

    def encode(self, query):
        intent, variant = query.split('/')
        index = int(intent.removeprefix('intent-'))
        if index not in range(len(self._centers)):
            raise ValueError('Unknown synthetic intent')
        digest = hashlib.sha256(variant.encode()).digest()
        jitter = (np.frombuffer(digest[:4], dtype=np.uint8).astype(float) - 127.5) / 127.5
        return self._centers[index] + 0.015 * jitter


def make_sim_env(workload_path, **settings):
    """Factory compatible with src.rl.train and src.rl.evaluate."""
    from ..datasets.workload import read
    return SemanticCacheEnv(read(workload_path), encoder=SyntheticEncoder(), **settings)


def make_workloads(directory, seed, requests_per_phase):
    source = [dict(intent_id=str(intent), query=f'intent-{intent}/variant-{variant}')
              for intent in range(4) for variant in range(12)]
    # Different ordered phase mixtures provide cache reuse and distribution shifts.
    n = requests_per_phase
    phases = [{"0": 3*n//4, "1": n-3*n//4},
              {"1": 3*n//4, "2": n-3*n//4},
              {"2": 3*n//4, "3": n-3*n//4}]
    settings = {}
    for offset, split in enumerate(('train', 'validation', 'held_out')):
        path = directory / f'{split}.jsonl'
        write(path, generate(source, phases=phases, seed=seed + offset))
        settings[split] = dict(workload_path=str(path.resolve()), capacity=16,
                               threshold=0.85, window=16, drift_window=16,
                               projection_count=8, histogram_bins=8, projection_seed=seed)
        (directory / f'{split}_settings.json').write_text(json.dumps(settings[split], indent=2))
    return settings


def run(output, seed=0, total_steps=2048, requests_per_phase=64, rollout_length=128,
        false_limit=0.05, recompute_limit=0.65):
    output = Path(output)
    config = Config(seed=seed, total_steps=total_steps, rollout_length=rollout_length,
                    minibatch_size=min(64, rollout_length), false_limit=false_limit,
                    recompute_limit=recompute_limit, eval_frequency=512,
                    checkpoint_frequency=512, log_frequency=128).validate()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    if requests_per_phase < 4:
        raise ValueError('requests_per_phase must be at least 4')
    output.mkdir(parents=True, exist_ok=True)
    settings = make_workloads(output, seed, requests_per_phase)
    agent, metadata = train(make_sim_env, settings['train'], settings['validation'],
                            output / 'training', config)
    best = metadata['best_validation'] is not None
    if best:
        from .policy import PPO
        agent, _ = PPO.load(output / 'training' / 'best.pt')
    results = compare(make_sim_env, settings['held_out'], agent,
                      dict(always_reuse=always_reuse, always_recompute=always_recompute,
                           drift_rule=drift_rule), [seed])
    checkpoint = 'best.pt' if best else 'final.pt'
    summary = dict(checkpoint=str(output / 'training' / checkpoint),
                   validation_feasible=best, held_out={name: report['runs'][0]['overall']
                                                      for name, report in results.items()})
    (output / 'held_out_results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New or empty directory for all run artifacts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total-steps', type=int, default=2048)
    parser.add_argument('--requests-per-phase', type=int, default=64)
    parser.add_argument('--rollout-length', type=int, default=128)
    parser.add_argument('--false-limit', type=float, default=0.05)
    parser.add_argument('--recompute-limit', type=float, default=0.65)
    args = parser.parse_args()
    summary = run(**vars(args))
    print(json.dumps(dict(checkpoint=summary['checkpoint'],
                          validation_feasible=summary['validation_feasible'],
                          held_out={name: {key: metrics[key] for key in
                              ('cumulative_task_utility', 'false_hit_rate', 'recomputation_rate')}
                              for name, metrics in summary['held_out'].items()}), indent=2))


if __name__ == '__main__':
    main()
