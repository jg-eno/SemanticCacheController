"""Method-independent evaluation on fresh Gymnasium environments."""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import random

import gymnasium as gym
import numpy as np

CORE_METRICS = ('requests', 'evaluable_labeled_requests', 'false_hit_rate',
                'recomputation_rate', 'mean_environment_reward')


def load_callable(spec):
    module, separator, name = spec.partition(':')
    if not separator or not module or not name:
        raise ValueError('Callable spec must have format module.path:name')
    result = getattr(importlib.import_module(module), name)
    if not callable(result):
        raise ValueError(f'Not callable: {spec}')
    return result


def make_environment(factory, settings):
    if isinstance(factory, str) and factory.startswith('gym:'):
        return gym.make(factory[4:], **settings)
    constructor = load_callable(factory) if isinstance(factory, str) else factory
    return constructor(**settings)


def make_method(spec, seed):
    """Construct a fresh observation-to-action callable for one evaluation."""
    if callable(spec):
        method = spec(seed=seed)
    else:
        if not isinstance(spec, dict):
            raise ValueError('Each method must be a dictionary')
        if not {'factory', 'settings'} <= set(spec):
            raise ValueError('Each method needs factory and settings fields')
        if set(spec) - {'factory', 'settings', 'env_settings'}:
            raise ValueError('Unexpected method configuration field')
        if not isinstance(spec['settings'], dict):
            raise ValueError('Method settings must be a JSON object')
        if 'env_settings' in spec and not isinstance(spec['env_settings'], dict):
            raise ValueError('env_settings must be a JSON object')
        method = load_callable(spec['factory'])(seed=seed, **spec['settings'])
    if not callable(method) and not callable(getattr(method, "select_action", None)):
        raise ValueError('Method factory must return a callable or an object with select_action()')
    return method


def step_record(reward, info, labels=None):
    status = info.get('reuse_status')
    effective = info.get('effective_action')
    if status not in ('correct', 'incorrect', 'unevaluable', 'not_reused') or effective not in (0, 1):
        raise ValueError('Step info needs a valid reuse_status and effective_action')
    if (status == 'not_reused') != (effective == 1):
        raise ValueError('Reuse status and effective action disagree')
    components = info.get('reward_components')
    if not isinstance(components, dict) or 'recompute_cost' not in components:
        raise ValueError('Step info needs reward_components with recompute_cost')
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in components.values()):
        raise ValueError('Reward components must be finite numbers')
    if not math.isfinite(reward) or not math.isclose(sum(components.values()), reward, abs_tol=1e-6):
        raise ValueError('Reward components must sum to the returned reward')
    penalty = components['recompute_cost']
    if penalty > 0 or (effective == 0 and penalty != 0):
        raise ValueError('Recomputation cost must be a signed penalty on recomputation only')
    labeled = info.get('labeled_request')
    if labeled is None and labels is not None:
        request_id = str(info.get('request_id', ''))
        if request_id not in labels:
            raise ValueError('Request missing from label map')
        labeled = labels[request_id] is not None
    if not isinstance(labeled, (bool, np.bool_)):
        raise ValueError('labeled_request must be boolean')
    if status in ('correct', 'incorrect') and not labeled:
        raise ValueError('Evaluable reuse requires a labeled incoming request')
    return dict(status=status, evaluable=bool(labeled) and status != 'unevaluable',
                reward=float(reward), recomputed=effective == 1,
                request_id=info.get('request_id'), phase_id=info.get('phase_id'))


def summarize(records):
    n = len(records)
    if not n:
        return None
    evaluable = sum(r['evaluable'] for r in records)
    incorrect = sum(r['status'] == 'incorrect' for r in records)
    recomputed = sum(r['recomputed'] for r in records)
    false_rate = incorrect / evaluable if evaluable else None
    return dict(requests=n, evaluable_labeled_requests=evaluable,
                false_hit_rate=false_rate, recomputation_rate=recomputed / n,
                mean_environment_reward=sum(r['reward'] for r in records) / n)


def evaluate_method(env_factory, env_settings, method, seed=0, labels=None,
                    max_episode_steps=100000):
    env = make_environment(env_factory, env_settings)
    records = []
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(max_episode_steps):
            # Pass the current environment to methods that need cache state.
            if hasattr(method, "select_action"):
                action = method.select_action(obs.copy(), env)
            else:
                action = method(obs.copy() if hasattr(obs, 'copy') else obs)

            obs, reward, terminated, truncated, info = env.step(action)
            records.append(step_record(reward, info, labels))
            if terminated or truncated:
                break
        else:
            raise ValueError('Episode exceeded max_episode_steps')
    finally:
        env.close()
    trace = [(r['request_id'], r['phase_id']) for r in records]
    digest = hashlib.sha256(
        json.dumps(trace, default=str).encode()
    ).hexdigest()

    return summarize(records), digest


def benchmark(env_factory, env_settings, methods, seed=0, labels=None,
              max_episode_steps=100000):
    """Evaluate method factories on identical fresh environments and request IDs."""
    if not methods:
        raise ValueError('At least one method is required')
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not isinstance(max_episode_steps, int) or isinstance(max_episode_steps, bool) or max_episode_steps < 1:
        raise ValueError('max_episode_steps must be a positive integer')
    results = {}
    trace_digest = None
    method_env_settings = {}
    for name, spec in methods.items():
        random.seed(seed)
        np.random.seed(seed)
        method = make_method(spec, seed)

        current_settings = dict(env_settings)
        if isinstance(spec, dict):
            overrides = spec.get('env_settings', {})
            current_settings.update(overrides)

        method_env_settings[name] = current_settings

        summary, digest = evaluate_method(env_factory,current_settings,method,seed,
            labels,max_episode_steps,)
        if trace_digest is None:
            trace_digest = digest
        elif digest != trace_digest:
            raise ValueError('Methods did not evaluate the same request trace')
        results[name] = summary
    
    return dict(seed=seed, settings=env_settings, method_env_settings=method_env_settings,
        trace_sha256=trace_digest, results=results,
        limitation='Offline simulation; intent labels are a proxy for answer correctness.',)


def load_json_object(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-factory', default='src.env.factory:make_env')
    parser.add_argument('--settings', required=True, help='JSON environment settings')
    parser.add_argument('--methods', required=True, help='JSON method-factory mapping')
    parser.add_argument('--labels', help='Optional request-ID to intent-label JSON map')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-episode-steps', type=int, default=100000)
    parser.add_argument('--output', default='benchmark/results.json',
                        help='Write the JSON report here (default: benchmark/results.json)')
    args = parser.parse_args()
    report = benchmark(args.env_factory, load_json_object(args.settings),
                       load_json_object(args.methods), args.seed,
                       load_json_object(args.labels) if args.labels else None,
                       args.max_episode_steps)
    payload = json.dumps(report, indent=2, allow_nan=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + '\n')


if __name__ == '__main__':
    main()
