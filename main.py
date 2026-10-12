"""Run the bundled temporal workload with the deterministic reuse baseline."""
from pathlib import Path

from src.datasets.workload import read
from src.evaluate import always_reuse
from src.semantic_cache import FEATURE_NAMES, Rewards, SemanticCacheEnv


def main(encoder=None):
    workload_path = Path(__file__).resolve().parent / 'src/datasets/example_workload.jsonl'
    env = SemanticCacheEnv(
        workload=read(workload_path),
        encoder=encoder,  # None selects the documented sentence-embedding model.
        capacity=32,
        threshold=0.85,
        window=16,  # Rolling miss/reuse metrics.
        drift_window=16,  # Each of the two adjacent JS comparison windows.
        projection_count=8,
        histogram_bins=8,
        projection_seed=0,
        rewards=Rewards(recompute_cost=0.2),
    )
    try:
        observation, _ = env.reset(seed=42)
        print('Initial features:', dict(zip(FEATURE_NAMES, observation.tolist())))
        total_reward = 0.0
        while True:
            observation, reward, terminated, truncated, info = env.step(always_reuse(observation))
            total_reward += reward
            print(
                f"step={info['step']} requested={info['requested_action']} "
                f"effective={info['effective_action']} hit={info['cache_hit']} "
                f"reuse={info['reuse_status']} reward={reward:.2f}"
            )
            if terminated or truncated:
                break
        print(f'Total simulated reward: {total_reward:.2f}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
