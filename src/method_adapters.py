"""Optional adapters; the benchmark itself has no policy implementation."""
from .benchmark import load_callable


def callable_method(seed=0, spec=None):
    """Wrap an existing observation-to-action callable as a method factory."""
    return load_callable(spec)


def ppo_checkpoint(seed=0, checkpoint=None):
    """Load a PPO checkpoint as a deterministic observation-to-action method."""
    import torch
    from .rl.policy import PPO

    torch.manual_seed(seed)
    agent, _ = PPO.load(checkpoint)
    return lambda observation: agent.model.action(observation, deterministic=True)[0]
