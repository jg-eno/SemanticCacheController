"""Fixed-threshold action methods for Gym environment benchmarks."""

class FixedThresholdMethod:
    """Reuse a matching cached response; recompute when there is no match."""

    def __init__(self, seed=0):
        self.seed = seed

    def select_action(self, observation, env):
        if env.candidate is not None:
            return 0  # REUSE
        return 1      # RECOMPUTE


def make_fixed_threshold_method(seed=0):
    return FixedThresholdMethod(seed=seed)
