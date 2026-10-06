import gymnasium as gym
from gymnasium import spaces
from src.datasets.dataset import WildCatDataset


class CacheDriftEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.ds = WildCatDataset()
        self.current_idx = 0
        self.action_space = spaces.Discrete(3)
        self.action_labels = {
            0: "REUSE",
            1: "VALIDATE",
            2: "RECOMPUTE"
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        return self.ds[self.current_idx]["content"], {}

    def step(self, action):
        query = self.ds[self.current_idx]["content"]
        self.current_idx += 1
        terminated = self.current_idx >= len(self.ds)
        truncated = False
        next_query = (
            self.ds[self.current_idx]["content"]
            if not terminated
            else None
        )
        return next_query, 0.0, terminated, truncated, {
            "query": query,
            "action": self.action_labels[action]
        }