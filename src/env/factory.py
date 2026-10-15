"""JSON-configurable factory for training and evaluating the repository environment."""
from ..datasets.workload import read
from ..models.semantic_cache import SentenceEncoder
from .semantic_cache_env import Rewards, SemanticCacheEnv


def make_env(workload_path, model='sentence-transformers/all-MiniLM-L6-v2',
             model_revision=None, rewards=None, **env_settings):
    """Build a fresh environment from a chronological JSONL trace."""
    if rewards is not None:
        rewards = Rewards(**rewards)
    return SemanticCacheEnv(read(workload_path),
                            encoder=SentenceEncoder(model, model_revision),
                            rewards=rewards, **env_settings)
