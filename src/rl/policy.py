"""Categorical actor, independent critics, and PPO-Lagrangian optimizer."""
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .config import Config, FEATURE_NAMES, observation


def network(hidden, output):
    return nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output))


class ActorCritics(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.actor = network(hidden, 2)
        self.critics = nn.ModuleList([network(hidden, 1) for _ in range(3)])

    def distribution(self, observations):
        if observations.shape[-1] != len(FEATURE_NAMES):
            raise ValueError(f'Actor accepts {len(FEATURE_NAMES)} features only')
        return Categorical(logits=self.actor(observations))

    def values(self, observations):
        return torch.cat([critic(observations) for critic in self.critics], dim=-1)

    @torch.no_grad()
    def action(self, obs, deterministic=False):
        state = torch.from_numpy(observation(obs))
        dist = self.distribution(state)
        action = dist.probs.argmax() if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action)), self.values(state).numpy()


def combined_advantage(advantages, multipliers):
    return advantages[:, 0] - multipliers[0] * advantages[:, 1] - multipliers[1] * advantages[:, 2]


def clipped_objective(ratios, advantage, clip):
    return torch.minimum(ratios * advantage, torch.clamp(ratios, 1-clip, 1+clip) * advantage)


class Multipliers:
    def __init__(self, config):
        self.config = config
        self.values = np.array([config.initial_false_multiplier if config.false_constraint else 0,
                                config.initial_recompute_multiplier if config.recompute_constraint else 0], dtype=float)

    def update(self, false_hits, evaluable_steps, recomputes, requests):
        if requests < 1 or not 0 <= false_hits <= evaluable_steps <= requests or not 0 <= recomputes <= requests:
            raise ValueError('Invalid dual-update counts')
        rates = [false_hits/evaluable_steps if evaluable_steps else None, recomputes/requests]
        config = self.config
        enabled = [config.false_constraint, config.recompute_constraint]
        for i, (rate, limit, lr, maximum) in enumerate(zip(rates, [config.false_limit, config.recompute_limit],
                [config.false_dual_lr, config.recompute_dual_lr], [config.max_false_multiplier, config.max_recompute_multiplier])):
            if enabled[i] and rate is not None:
                self.values[i] = np.clip(self.values[i] + lr * (rate-limit), 0, maximum)
        return rates


class PPO:
    def __init__(self, config):
        self.config = config.validate()
        self.model = ActorCritics(config.hidden_size)
        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.model.critics.parameters(), lr=config.critic_lr)
        self.multipliers = Multipliers(config)

    def update(self, batch):
        cfg = self.config
        obs = torch.tensor(batch['observations'], dtype=torch.float32)
        actions = torch.tensor(batch['actions'], dtype=torch.long)
        old_logp = torch.tensor(batch['logp'], dtype=torch.float32)
        advantages = torch.tensor(batch['advantages'], dtype=torch.float32)
        returns = torch.tensor(batch['returns'], dtype=torch.float32)
        masks = torch.tensor(batch['masks'], dtype=torch.bool)
        # False-rate gradients use the evaluable-request denominator, not all requests.
        known = int(masks[:, 1].sum())
        if known:
            advantages[:, 1] *= len(obs) / known
        advantage = combined_advantage(advantages, self.multipliers.values)
        # Scale only, without centering: preserve constraint direction and relative weighting.
        advantage = advantage / advantage.std(unbiased=False).clamp_min(1.0)
        metrics = []
        for _ in range(cfg.epochs):
            for indices in torch.randperm(len(obs)).split(cfg.minibatch_size):
                dist = self.model.distribution(obs[indices])
                logratio = dist.log_prob(actions[indices]) - old_logp[indices]
                ratios = logratio.exp()
                if not torch.isfinite(ratios).all() or not torch.isfinite(logratio).all():
                    raise FloatingPointError('Nonfinite PPO probability ratio')
                entropy = dist.entropy().mean()
                actor_loss = -clipped_objective(ratios, advantage[indices], cfg.clip).mean() - cfg.entropy_coef * entropy
                values = self.model.values(obs[indices])
                losses = []
                for head in range(3):
                    mask = masks[indices, head]
                    losses.append(((values[mask, head] - returns[indices][mask, head]) ** 2).mean()
                                  if mask.any() else values[:, head].sum() * 0)
                critic_loss = cfg.value_coef * sum(losses)
                if not torch.isfinite(torch.stack([actor_loss, critic_loss, *losses])).all():
                    raise FloatingPointError('Nonfinite PPO loss')
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                (actor_loss + critic_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                    raise FloatingPointError('Nonfinite policy or critic parameters')
                metrics.append([float(actor_loss.detach()), *(float(loss.detach()) for loss in losses),
                                float(entropy.detach()), float(((ratios-1)-logratio).mean().detach()),
                                float(((ratios-1).abs() > cfg.clip).float().mean())])
        keys = ['actor_loss', 'reward_critic_loss', 'false_hit_critic_loss', 'recompute_critic_loss', 'entropy', 'approx_kl', 'clip_fraction']
        return dict(zip(keys, map(float, np.mean(metrics, axis=0))))

    def save(self, path, metadata=None):
        torch.save(dict(config=asdict(self.config), model=self.model.state_dict(),
                        actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                        multipliers=self.multipliers.values.tolist(), metadata=metadata or {}), path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        agent = cls(Config(**checkpoint['config']))
        agent.model.load_state_dict(checkpoint['model'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        agent.multipliers.values[:] = checkpoint['multipliers']
        return agent, checkpoint['metadata']
