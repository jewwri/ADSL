from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import to_torch


LOG_STD_MIN, LOG_STD_MAX = -5, 2


@dataclass
class UpdateStats:
    critic_loss: float
    actor_loss: float
    alpha: float


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float):
        super().__init__()
        self.act_limit = act_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, std = self(obs)
        eps = torch.randn_like(mu)
        pre_tanh = mu + eps * std
        action = torch.tanh(pre_tanh) * self.act_limit
        logp = -0.5 * (
            ((pre_tanh - mu) / (std + 1e-8)) ** 2 + 2 * torch.log(std + 1e-8) + math.log(2 * math.pi)
        )
        logp = logp.sum(-1, keepdim=True)
        logp -= torch.log(1 - torch.tanh(pre_tanh) ** 2 + 1e-6).sum(-1, keepdim=True)
        return action, logp

    def act(self, obs: np.ndarray, device: torch.device) -> np.ndarray:
        obs_t = to_torch(obs[None, :], device)
        with torch.no_grad():
            mu, _ = self(obs_t)
            act = torch.tanh(mu) * self.act_limit
        return act.cpu().numpy()[0]


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, act], dim=-1))


def make_env(env_id: str, max_episode_steps: int | None = None) -> gym.Env:
    if max_episode_steps is None:
        return gym.make(env_id)
    return gym.make(env_id, max_episode_steps=max_episode_steps)


def soft_update(src: nn.Module, dst: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, pt in zip(src.parameters(), dst.parameters()):
            pt.data.mul_(1 - tau)
            pt.data.add_(tau * p.data)


def sac_update(
    actor: Actor,
    critic1: Critic,
    critic2: Critic,
    target1: Critic,
    target2: Critic,
    log_alpha: torch.Tensor,
    opt_actor,
    opt_critic,
    opt_alpha,
    batch: dict[str, np.ndarray],
    gamma: float,
    tau: float,
    target_entropy: float,
    device: torch.device,
) -> UpdateStats:
    obs = to_torch(batch["obs"], device)
    act = to_torch(batch["act"], device)
    rew = to_torch(batch["rew"], device)
    obs2 = to_torch(batch["obs2"], device)
    done = to_torch(batch["done"], device)
    alpha = log_alpha.exp()

    with torch.no_grad():
        next_action, next_logp = actor.sample(obs2)
        q1_t = target1(obs2, next_action)
        q2_t = target2(obs2, next_action)
        q_target = torch.min(q1_t, q2_t) - alpha * next_logp
        backup = rew + gamma * (1 - done) * q_target

    q1 = critic1(obs, act)
    q2 = critic2(obs, act)
    critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
    opt_critic.zero_grad(set_to_none=True)
    critic_loss.backward()
    opt_critic.step()

    policy_action, logp = actor.sample(obs)
    q_pi = torch.min(critic1(obs, policy_action), critic2(obs, policy_action))
    actor_loss = (alpha * logp - q_pi).mean()
    opt_actor.zero_grad(set_to_none=True)
    actor_loss.backward()
    opt_actor.step()

    alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
    opt_alpha.zero_grad(set_to_none=True)
    alpha_loss.backward()
    opt_alpha.step()

    soft_update(critic1, target1, tau)
    soft_update(critic2, target2, tau)

    return UpdateStats(
        critic_loss=float(critic_loss.detach()),
        actor_loss=float(actor_loss.detach()),
        alpha=float(log_alpha.exp().detach()),
    )

