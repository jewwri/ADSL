from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CORRUPTION_TYPES = [
    "clean",
    "reward_poisoning",
    "observation_corruption",
    "action_perturbation",
]


@dataclass
class CorruptionOutcome:
    obs: np.ndarray
    act: np.ndarray
    rew: float
    corrupted: bool
    corruption_type: str


class CorruptionEngine:
    def __init__(self, config, seed: int):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def _active(self, step: int) -> bool:
        if not self.config.enabled:
            return False
        if step < self.config.start_step:
            return False
        if self.config.end_step is not None and step >= self.config.end_step:
            return False
        if self.config.schedule == "random_sparse":
            return bool(self.rng.random() < self.config.random_sparse_p)
        if self.config.schedule == "bursty":
            cycle_step = (step - self.config.start_step) % max(1, self.config.burst_period)
            return cycle_step < self.config.burst_length
        if self.config.schedule == "persistent":
            return True
        return False

    def _sample_type(self) -> str:
        if self.config.type == "mixed":
            return self.rng.choice(CORRUPTION_TYPES[1:]).item()
        return self.config.type

    def apply(self, obs: np.ndarray, act: np.ndarray, rew: float, step: int) -> CorruptionOutcome:
        if not self._active(step):
            return CorruptionOutcome(obs=obs, act=act, rew=rew, corrupted=False, corruption_type="clean")

        corruption_type = self._sample_type()
        obs_out = np.array(obs, copy=True)
        act_out = np.array(act, copy=True)
        rew_out = float(rew)

        if corruption_type == "reward_poisoning":
            if self.rng.random() < self.config.reward_flip_p:
                rew_out = -rew_out
            else:
                rew_out = rew_out + float(self.rng.normal(0.0, 1.0))
        elif corruption_type == "observation_corruption":
            noise = self.rng.normal(0.0, self.config.observation_noise_scale, size=obs_out.shape)
            obs_out = obs_out + noise.astype(np.float32)
        elif corruption_type == "action_perturbation":
            noise = self.rng.normal(0.0, self.config.action_noise_scale, size=act_out.shape)
            act_out = act_out + noise.astype(np.float32)

        return CorruptionOutcome(
            obs=obs_out.astype(np.float32),
            act=act_out.astype(np.float32),
            rew=rew_out,
            corrupted=True,
            corruption_type=corruption_type,
        )
