from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    act: np.ndarray
    rew: float
    obs2: np.ndarray
    done: float
    corrupted: bool
    corruption_type: str


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.obs2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.corrupted = np.zeros((size, 1), dtype=np.float32)
        self.corruption_type = np.empty((size,), dtype=object)

    def store(self, t: Transition) -> None:
        self.obs[self.ptr] = t.obs
        self.act[self.ptr] = t.act
        self.rew[self.ptr] = t.rew
        self.obs2[self.ptr] = t.obs2
        self.done[self.ptr] = t.done
        self.corrupted[self.ptr] = 1.0 if t.corrupted else 0.0
        self.corruption_type[self.ptr] = t.corruption_type
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=batch_size)
        return self._slice(idx)

    def sample_clean(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        clean_idx = np.flatnonzero(self.corrupted[: self.size, 0] < 0.5)
        if clean_idx.size == 0:
            return self.sample(batch_size, rng)
        idx = rng.choice(clean_idx, size=batch_size, replace=clean_idx.size < batch_size)
        return self._slice(idx)

    def clean_size(self) -> int:
        return int(np.flatnonzero(self.corrupted[: self.size, 0] < 0.5).size)

    def _slice(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "obs2": self.obs2[idx],
            "done": self.done[idx],
            "corrupted": self.corrupted[idx],
            "corruption_type": self.corruption_type[idx],
        }


class WindowBuffer:
    def __init__(self, length: int):
        self.length = length
        self.items: deque[Transition] = deque(maxlen=length)

    def append(self, transition: Transition) -> None:
        self.items.append(transition)

    def ready(self) -> bool:
        return len(self.items) == self.length

    def as_dict(self) -> dict[str, np.ndarray]:
        window = list(self.items)
        return {
            "obs": np.stack([t.obs for t in window]),
            "act": np.stack([t.act for t in window]),
            "rew": np.asarray([[t.rew] for t in window], dtype=np.float32),
            "obs2": np.stack([t.obs2 for t in window]),
            "done": np.asarray([[t.done] for t in window], dtype=np.float32),
            "corrupted": np.asarray([[1.0 if t.corrupted else 0.0] for t in window], dtype=np.float32),
            "corruption_type": np.asarray([t.corruption_type for t in window], dtype=object),
        }
