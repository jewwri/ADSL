from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DetectorResult:
    risk: float
    is_flagged: bool
    features: np.ndarray


WINDOW_FEATURE_NAMES = [
    "reward_mean",
    "reward_std",
    "reward_min",
    "reward_max",
    "action_norm_mean",
    "action_norm_std",
    "state_shift_mean",
    "state_shift_std",
    "state_shift_max",
    "temporal_abs_mean",
    "temporal_abs_max",
]

STEP_FEATURE_NAMES = [
    "step_reward_abs",
    "step_action_norm",
    "step_state_shift",
    "step_state_abs_mean",
]


def compute_window_features(window: dict[str, np.ndarray], feature_cfg) -> np.ndarray:
    obs = window["obs"]
    act = np.asarray(window["act"], dtype=np.float32)
    if act.ndim == 1:
        act = act[:, None]
    rew = window["rew"].reshape(-1)
    obs2 = window["obs2"]
    feats: list[float] = []

    if feature_cfg.reward_stats:
        feats.extend([float(rew.mean()), float(rew.std()), float(rew.min()), float(rew.max())])
    if feature_cfg.action_stats:
        norms = np.linalg.norm(act, axis=1)
        feats.extend([float(norms.mean()), float(norms.std())])
    if feature_cfg.state_shift:
        delta = np.linalg.norm(obs2 - obs, axis=1)
        feats.extend([float(delta.mean()), float(delta.std()), float(delta.max())])
    if feature_cfg.temporal_delta:
        obs_d = np.diff(obs, axis=0) if len(obs) > 1 else np.zeros_like(obs[:1])
        feats.extend([float(np.abs(obs_d).mean()), float(np.abs(obs_d).max())])

    return np.asarray(feats, dtype=np.float32)


def compute_single_transition_features(transition: dict[str, np.ndarray]) -> np.ndarray:
    obs = np.asarray(transition["obs"], dtype=np.float32).reshape(-1)
    act = np.asarray(transition["act"], dtype=np.float32).reshape(-1)
    rew = float(np.asarray(transition["rew"], dtype=np.float32).reshape(-1)[0])
    obs2 = np.asarray(transition["obs2"], dtype=np.float32).reshape(-1)
    delta = obs2 - obs
    feats = [
        abs(rew),
        float(np.linalg.norm(act)),
        float(np.linalg.norm(delta)),
        float(np.abs(delta).mean()),
    ]
    return np.asarray(feats, dtype=np.float32)


class WindowRiskModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeuristicDetector:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self._baseline_mean: np.ndarray | None = None
        self._baseline_var: np.ndarray | None = None
        self._baseline_count = 0

    def update_baseline(self, features: np.ndarray) -> None:
        x = np.asarray(features, dtype=np.float32)
        if self._baseline_mean is None:
            self._baseline_mean = x.copy()
            self._baseline_var = np.zeros_like(x)
            self._baseline_count = 1
            return

        self._baseline_count += 1
        delta = x - self._baseline_mean
        self._baseline_mean = self._baseline_mean + delta / self._baseline_count
        delta2 = x - self._baseline_mean
        self._baseline_var = self._baseline_var + delta * delta2

    def _baseline_std(self) -> np.ndarray | None:
        if self._baseline_mean is None or self._baseline_var is None or self._baseline_count < 2:
            return None
        return np.sqrt(self._baseline_var / max(1, self._baseline_count - 1) + 1e-6)

    def score(self, features: np.ndarray) -> float:
        baseline_std = self._baseline_std()
        if self._baseline_mean is not None and baseline_std is not None:
            z = np.abs((features - self._baseline_mean) / baseline_std)
            score = 0.6 * z.mean() + 0.4 * z.max()
            return float(np.clip(score / 4.0, 0.0, 1.0))

        centered = np.abs(features - features.mean())
        denom = np.linalg.norm(features) + 1e-6
        return float(np.clip(centered.mean() / denom * 5.0, 0.0, 1.0))

    def detect(self, window: dict[str, np.ndarray], feature_cfg) -> DetectorResult:
        features = compute_window_features(window, feature_cfg)
        risk = self.score(features)
        return DetectorResult(risk=risk, is_flagged=risk >= self.threshold, features=features)
