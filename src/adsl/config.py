from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    id: str
    max_episode_steps: int | None = None


@dataclass
class TrainingConfig:
    total_steps: int = 10_000
    batch_size: int = 64
    replay_size: int = 100_000
    start_steps: int = 1_000
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    eval_every: int = 2_000
    eval_episodes: int = 5
    early_stopping_enabled: bool = False
    early_stopping_min_steps: int = 100_000
    early_stopping_patience_evals: int = 25
    early_stopping_min_delta: float = 0.01
    early_stopping_smoothing_window: int = 5


@dataclass
class DetectorFeatureConfig:
    reward_stats: bool = True
    action_stats: bool = True
    state_shift: bool = True
    temporal_delta: bool = True


@dataclass
class DetectorConfig:
    enabled: bool = True
    window_length: int = 16
    trigger_threshold: float = 0.5
    warmup_steps: int = 500
    features: DetectorFeatureConfig = field(default_factory=DetectorFeatureConfig)


@dataclass
class ExpertsConfig:
    enabled: bool = True
    mode: str = "experts"
    classes: list[str] = field(default_factory=lambda: ["clean", "reward_poisoning"])


@dataclass
class ControllerConfig:
    enabled: bool = True
    mode: str = "mcts"
    harm_threshold: float = 0.5
    sanitize_replay: bool = False
    harm_weights: dict[str, float] = field(
        default_factory=lambda: {
            "baseline_deviation": 0.55,
            "predicted_return_drop": 0.25,
            "detector_risk": 0.20,
        }
    )
    mcts_simulations: int = 24
    mcts_horizon: int = 3
    mcts_exploration_c: float = 1.4
    baseline_warmup_steps: int = 1000
    baseline_reference_size: int = 128
    deviation_threshold: float = 0.15
    sanitize_replay_mode: str = "clean_only_replacement"


@dataclass
class CorruptionConfig:
    enabled: bool = False
    type: str = "clean"
    schedule: str = "random_sparse"
    severity: str = "low"
    start_step: int = 0
    end_step: int | None = None
    random_sparse_p: float = 0.0
    burst_length: int = 20
    burst_period: int = 200
    observation_noise_scale: float = 0.1
    reward_flip_p: float = 0.05
    action_noise_scale: float = 0.1
    attacker_capability: str = "black_box"
    attacker_parameter_access: bool = False
    attack_surface: str = "pre_replay_admission"
    poison_budget: float = 0.0
    poison_budget_unit: str = "event_fraction"


@dataclass
class LoggingConfig:
    save_transition_windows: bool = True
    save_model_checkpoints: bool = False


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 0
    output_root: str = "results/runs"
    env: EnvConfig = field(default_factory=lambda: EnvConfig(id="HalfCheetah-v4"))
    training: TrainingConfig = field(default_factory=TrainingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    experts: ExpertsConfig = field(default_factory=ExpertsConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _construct(cls, data: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in allowed})


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig(
        name=raw["name"],
        seed=raw.get("seed", 0),
        output_root=raw.get("output_root", "results/runs"),
        env=_construct(EnvConfig, raw.get("env", {})),
        training=_construct(TrainingConfig, raw.get("training", {})),
        detector=DetectorConfig(
            enabled=raw.get("detector", {}).get("enabled", True),
            window_length=raw.get("detector", {}).get("window_length", 16),
            trigger_threshold=raw.get("detector", {}).get("trigger_threshold", 0.5),
            warmup_steps=raw.get("detector", {}).get("warmup_steps", 500),
            features=_construct(
                DetectorFeatureConfig,
                raw.get("detector", {}).get("features", {}),
            ),
        ),
        experts=_construct(ExpertsConfig, raw.get("experts", {})),
        controller=_construct(ControllerConfig, raw.get("controller", {})),
        corruption=_construct(CorruptionConfig, raw.get("corruption", {})),
        logging=_construct(LoggingConfig, raw.get("logging", {})),
    )
