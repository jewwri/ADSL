from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from .config import ExperimentConfig
from .corruption import CorruptionEngine
from .data import ReplayBuffer, Transition, WindowBuffer
from .detection import (
    STEP_FEATURE_NAMES,
    WINDOW_FEATURE_NAMES,
    compute_single_transition_features,
    compute_window_features,
)
from .logging_utils import RunRecorder
from .pipelines import (
    _accept_result,
    _act_dim,
    _act_limit,
    _action_from_env,
    _mix_batches,
    _obs_dim,
    compute_detection_metrics,
    export_summary_row,
    evaluate_policy,
)
from .rl import Actor, Critic, make_env, sac_update
from .telemetry import TelemetryTracker
from .utils import dump_json, ensure_dir, set_seed

try:
    from sklearn.ensemble import IsolationForest
except ImportError:  # pragma: no cover - dependency is optional.
    IsolationForest = None


class IsolationForestWindowDetector:
    def __init__(
        self,
        *,
        contamination: float,
        n_estimators: int,
        random_state: int,
        min_fit_windows: int,
    ) -> None:
        if IsolationForest is None:
            raise ImportError(
                "Isolation Forest baseline requires scikit-learn. "
                "Install with `pip install .[detectors]`."
            )
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.min_fit_windows = max(8, int(min_fit_windows))
        self._warmup_features: list[np.ndarray] = []
        self._fitted = False
        self._threshold_raw = 0.0
        self._raw_min = 0.0
        self._raw_max = 1.0

    @property
    def fitted(self) -> bool:
        return self._fitted

    def collect(self, features: np.ndarray) -> None:
        self._warmup_features.append(np.asarray(features, dtype=np.float32))

    def fit(self) -> float:
        if self._fitted or len(self._warmup_features) < self.min_fit_windows:
            return 0.0
        data = np.stack(self._warmup_features).astype(np.float32)
        started = perf_counter()
        self.model.fit(data)
        fit_ms = (perf_counter() - started) * 1000.0
        raw_scores = -self.model.score_samples(data)
        self._threshold_raw = float(-self.model.offset_)
        self._raw_min = float(raw_scores.min())
        self._raw_max = float(max(raw_scores.max(), self._threshold_raw))
        self._fitted = True
        return fit_ms

    def detect(self, features: np.ndarray) -> tuple[float, bool, float]:
        if not self._fitted:
            return 0.0, False, 0.0
        raw = float(-self.model.score_samples(np.asarray(features, dtype=np.float32)[None, :])[0])
        self._raw_min = min(self._raw_min, raw)
        self._raw_max = max(self._raw_max, raw)
        denom = max(1e-6, self._raw_max - self._raw_min)
        risk = float(np.clip((raw - self._raw_min) / denom, 0.0, 1.0))
        return risk, bool(raw >= self._threshold_raw), raw


def run_isolation_forest_experiment(
    config: ExperimentConfig,
    *,
    gate_mode: str = "sanitize",
    contamination: float = 0.05,
    n_estimators: int = 200,
    min_fit_windows: int = 128,
    risk_threshold: float = 0.5,
) -> Path:
    if gate_mode not in {"accept", "attenuate", "block", "sanitize"}:
        raise ValueError(f"Unsupported gate mode: {gate_mode}")
    if not 0.0 <= risk_threshold <= 1.0:
        raise ValueError("risk_threshold must be in [0.0, 1.0]")

    set_seed(config.seed)
    rng = np.random.default_rng(config.seed + 17)
    env = make_env(config.env.id, config.env.max_episode_steps)
    eval_env = make_env(config.env.id, config.env.max_episode_steps)
    if hasattr(env.action_space, "n"):
        env.close()
        eval_env.close()
        raise ValueError(
            f"Continuous-control experiments only: {config.env.id} has a discrete action space."
        )
    obs, _ = env.reset(seed=config.seed)
    obs = np.asarray(obs, dtype=np.float32)

    obs_dim = _obs_dim(env.observation_space)
    act_dim = _act_dim(env.action_space)
    act_limit = _act_limit(env.action_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(obs_dim, act_dim, act_limit).to(device)
    critic1 = Critic(obs_dim, act_dim).to(device)
    critic2 = Critic(obs_dim, act_dim).to(device)
    target1 = Critic(obs_dim, act_dim).to(device)
    target2 = Critic(obs_dim, act_dim).to(device)

    target1.load_state_dict(critic1.state_dict())
    target2.load_state_dict(critic2.state_dict())
    opt_actor = optim.Adam(actor.parameters(), lr=config.training.lr)
    opt_critic = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=config.training.lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    opt_alpha = optim.Adam([log_alpha], lr=config.training.lr)
    target_entropy = -act_dim

    run_name = f"{config.name}_seed{config.seed}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    run_dir = ensure_dir(Path(config.output_root) / run_name)
    recorder = RunRecorder(run_dir=run_dir)
    dump_json(run_dir / "config.json", asdict(config))
    telemetry = TelemetryTracker()
    dump_json(
        run_dir / "detector_baseline.json",
        {
            "detector_backend": "isolation_forest",
            "gate_mode": gate_mode,
            "iforest_contamination": contamination,
            "iforest_n_estimators": n_estimators,
            "iforest_min_fit_windows": min_fit_windows,
            "detector_threshold": risk_threshold,
        },
    )

    replay = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=config.training.replay_size)
    windows = WindowBuffer(config.detector.window_length)
    corruption = CorruptionEngine(config.corruption, seed=config.seed + 101)
    detector = IsolationForestWindowDetector(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=config.seed + 701,
        min_fit_windows=min_fit_windows,
    )

    detector_labels: list[int] = []
    detector_flags: list[int] = []
    accepted_updates = 0
    blocked_updates = 0
    harmful_accepts = 0
    benign_blocks = 0
    sanitized_transitions = 0
    attack_steps = 0
    flagged_windows = 0
    flagged_harmful_windows = 0
    intervention_counts = {name: 0 for name in ("accept", "attenuate", "block", "sanitize")}
    sanitize_clean_replay_uses = 0
    attenuate_clean_replay_uses = 0
    detector_runtime_ms_total = 0.0
    detector_fit_runtime_ms_total = 0.0
    detector_runtime_calls = 0

    for step in range(config.training.total_steps):
        action = _action_from_env(env, actor, obs, step, config.training.start_steps, device)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)

        corrupted = corruption.apply(next_obs, action, float(reward), step)
        if corrupted.corrupted:
            attack_steps += 1
        done = float(terminated or truncated)
        transition = Transition(
            obs=obs,
            act=corrupted.act,
            rew=corrupted.rew,
            obs2=corrupted.obs,
            done=done,
            corrupted=corrupted.corrupted,
            corruption_type=corrupted.corruption_type,
        )
        windows.append(transition)

        detector_risk = 0.0
        detector_raw_score = 0.0
        detector_flagged = False
        decision = _accept_result()
        decision_log_idx: int | None = None
        reward_shift = abs(corrupted.rew - reward)
        action_shift = float(np.linalg.norm(corrupted.act - action))
        obs_shift = float(np.linalg.norm(corrupted.obs - next_obs))

        if windows.ready() and config.detector.enabled:
            window = windows.as_dict()
            started = perf_counter()
            features = compute_window_features(window, config.detector.features)
            if step < config.detector.warmup_steps:
                detector.collect(features)
            else:
                detector_fit_runtime_ms_total += detector.fit()
                detector_risk, detector_flagged, detector_raw_score = detector.detect(features)
                detector_flagged = bool(detector_flagged and detector_risk >= risk_threshold)
            detector_runtime_ms = (perf_counter() - started) * 1000.0
            detector_runtime_ms_total += detector_runtime_ms
            detector_runtime_calls += 1

            detector_labels.append(int(window["corrupted"].max() > 0))
            detector_flags.append(int(detector_flagged))

            if detector_flagged:
                decision = _accept_result(trust=max(0.0, 1.0 - detector_risk))
                decision.action = gate_mode
            else:
                decision = _accept_result(trust=max(0.0, 1.0 - detector_risk))

            recorder.log_decision(
                {
                    **telemetry.sample(),
                    "global_step": step,
                    "timestamp_utc": datetime.utcnow().isoformat(),
                    "env_id": config.env.id,
                    "seed": config.seed,
                    "schedule": config.corruption.schedule,
                    "poison_type": config.corruption.type,
                    "detector_backend": "isolation_forest",
                    "detector_gate_mode": gate_mode,
                    "detector_threshold": risk_threshold,
                    "detector_risk": detector_risk,
                    "detector_raw_score": detector_raw_score,
                    "detector_runtime_ms": detector_runtime_ms,
                    "detector_flagged": int(detector_flagged),
                    "controller_action": decision.action,
                    "controller_score": decision.score,
                    "trust": decision.trust,
                    "predicted_deviation": 0.0,
                    "predicted_return_drop": 0.0,
                    "reward_shift": reward_shift,
                    "action_shift": action_shift,
                    "obs_shift": obs_shift,
                    "sanitize_replay_mode": "clean_only_replacement",
                    "attenuate_replay_mode": "weighted_mix",
                    "clean_replay_available": int(replay.clean_size() > 0),
                    "used_clean_only_replay": 0,
                }
            )
            decision_log_idx = len(recorder.decisions) - 1
            if detector_flagged:
                flagged_windows += 1
                flagged_harmful_windows += int(window["corrupted"].max() > 0)
            intervention_counts[decision.action] += 1

            if config.logging.save_transition_windows:
                single_features = compute_single_transition_features(
                    {
                        "obs": transition.obs,
                        "act": transition.act,
                        "rew": np.asarray([transition.rew], dtype=np.float32),
                        "obs2": transition.obs2,
                    }
                )
                window_payload = {
                    **telemetry.sample(),
                    "global_step": step,
                    "timestamp_utc": datetime.utcnow().isoformat(),
                    "env_id": config.env.id,
                    "seed": config.seed,
                    "schedule": config.corruption.schedule,
                    "poison_type": config.corruption.type,
                    "window_length": config.detector.window_length,
                    "label": int(window["corrupted"].max() > 0),
                    "detector_flag": int(detector_flagged),
                    "detector_backend": "isolation_forest",
                    "detector_gate_mode": gate_mode,
                    "detector_threshold": risk_threshold,
                    "detector_runtime_ms": detector_runtime_ms,
                    "detector_raw_score": detector_raw_score,
                }
                for name, value in zip(WINDOW_FEATURE_NAMES, features):
                    window_payload[f"window_{name}"] = float(value)
                for name, value in zip(STEP_FEATURE_NAMES, single_features):
                    window_payload[f"step_{name}"] = float(value)
                recorder.log_detector_window(window_payload)

        should_store = True
        if bool(transition.corrupted) and decision.action in {"block", "sanitize"}:
            should_store = False
            sanitized_transitions += 1

        if should_store:
            replay.store(transition)

        if replay.size >= config.training.batch_size:
            train_batch = replay.sample(config.training.batch_size, rng)
            harmful_update = bool(transition.corrupted and decision.action in {"accept", "attenuate"})
            clean_replay_available = replay.clean_size() > 0
            used_clean_only_replay = 0

            if decision.action == "block":
                blocked_updates += 1
                if not transition.corrupted:
                    benign_blocks += 1
            else:
                if decision.action == "sanitize" and clean_replay_available:
                    train_batch = replay.sample_clean(config.training.batch_size, rng)
                    sanitize_clean_replay_uses += 1
                    used_clean_only_replay = 1
                elif decision.action == "attenuate" and clean_replay_available:
                    clean_batch = replay.sample_clean(config.training.batch_size, rng)
                    train_batch = _mix_batches(
                        train_batch,
                        clean_batch,
                        1.0 - float(config.controller.attenuate_clean_ratio),
                    )
                    attenuate_clean_replay_uses += 1
                    used_clean_only_replay = 1

                accepted_updates += 1
                if harmful_update:
                    harmful_accepts += 1
                sac_update(
                    actor=actor,
                    critic1=critic1,
                    critic2=critic2,
                    target1=target1,
                    target2=target2,
                    log_alpha=log_alpha,
                    opt_actor=opt_actor,
                    opt_critic=opt_critic,
                    opt_alpha=opt_alpha,
                    batch=train_batch,
                    gamma=config.training.gamma,
                    tau=config.training.tau,
                    target_entropy=target_entropy,
                    device=device,
                )
            if decision_log_idx is not None:
                recorder.decisions[decision_log_idx]["used_clean_only_replay"] = used_clean_only_replay

        obs = next_obs if not (terminated or truncated) else np.asarray(env.reset()[0], dtype=np.float32)

        if (step + 1) % config.training.eval_every == 0:
            eval_return_mean = evaluate_policy(eval_env, actor, device, config.training.eval_episodes)
            harmful_accept_rate = harmful_accepts / max(1, accepted_updates)
            benign_block_rate = benign_blocks / max(1, blocked_updates)
            recorder.log_metric(
                {
                    **telemetry.sample(),
                    "run_name": run_name,
                    "timestamp_utc": datetime.utcnow().isoformat(),
                    "env_id": config.env.id,
                    "seed": config.seed,
                    "global_step": step + 1,
                    "eval_return_mean": eval_return_mean,
                    "accepted_updates": accepted_updates,
                    "blocked_updates": blocked_updates,
                    "sanitized_transitions": sanitized_transitions,
                    "flagged_windows": flagged_windows,
                    "flagged_harmful_windows": flagged_harmful_windows,
                    "interventions_accept": intervention_counts["accept"],
                    "interventions_attenuate": intervention_counts["attenuate"],
                    "interventions_block": intervention_counts["block"],
                    "interventions_sanitize": intervention_counts["sanitize"],
                    "sanitize_clean_replay_uses": sanitize_clean_replay_uses,
                    "attenuate_clean_replay_uses": attenuate_clean_replay_uses,
                    "attack_steps": attack_steps,
                    "harmful_accept_rate": harmful_accept_rate,
                    "benign_block_rate": benign_block_rate,
                    "detector_backend": "isolation_forest",
                    "detector_gate_mode": gate_mode,
                    "detector_threshold": risk_threshold,
                    "detector_runtime_ms_mean": detector_runtime_ms_total / max(1, detector_runtime_calls),
                    "detector_fit_runtime_ms_total": detector_fit_runtime_ms_total,
                    "detector_precision": np.nan,
                    "detector_recall": np.nan,
                    "detector_f1": np.nan,
                }
            )

    final_payload = {
        **telemetry.sample(),
        "run_name": run_name,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "env_id": config.env.id,
        "seed": config.seed,
        "global_step": config.training.total_steps,
        "eval_return_mean": evaluate_policy(eval_env, actor, device, config.training.eval_episodes),
        "accepted_updates": accepted_updates,
        "blocked_updates": blocked_updates,
        "sanitized_transitions": sanitized_transitions,
        "flagged_windows": flagged_windows,
        "flagged_harmful_windows": flagged_harmful_windows,
        "interventions_accept": intervention_counts["accept"],
        "interventions_attenuate": intervention_counts["attenuate"],
        "interventions_block": intervention_counts["block"],
        "interventions_sanitize": intervention_counts["sanitize"],
        "sanitize_clean_replay_uses": sanitize_clean_replay_uses,
        "attenuate_clean_replay_uses": attenuate_clean_replay_uses,
        "attack_steps": attack_steps,
        "harmful_accept_rate": harmful_accepts / max(1, accepted_updates),
        "benign_block_rate": benign_blocks / max(1, blocked_updates),
        "detector_backend": "isolation_forest",
        "detector_gate_mode": gate_mode,
        "detector_threshold": risk_threshold,
        "detector_runtime_ms_mean": detector_runtime_ms_total / max(1, detector_runtime_calls),
        "detector_fit_runtime_ms_total": detector_fit_runtime_ms_total,
    }

    if detector_flags:
        final_payload.update(compute_detection_metrics(detector_labels, detector_flags))

    recorder.log_metric(final_payload)
    recorder.flush()
    export_summary_row(run_dir)
    env.close()
    eval_env.close()
    return run_dir
