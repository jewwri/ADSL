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
from .control import BaselineReference, LookaheadContext, LookaheadController, MCTSResult, ShadowLearnerState
from .corruption import CorruptionEngine
from .data import ReplayBuffer, Transition, WindowBuffer
from .detection import (
    STEP_FEATURE_NAMES,
    WINDOW_FEATURE_NAMES,
    HeuristicDetector,
    compute_single_transition_features,
)
from .early_stopping import build_early_stopping_monitor
from .experts import ExpertClassifier, train_supervised_classifier
from .logging_utils import RunRecorder
from .rl import Actor, Critic, make_env, sac_update
from .telemetry import TelemetryTracker
from .utils import dump_json, ensure_dir, set_seed


def _action_from_env(env, actor: Actor | None, obs: np.ndarray, step: int, start_steps: int, device):
    if actor is None or step < start_steps:
        action = env.action_space.sample()
    else:
        action = actor.act(obs.astype(np.float32), device)
    return np.asarray(action, dtype=np.float32)


def _obs_dim(space) -> int:
    return int(np.prod(space.shape))


def _act_dim(space) -> int:
    if hasattr(space, "n"):
        return 1
    return int(np.prod(space.shape))


def _act_limit(space) -> float:
    if hasattr(space, "high"):
        return float(np.max(space.high))
    return 1.0


def _window_to_batch(window: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "obs": np.asarray(window["obs"], dtype=np.float32),
        "act": np.asarray(window["act"], dtype=np.float32),
        "rew": np.asarray(window["rew"], dtype=np.float32),
        "obs2": np.asarray(window["obs2"], dtype=np.float32),
        "done": np.asarray(window["done"], dtype=np.float32),
        "corrupted": np.asarray(window["corrupted"], dtype=np.float32),
        "corruption_type": np.asarray(window["corruption_type"], dtype=object),
    }


def _accept_result(trust: float = 1.0) -> MCTSResult:
    return MCTSResult(
        action="accept",
        score=0.0,
        trust=trust,
        visit_counts={name: 0 for name in ("accept", "sanitize")},
        action_values={name: 0.0 for name in ("accept", "sanitize")},
        predicted_deviation=0.0,
        predicted_return_drop=0.0,
    )


def _build_shadow_state(
    actor: Actor,
    critic1: Critic,
    critic2: Critic,
    target1: Critic,
    target2: Critic,
    log_alpha: torch.Tensor,
    device: torch.device,
    config: ExperimentConfig,
    target_entropy: float,
) -> ShadowLearnerState:
    return ShadowLearnerState(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        target1=target1,
        target2=target2,
        log_alpha=log_alpha,
        device=device,
        gamma=config.training.gamma,
        tau=config.training.tau,
        target_entropy=target_entropy,
        lr=config.training.lr,
    )


def run_experiment(config: ExperimentConfig) -> Path:
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

    replay = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=config.training.replay_size)
    windows = WindowBuffer(config.detector.window_length)
    corruption = CorruptionEngine(config.corruption, seed=config.seed + 101)
    detector = HeuristicDetector(threshold=config.detector.trigger_threshold)
    expert_classes = config.experts.classes
    expert_model = ExpertClassifier(input_dim=11, classes=expert_classes)
    controller = LookaheadController(config.controller)
    baseline = BaselineReference(config.controller.baseline_reference_size)
    baseline_captured = False

    detector_labels: list[int] = []
    detector_features: list[np.ndarray] = []
    detector_flags: list[int] = []
    expert_labels: list[str] = []

    accepted_updates = 0
    harmful_accepts = 0
    sanitized_transitions = 0
    attack_steps = 0
    flagged_windows = 0
    flagged_harmful_windows = 0
    intervention_counts = {name: 0 for name in ("accept", "sanitize")}
    sanitize_clean_replay_uses = 0
    captured_suspicious_windows = 0
    captured_harmful_windows = 0
    detector_runtime_ms_total = 0.0
    detector_runtime_calls = 0
    mcts_runtime_ms_total = 0.0
    mcts_runtime_calls = 0
    early_stopping = build_early_stopping_monitor(config.training)
    early_stopped = False
    stop_reason = ""
    completed_steps = 0

    for step in range(config.training.total_steps):
        completed_steps = step + 1
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

        if step < config.corruption.start_step or not corrupted.corrupted:
            baseline.update_memory(next_obs)
        if (
            config.controller.enabled
            and not baseline_captured
            and step + 1 >= config.controller.baseline_warmup_steps
        ):
            baseline.capture_actor(actor)
            baseline_captured = True

        detector_risk = 0.0
        detector_flagged = False
        dominant_label = transition.corruption_type
        expert_prediction = None
        decision = _accept_result()
        reward_shift = abs(corrupted.rew - reward)
        action_shift = float(np.linalg.norm(corrupted.act - action))
        obs_shift = float(np.linalg.norm(corrupted.obs - next_obs))
        decision_log_idx: int | None = None

        if windows.ready() and config.detector.enabled:
            window = windows.as_dict()
            detector_started = perf_counter()
            detector_result = detector.detect(window, config.detector.features)
            if step < config.detector.warmup_steps:
                detector.update_baseline(detector_result.features)
                detector_result = detector.detect(window, config.detector.features)
            detector_runtime_ms = (perf_counter() - detector_started) * 1000.0
            detector_runtime_ms_total += detector_runtime_ms
            detector_runtime_calls += 1
            detector_risk = detector_result.risk
            detector_flagged = bool(detector_result.is_flagged)
            detector_labels.append(int(window["corrupted"].max() > 0))
            detector_flags.append(int(detector_flagged))
            detector_features.append(detector_result.features)
            dominant_label = str(pd.Series(window["corruption_type"]).mode().iloc[0])
            expert_labels.append(dominant_label)

            # The optional expert classifier is a lightweight advisory module over
            # detector-window features. It does not replace the detector or MCTS.
            if detector_flagged and config.experts.enabled:
                expert_prediction = expert_model.predict(detector_result.features)

            if detector_flagged and config.controller.enabled and baseline.ready():
                flagged_batch = _window_to_batch(window)
                clean_replay_size = replay.clean_size()
                clean_batch = replay.sample_clean(len(flagged_batch["obs"]), rng) if replay.size > 0 else flagged_batch
                learner_state = _build_shadow_state(
                    actor=actor,
                    critic1=critic1,
                    critic2=critic2,
                    target1=target1,
                    target2=target2,
                    log_alpha=log_alpha,
                    device=device,
                    config=config,
                    target_entropy=target_entropy,
                )
                mcts_started = perf_counter()
                decision = controller.decide(
                    learner_state=learner_state,
                    baseline=baseline,
                    ctx=LookaheadContext(
                        detector_risk=detector_risk,
                        flagged_batch=flagged_batch,
                        clean_batch=clean_batch,
                        reference_obs=flagged_batch["obs"],
                        reward_shift=reward_shift,
                        action_shift=action_shift,
                        obs_shift=obs_shift,
                        replay_size=replay.size,
                        clean_replay_size=clean_replay_size,
                        replay_clean_fraction=clean_replay_size / max(1, replay.size),
                        global_step=step,
                    ),
                )
                mcts_runtime_ms = (perf_counter() - mcts_started) * 1000.0
                mcts_runtime_ms_total += mcts_runtime_ms
                mcts_runtime_calls += 1
                mcts_payload = {
                    **telemetry.sample(),
                    "global_step": step,
                    "env_id": config.env.id,
                    "seed": config.seed,
                    "schedule": config.corruption.schedule,
                    "poison_type": config.corruption.type,
                    "attack_budget": config.corruption.poison_budget,
                    "attack_budget_unit": config.corruption.poison_budget_unit,
                    "attacker_capability": config.corruption.attacker_capability,
                    "attack_surface": config.corruption.attack_surface,
                    "detector_risk": detector_risk,
                    "selected_action": decision.action,
                    "score": decision.score,
                    "trust": decision.trust,
                    "predicted_deviation": decision.predicted_deviation,
                    "predicted_return_drop": decision.predicted_return_drop,
                    "reward_shift": reward_shift,
                    "action_shift": action_shift,
                    "obs_shift": obs_shift,
                    "replay_size": replay.size,
                    "clean_replay_size": clean_replay_size,
                    "replay_clean_fraction": clean_replay_size / max(1, replay.size),
                    "detector_runtime_ms": detector_runtime_ms,
                    "mcts_runtime_ms": mcts_runtime_ms,
                    "sanitize_replay_mode": config.controller.sanitize_replay_mode,
                }
                for action_name, visits in decision.visit_counts.items():
                    mcts_payload[f"visits_{action_name}"] = visits
                for action_name, value in decision.action_values.items():
                    mcts_payload[f"value_{action_name}"] = value
                recorder.log_mcts_trace(mcts_payload)
            else:
                decision = _accept_result(trust=max(0.0, 1.0 - detector_risk))
                mcts_runtime_ms = 0.0

            recorder.log_decision(
                {
                    **telemetry.sample(),
                    "global_step": step,
                    "env_id": config.env.id,
                    "seed": config.seed,
                    "schedule": config.corruption.schedule,
                    "poison_type": config.corruption.type,
                    "attack_budget": config.corruption.poison_budget,
                    "attack_budget_unit": config.corruption.poison_budget_unit,
                    "attacker_capability": config.corruption.attacker_capability,
                    "attack_surface": config.corruption.attack_surface,
                    "detector_risk": detector_risk,
                    "detector_runtime_ms": detector_runtime_ms,
                    "mcts_runtime_ms": mcts_runtime_ms,
                    "detector_flagged": int(detector_flagged),
                    "corruption_type": dominant_label,
                    "predicted_type": expert_prediction.label if expert_prediction else "clean",
                    "prediction_confidence": expert_prediction.confidence if expert_prediction else 0.0,
                    "controller_action": decision.action,
                    "controller_score": decision.score,
                    "trust": decision.trust,
                    "predicted_deviation": decision.predicted_deviation,
                    "predicted_return_drop": decision.predicted_return_drop,
                    "reward_shift": reward_shift,
                    "action_shift": action_shift,
                    "obs_shift": obs_shift,
                    "sanitize_replay_mode": config.controller.sanitize_replay_mode,
                    "clean_replay_available": int(replay.clean_size() > 0),
                    "used_clean_only_replay": 0,
                }
            )
            decision_log_idx = len(recorder.decisions) - 1
            window_has_corruption = int(window["corrupted"].max() > 0)
            if detector_flagged:
                flagged_windows += 1
                flagged_harmful_windows += window_has_corruption
                captured_suspicious_windows += 1
                captured_harmful_windows += window_has_corruption
            intervention_counts[decision.action] += 1

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
                "env_id": config.env.id,
                "seed": config.seed,
                "schedule": config.corruption.schedule,
                "poison_type": config.corruption.type,
                "window_length": config.detector.window_length,
                "label": window_has_corruption,
                "detector_flag": int(detector_flagged),
                "controller_action": decision.action,
                "expert_capture": int(detector_flagged),
                "detector_runtime_ms": detector_runtime_ms,
            }
            for name, value in zip(WINDOW_FEATURE_NAMES, detector_result.features):
                window_payload[f"window_{name}"] = float(value)
            for name, value in zip(STEP_FEATURE_NAMES, single_features):
                window_payload[f"step_{name}"] = float(value)
            if config.logging.save_transition_windows:
                recorder.log_detector_window(window_payload)
            if detector_flagged:
                recorder.log_captured_window(window_payload)

        should_store = True
        if (
            config.controller.enabled
            and config.controller.sanitize_replay
            and detector_flagged
            and decision.action == "sanitize"
        ):
            should_store = False
            sanitized_transitions += 1

        if should_store:
            replay.store(transition)

        if replay.size >= config.training.batch_size:
            train_batch = replay.sample(config.training.batch_size, rng)
            harmful_update = bool(transition.corrupted and decision.action == "accept")
            clean_replay_available = replay.clean_size() > 0
            used_clean_only_replay = 0

            if decision.action == "sanitize" and clean_replay_available:
                train_batch = replay.sample_clean(config.training.batch_size, rng)
                sanitize_clean_replay_uses += 1
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
            stop_decision = early_stopping.update(step=step + 1, eval_return=eval_return_mean)
            harmful_accept_rate = harmful_accepts / max(1, accepted_updates)
            recorder.log_metric(
                {
                    **telemetry.sample(),
                    "run_name": run_name,
                    "env_id": config.env.id,
                    "seed": config.seed,
                    "global_step": step + 1,
                    "target_steps": config.training.total_steps,
                    "eval_return_mean": eval_return_mean,
                    "early_stopping_enabled": int(config.training.early_stopping_enabled),
                    "early_stopping_min_steps": config.training.early_stopping_min_steps,
                    "early_stopping_patience_evals": config.training.early_stopping_patience_evals,
                    "early_stopping_min_delta": config.training.early_stopping_min_delta,
                    "early_stopping_smoothing_window": config.training.early_stopping_smoothing_window,
                    "early_stopped": int(stop_decision.should_stop),
                    "stop_reason": stop_decision.reason,
                    "early_stopping_best_smoothed_return": stop_decision.best_smoothed_return,
                    "early_stopping_stale_evals": stop_decision.stale_evaluations,
                    "accepted_updates": accepted_updates,
                    "blocked_updates": 0,
                    "sanitized_transitions": sanitized_transitions,
                    "flagged_windows": flagged_windows,
                    "flagged_harmful_windows": flagged_harmful_windows,
                    "captured_suspicious_windows": captured_suspicious_windows,
                    "captured_harmful_windows": captured_harmful_windows,
                    "interventions_accept": intervention_counts["accept"],
                    "interventions_attenuate": 0,
                    "interventions_block": 0,
                    "interventions_sanitize": intervention_counts["sanitize"],
                    "sanitize_clean_replay_uses": sanitize_clean_replay_uses,
                    "attenuate_clean_replay_uses": 0,
                    "attack_steps": attack_steps,
                    "harmful_accept_rate": harmful_accept_rate,
                    "benign_block_rate": 0.0,
                    "policy_backbone": "SACActorMLP256x256TanhGaussian",
                    "reference_actor_role": "clean_policy_reference_snapshot",
                    "experts_enabled": int(config.experts.enabled),
                    "experts_mode": config.experts.mode,
                    "sanitize_replay_mode": config.controller.sanitize_replay_mode,
                    "attack_budget": config.corruption.poison_budget,
                    "attack_budget_unit": config.corruption.poison_budget_unit,
                    "attacker_capability": config.corruption.attacker_capability,
                    "attack_surface": config.corruption.attack_surface,
                    "detector_runtime_ms_total": detector_runtime_ms_total,
                    "detector_runtime_ms_mean": detector_runtime_ms_total / max(1, detector_runtime_calls),
                    "detector_runtime_calls": detector_runtime_calls,
                    "mcts_runtime_ms_total": mcts_runtime_ms_total,
                    "mcts_runtime_ms_mean": mcts_runtime_ms_total / max(1, mcts_runtime_calls),
                    "mcts_runtime_calls": mcts_runtime_calls,
                    "detector_precision": np.nan,
                    "detector_recall": np.nan,
                    "detector_f1": np.nan,
                }
            )
            if stop_decision.should_stop:
                early_stopped = True
                stop_reason = stop_decision.reason
                break

    final_payload = {
        **telemetry.sample(),
        "run_name": run_name,
        "env_id": config.env.id,
        "seed": config.seed,
        "global_step": completed_steps,
        "target_steps": config.training.total_steps,
        "eval_return_mean": evaluate_policy(eval_env, actor, device, config.training.eval_episodes),
        "early_stopping_enabled": int(config.training.early_stopping_enabled),
        "early_stopping_min_steps": config.training.early_stopping_min_steps,
        "early_stopping_patience_evals": config.training.early_stopping_patience_evals,
        "early_stopping_min_delta": config.training.early_stopping_min_delta,
        "early_stopping_smoothing_window": config.training.early_stopping_smoothing_window,
        "early_stopped": int(early_stopped),
        "stop_reason": stop_reason,
        "early_stopping_best_smoothed_return": early_stopping.best_smoothed_return,
        "early_stopping_stale_evals": early_stopping.stale_evaluations,
        "accepted_updates": accepted_updates,
        "blocked_updates": 0,
        "sanitized_transitions": sanitized_transitions,
        "flagged_windows": flagged_windows,
        "flagged_harmful_windows": flagged_harmful_windows,
        "captured_suspicious_windows": captured_suspicious_windows,
        "captured_harmful_windows": captured_harmful_windows,
        "interventions_accept": intervention_counts["accept"],
        "interventions_attenuate": 0,
        "interventions_block": 0,
        "interventions_sanitize": intervention_counts["sanitize"],
        "sanitize_clean_replay_uses": sanitize_clean_replay_uses,
        "attenuate_clean_replay_uses": 0,
        "attack_steps": attack_steps,
        "harmful_accept_rate": harmful_accepts / max(1, accepted_updates),
        "benign_block_rate": 0.0,
        "policy_backbone": "SACActorMLP256x256TanhGaussian",
        "reference_actor_role": "clean_policy_reference_snapshot",
        "experts_enabled": int(config.experts.enabled),
        "experts_mode": config.experts.mode,
        "sanitize_replay_mode": config.controller.sanitize_replay_mode,
        "attack_budget": config.corruption.poison_budget,
        "attack_budget_unit": config.corruption.poison_budget_unit,
        "attacker_capability": config.corruption.attacker_capability,
        "attack_surface": config.corruption.attack_surface,
        "detector_runtime_ms_total": detector_runtime_ms_total,
        "detector_runtime_ms_mean": detector_runtime_ms_total / max(1, detector_runtime_calls),
        "detector_runtime_calls": detector_runtime_calls,
        "mcts_runtime_ms_total": mcts_runtime_ms_total,
        "mcts_runtime_ms_mean": mcts_runtime_ms_total / max(1, mcts_runtime_calls),
        "mcts_runtime_calls": mcts_runtime_calls,
    }

    if detector_features:
        final_payload.update(compute_detection_metrics(detector_labels, detector_flags))

    if detector_features and len(set(expert_labels)) > 1:
        features = np.stack(detector_features)
        labels = np.asarray(expert_labels, dtype=object)
        expert_model = train_supervised_classifier(features, labels, expert_classes)
        metrics = compute_offline_metrics(detector_labels, detector_flags, features, expert_model, labels)
        final_payload.update(metrics)

    recorder.log_metric(final_payload)

    recorder.flush()
    export_summary_row(run_dir)
    env.close()
    eval_env.close()
    return run_dir


def evaluate_policy(env, actor, device, episodes: int) -> float:
    if actor is None:
        return float("nan")
    rets = []
    for _ in range(episodes):
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        done = False
        ep_ret = 0.0
        while not done:
            action = actor.act(obs, device)
            obs, rew, terminated, truncated, _ = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            done = bool(terminated or truncated)
            ep_ret += float(rew)
        rets.append(ep_ret)
    return float(np.mean(rets))


def compute_detection_metrics(detector_labels: list[int], detector_flags: list[int]) -> dict:
    positive = np.asarray(detector_labels, dtype=np.int64)
    predicted_positive = np.asarray(detector_flags, dtype=np.int64)
    tp = int(((positive == 1) & (predicted_positive == 1)).sum())
    fp = int(((positive == 0) & (predicted_positive == 1)).sum())
    fn = int(((positive == 1) & (predicted_positive == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "detector_precision": precision,
        "detector_recall": recall,
        "detector_f1": f1,
    }


def compute_offline_metrics(
    detector_labels: list[int],
    detector_flags: list[int],
    features: np.ndarray,
    expert_model: ExpertClassifier,
    expert_labels: np.ndarray,
) -> dict:
    predicted = []
    for feature in features:
        predicted.append(expert_model.predict(feature).label)
    predicted = np.asarray(predicted, dtype=object)
    cls_acc = float((predicted == expert_labels).mean())
    metrics = compute_detection_metrics(detector_labels, detector_flags)
    metrics["classifier_accuracy"] = cls_acc
    return metrics


def export_summary_row(run_dir: Path) -> None:
    rows = []
    for child in sorted(run_dir.parent.iterdir()):
        metrics_path = child / "metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        summary = df.sort_values("global_step").groupby("global_step", as_index=False).tail(1).tail(1).copy()
        summary["run_dir"] = str(child)
        rows.append(summary)

    out_path = run_dir.parent / "summary.csv"
    if not rows:
        if out_path.exists():
            out_path.unlink()
        return

    pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
