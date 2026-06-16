import _bootstrap  # noqa: F401

import argparse
import json
from dataclasses import replace
from pathlib import Path

from adsl.config import (
    ControllerConfig,
    CorruptionConfig,
    DetectorConfig,
    EnvConfig,
    ExperimentConfig,
    ExpertsConfig,
    LoggingConfig,
    TrainingConfig,
)
from adsl.pipelines import run_experiment


ENV_PRESETS = {
    "HalfCheetah-v4": TrainingConfig(
        total_steps=200000,
        batch_size=256,
        replay_size=100000,
        start_steps=1000,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        eval_every=1000,
        eval_episodes=2,
    ),
    "Walker2d-v4": TrainingConfig(
        total_steps=200000,
        batch_size=256,
        replay_size=100000,
        start_steps=1000,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        eval_every=1000,
        eval_episodes=2,
    ),
    "Hopper-v4": TrainingConfig(
        total_steps=200000,
        batch_size=256,
        replay_size=100000,
        start_steps=1000,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        eval_every=1000,
        eval_episodes=2,
    ),
}


def _corruption_for(poison_type: str, schedule: str, condition: str) -> CorruptionConfig:
    budget = 0.08 if schedule == "random_sparse" else (40 / 200)
    common = {
        "enabled": condition != "clean",
        "type": poison_type,
        "schedule": schedule,
        "severity": "medium",
        "start_step": 1000,
        "attacker_capability": "black_box",
        "attacker_parameter_access": False,
        "attack_surface": "pre_replay_admission",
        "poison_budget": budget if condition != "clean" else 0.0,
        "poison_budget_unit": "event_fraction",
    }

    if poison_type == "reward_poisoning":
        common.update(
            {
                "reward_flip_p": 1.0,
                "observation_noise_scale": 0.0,
                "action_noise_scale": 0.0,
            }
        )
    elif poison_type == "action_perturbation":
        common.update(
            {
                "reward_flip_p": 0.0,
                "observation_noise_scale": 0.0,
                "action_noise_scale": 0.35 if schedule == "random_sparse" else 0.45,
            }
        )
    elif poison_type == "observation_corruption":
        common.update(
            {
                "reward_flip_p": 0.0,
                "observation_noise_scale": 0.30 if schedule == "random_sparse" else 0.40,
                "action_noise_scale": 0.0,
            }
        )

    if schedule == "random_sparse":
        common["random_sparse_p"] = 0.08
    else:
        common["burst_length"] = 40
        common["burst_period"] = 200

    return CorruptionConfig(**common)


def build_config(
    env_id: str,
    schedule: str,
    poison_type: str,
    condition: str,
    seed: int,
    output_root: str,
    total_steps: int | None = None,
) -> ExperimentConfig:
    training = replace(ENV_PRESETS[env_id])
    if total_steps is not None:
        training.total_steps = int(total_steps)
    detector = DetectorConfig(
        enabled=True,
        window_length=50,
        trigger_threshold=0.12 if schedule == "random_sparse" else 0.14,
        warmup_steps=1000,
    )
    # The retained dissertation matrix isolates detector + reference actor + MCTS
    # behavior. The advisory expert classifier is disabled in the main matrix.
    experts = ExpertsConfig(enabled=False, mode="none", classes=["clean", poison_type])
    controller = ControllerConfig(
        enabled=condition == "attack_defended",
        mode="mcts" if condition == "attack_defended" else "none",
        harm_threshold=0.5,
        sanitize_replay=condition == "attack_defended",
        mcts_simulations=16,
        mcts_horizon=2,
        mcts_exploration_c=1.4,
        baseline_warmup_steps=1000,
        baseline_reference_size=128,
        deviation_threshold=0.15,
        sanitize_replay_mode="clean_only_replacement",
    )
    corruption = _corruption_for(poison_type, schedule, condition)

    return ExperimentConfig(
        name=f"dissertation_{env_id.replace('-', '').replace('v', 'v_')}_{poison_type}_{schedule}_{condition}",
        seed=seed,
        output_root=output_root,
        env=EnvConfig(id=env_id),
        training=training,
        detector=detector,
        experts=experts,
        controller=controller,
        corruption=corruption,
        logging=LoggingConfig(save_transition_windows=False, save_model_checkpoints=False),
    )


def _apply_early_stopping_args(config: ExperimentConfig, args) -> None:
    config.training.early_stopping_enabled = bool(args.early_stopping)
    config.training.early_stopping_min_steps = int(args.early_stopping_min_steps)
    config.training.early_stopping_patience_evals = int(args.early_stopping_patience_evals)
    config.training.early_stopping_min_delta = float(args.early_stopping_min_delta)
    config.training.early_stopping_smoothing_window = int(args.early_stopping_smoothing_window)
    if args.early_stopping and not config.name.endswith("_isolated_earlystop"):
        config.name = f"{config.name}_isolated_earlystop"


def _resolve_output_root(args) -> str:
    default_root = "results/dissertation/mcts_poison_runs_200k"
    if args.early_stopping and args.output_root == default_root:
        return "results/dissertation/isolated_earlystop_mcts_runs"
    return args.output_root


def _is_complete_run(run_dir: Path, total_steps: int) -> bool:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return False
    try:
        import pandas as pd

        df = pd.read_csv(metrics_path)
    except Exception:
        return False
    if df.empty or "global_step" not in df.columns:
        return False
    return int(df["global_step"].max()) >= total_steps


def _find_matching_completed_run(output_root: str, config: ExperimentConfig) -> Path | None:
    root = Path(output_root)
    if not root.exists():
        return None
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        try:
            raw = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if (
            raw.get("env", {}).get("id") == config.env.id
            and raw.get("seed") == config.seed
            and raw.get("corruption", {}).get("schedule") == config.corruption.schedule
            and raw.get("corruption", {}).get("type") == config.corruption.type
            and raw.get("controller", {}).get("enabled") == config.controller.enabled
            and raw.get("corruption", {}).get("enabled") == config.corruption.enabled
        ):
            if _is_complete_run(run_dir, config.training.total_steps):
                return run_dir
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="results/dissertation/mcts_poison_runs_200k")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--early-stopping-min-steps", type=int, default=100000)
    parser.add_argument("--early-stopping-patience-evals", type=int, default=25)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.01)
    parser.add_argument("--early-stopping-smoothing-window", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--envs", nargs="+", default=["HalfCheetah-v4", "Walker2d-v4", "Hopper-v4"])
    parser.add_argument("--schedules", nargs="+", default=["random_sparse", "bursty"])
    parser.add_argument(
        "--poison-types",
        nargs="+",
        default=["reward_poisoning", "action_perturbation", "observation_corruption"],
    )
    args = parser.parse_args()
    output_root = _resolve_output_root(args)

    for env_id in args.envs:
        for schedule in args.schedules:
            for poison_type in args.poison_types:
                for condition in ["clean", "attack_none", "attack_defended"]:
                    for seed in args.seeds:
                        config = build_config(
                            env_id,
                            schedule,
                            poison_type,
                            condition,
                            seed,
                            output_root,
                            total_steps=args.total_steps,
                        )
                        _apply_early_stopping_args(config, args)
                        existing = _find_matching_completed_run(output_root, config)
                        if existing is not None:
                            print(f"SKIP {existing}")
                            continue
                        run_dir = run_experiment(config)
                        print(run_dir)


if __name__ == "__main__":
    main()
