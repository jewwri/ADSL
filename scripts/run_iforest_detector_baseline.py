import _bootstrap  # noqa: F401

import argparse
from dataclasses import replace

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
from adsl.pipelines_iforest import run_isolation_forest_experiment


ENV_PRESETS = {
    "HalfCheetah-v4": TrainingConfig(
        total_steps=200000,
        batch_size=256,
        replay_size=100000,
        start_steps=1000,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        eval_every=5000,
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
        eval_every=5000,
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
        eval_every=5000,
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


def build_iforest_config(
    env_id: str,
    schedule: str,
    poison_type: str,
    seed: int,
    output_root: str,
    *,
    total_steps: int | None = None,
    window_length: int = 200,
    warmup_steps: int = 1000,
) -> ExperimentConfig:
    training = replace(ENV_PRESETS[env_id])
    if total_steps is not None:
        training.total_steps = int(total_steps)

    return ExperimentConfig(
        name=f"iforest_{env_id.replace('-', '').replace('v', 'v_')}_{poison_type}_{schedule}_detector_only",
        seed=seed,
        output_root=output_root,
        env=EnvConfig(id=env_id),
        training=training,
        detector=DetectorConfig(
            enabled=True,
            window_length=window_length,
            trigger_threshold=0.0,
            warmup_steps=warmup_steps,
        ),
        experts=ExpertsConfig(enabled=False, mode="none", classes=["clean", poison_type]),
        controller=ControllerConfig(
            enabled=False,
            mode="none",
            harm_threshold=1.0,
            sanitize_replay=False,
            attenuate_clean_ratio=0.5,
        ),
        corruption=_corruption_for(poison_type, schedule, condition="attack_none"),
        logging=LoggingConfig(save_transition_windows=True, save_model_checkpoints=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="results/dissertation/iforest_detector_runs")
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--window-length", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--iforest-contamination", type=float, default=0.05)
    parser.add_argument("--iforest-n-estimators", type=int, default=200)
    parser.add_argument("--iforest-min-fit-windows", type=int, default=128)
    parser.add_argument(
        "--gate-mode",
        choices=["accept", "attenuate", "block", "sanitize"],
        default="sanitize",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--envs", nargs="+", default=["HalfCheetah-v4", "Walker2d-v4", "Hopper-v4"])
    parser.add_argument("--schedules", nargs="+", default=["random_sparse", "bursty"])
    parser.add_argument(
        "--poison-types",
        nargs="+",
        default=["reward_poisoning", "action_perturbation", "observation_corruption"],
    )
    args = parser.parse_args()

    for env_id in args.envs:
        for schedule in args.schedules:
            for poison_type in args.poison_types:
                for seed in args.seeds:
                    config = build_iforest_config(
                        env_id=env_id,
                        schedule=schedule,
                        poison_type=poison_type,
                        seed=seed,
                        output_root=args.output_root,
                        total_steps=args.total_steps,
                        window_length=args.window_length,
                        warmup_steps=args.warmup_steps,
                    )
                    run_dir = run_isolation_forest_experiment(
                        config,
                        gate_mode=args.gate_mode,
                        contamination=args.iforest_contamination,
                        n_estimators=args.iforest_n_estimators,
                        min_fit_windows=args.iforest_min_fit_windows,
                    )
                    print(run_dir)


if __name__ == "__main__":
    main()
