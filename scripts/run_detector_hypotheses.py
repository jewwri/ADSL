import _bootstrap  # noqa: F401

import argparse
from dataclasses import replace

from adsl.config import ControllerConfig, LoggingConfig
from adsl.pipelines import run_experiment
from run_dissertation_campaign import build_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="results/dissertation/detector_runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--envs", nargs="+", default=["Pendulum-v1", "HalfCheetah-v4"])
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
                    config = build_config(
                        env_id=env_id,
                        schedule=schedule,
                        poison_type=poison_type,
                        condition="attack_none",
                        seed=seed,
                        output_root=args.output_root,
                    )
                    config.name = f"detector_{env_id.replace('-', '').replace('v', 'v_')}_{poison_type}_{schedule}"
                    config.controller = ControllerConfig(enabled=False, mode="none", harm_threshold=1.0)
                    config.logging = LoggingConfig(save_transition_windows=True, save_model_checkpoints=False)
                    if env_id == "HalfCheetah-v4":
                        config.training = replace(config.training, total_steps=6000, eval_every=1000, eval_episodes=1)
                    else:
                        config.training = replace(config.training, total_steps=6000, eval_every=1000, eval_episodes=2)
                    run_dir = run_experiment(config)
                    print(run_dir)


if __name__ == "__main__":
    main()
