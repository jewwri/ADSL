import _bootstrap  # noqa: F401

import argparse
import json
from datetime import datetime
from pathlib import Path

from adsl.pipelines import run_experiment
from adsl.pipelines_iforest import run_isolation_forest_experiment
from run_dissertation_campaign import build_config as build_mcts_config
from run_iforest_detector_baseline import build_iforest_config


def _slugify(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "campaign"


def _campaign_root(args) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    threshold = "default" if args.detector_threshold is None else str(args.detector_threshold).replace(".", "p")
    label = (
        f"{args.backend}_steps{args.total_steps}_window{args.window_length}_"
        f"threshold{threshold}_{stamp}"
    )
    if args.campaign_name:
        label = f"{_slugify(args.campaign_name)}_{label}"
    return Path(args.output_root) / label


def _write_manifest(path: Path, args, resolved_output_root: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    manifest = vars(args).copy()
    manifest["campaign_root"] = str(resolved_output_root)
    manifest["created_utc"] = datetime.utcnow().isoformat()
    (path / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mcts", "iforest"], default="mcts")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--campaign-name", default=None)
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--window-length", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--detector-threshold", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--envs", nargs="+", default=["HalfCheetah-v4", "Walker2d-v4", "Hopper-v4"])
    parser.add_argument("--schedules", nargs="+", default=["random_sparse", "bursty"])
    parser.add_argument(
        "--poison-types",
        nargs="+",
        default=["reward_poisoning", "action_perturbation", "observation_corruption"],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["clean", "attack_none", "attack_defended"],
        choices=["clean", "attack_none", "attack_defended"],
        help="Used only for backend=mcts.",
    )
    parser.add_argument(
        "--gate-mode",
        choices=["accept", "attenuate", "block", "sanitize"],
        default="sanitize",
        help="Used only for backend=iforest.",
    )
    parser.add_argument("--iforest-contamination", type=float, default=0.05)
    parser.add_argument("--iforest-n-estimators", type=int, default=200)
    parser.add_argument("--iforest-min-fit-windows", type=int, default=128)
    args = parser.parse_args()
    campaign_root = _campaign_root(args)
    _write_manifest(campaign_root, args, campaign_root)

    if args.backend == "mcts":
        for env_id in args.envs:
            for schedule in args.schedules:
                for poison_type in args.poison_types:
                    for condition in args.conditions:
                        for seed in args.seeds:
                            config = build_mcts_config(
                                env_id=env_id,
                                schedule=schedule,
                                poison_type=poison_type,
                                condition=condition,
                                seed=seed,
                                output_root=str(campaign_root),
                                total_steps=args.total_steps,
                            )
                            config.detector.window_length = int(args.window_length)
                            config.detector.warmup_steps = int(args.warmup_steps)
                            if args.detector_threshold is not None:
                                config.detector.trigger_threshold = float(args.detector_threshold)
                            run_dir = run_experiment(config)
                            print(run_dir)
        return

    detector_threshold = 0.5 if args.detector_threshold is None else float(args.detector_threshold)
    for env_id in args.envs:
        for schedule in args.schedules:
            for poison_type in args.poison_types:
                for seed in args.seeds:
                    config = build_iforest_config(
                        env_id=env_id,
                        schedule=schedule,
                        poison_type=poison_type,
                        seed=seed,
                        output_root=str(campaign_root),
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
                        risk_threshold=detector_threshold,
                    )
                    print(run_dir)


if __name__ == "__main__":
    main()
