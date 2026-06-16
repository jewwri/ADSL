import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _infer_condition(config: dict) -> str:
    if not config["corruption"]["enabled"]:
        return "clean"
    if config["controller"]["enabled"] and config["controller"].get("sanitize_replay", False):
        return "attack_defended"
    return "attack_none"


def _infer_attack_budget(config: dict) -> float:
    explicit = config.get("corruption", {}).get("poison_budget")
    if explicit is not None:
        return float(explicit)
    if not config.get("corruption", {}).get("enabled", False):
        return 0.0
    schedule = config.get("corruption", {}).get("schedule", "random_sparse")
    if schedule == "random_sparse":
        return float(config.get("corruption", {}).get("random_sparse_p", 0.08))
    burst_length = float(config.get("corruption", {}).get("burst_length", 40))
    burst_period = float(config.get("corruption", {}).get("burst_period", 200))
    return burst_length / max(burst_period, 1.0)


def _load_runs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_rows: list[dict] = []
    curve_frames: list[pd.DataFrame] = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.csv"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue

        try:
            config = json.loads(config_path.read_text())
            metrics = pd.read_csv(metrics_path)
        except Exception:
            continue
        if metrics.empty:
            continue

        metrics = metrics.sort_values("global_step").groupby("global_step", as_index=False).tail(1)
        condition = _infer_condition(config)
        attack_start_step = int(config["corruption"].get("start_step", 0))
        post_attack = metrics[metrics["global_step"] >= attack_start_step].copy()
        evaluation_auc = (
            float(np.trapezoid(post_attack["eval_return_mean"].to_numpy(), post_attack["global_step"].to_numpy()))
            if len(post_attack) >= 2
            else float("nan")
        )

        shared = {
            "run_dir": str(run_dir),
            "experiment_name": config["name"],
            "env_id": config["env"]["id"],
            "seed": int(config["seed"]),
            "schedule": config["corruption"]["schedule"],
            "poison_type": config["corruption"]["type"],
            "condition": condition,
            "attack_start_step": attack_start_step,
            "window_length": int(config["detector"]["window_length"]),
            "detector_threshold": float(config["detector"]["trigger_threshold"]),
            "controller_mode": config["controller"]["mode"],
            "mcts_simulations": int(config["controller"].get("mcts_simulations", 0)),
            "mcts_horizon": int(config["controller"].get("mcts_horizon", 0)),
            "sanitize_replay_mode": config["controller"].get("sanitize_replay_mode", "clean_only_replacement"),
            "target_steps": int(config["training"]["total_steps"]),
            "batch_size": int(config["training"]["batch_size"]),
            "replay_size": int(config["training"]["replay_size"]),
            "attack_budget": _infer_attack_budget(config),
            "attack_budget_unit": config["corruption"].get("poison_budget_unit", "event_fraction"),
            "attacker_capability": config["corruption"].get("attacker_capability", "black_box"),
            "attack_surface": config["corruption"].get("attack_surface", "pre_replay_admission"),
        }

        final = metrics.tail(1).iloc[0].to_dict()
        final_rows.append(
            {
                **shared,
                "run_name": final.get("run_name", run_dir.name),
                "global_step": int(final["global_step"]),
                "completed": int(final["global_step"]) >= shared["target_steps"],
                "final_return": float(final.get("eval_return_mean", float("nan"))),
                "evaluation_auc": evaluation_auc,
                "accepted_updates": float(final.get("accepted_updates", float("nan"))),
                "blocked_updates": float(final.get("blocked_updates", float("nan"))),
                "sanitized_transitions": float(final.get("sanitized_transitions", float("nan"))),
                "flagged_windows": float(final.get("flagged_windows", float("nan"))),
                "flagged_harmful_windows": float(final.get("flagged_harmful_windows", float("nan"))),
                "interventions_accept": float(final.get("interventions_accept", float("nan"))),
                "interventions_attenuate": float(final.get("interventions_attenuate", float("nan"))),
                "interventions_block": float(final.get("interventions_block", float("nan"))),
                "interventions_sanitize": float(final.get("interventions_sanitize", float("nan"))),
                "sanitize_clean_replay_uses": float(final.get("sanitize_clean_replay_uses", float("nan"))),
                "attenuate_clean_replay_uses": float(final.get("attenuate_clean_replay_uses", float("nan"))),
                "attack_steps": float(final.get("attack_steps", float("nan"))),
                "harmful_accept_rate": float(final.get("harmful_accept_rate", float("nan"))),
                "benign_block_rate": float(final.get("benign_block_rate", float("nan"))),
                "detector_precision": float(final.get("detector_precision", float("nan"))),
                "detector_recall": float(final.get("detector_recall", float("nan"))),
                "detector_f1": float(final.get("detector_f1", float("nan"))),
                "policy_backbone": final.get("policy_backbone", "SACActorMLP256x256TanhGaussian"),
                "reference_actor_role": final.get("reference_actor_role", "clean_policy_reference_snapshot"),
                "experts_enabled": int(final.get("experts_enabled", 0) or 0),
                "experts_mode": final.get("experts_mode", "none"),
            }
        )

        curve = metrics.copy()
        for key, value in shared.items():
            curve[key] = value
        curve["completed"] = curve["global_step"] >= shared["target_steps"]
        curve_frames.append(curve)

    final_df = pd.DataFrame(final_rows)
    curve_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()

    key_cols = ["env_id", "schedule", "poison_type", "condition", "seed"]
    if not final_df.empty:
        final_df = (
            final_df.sort_values(["completed", "global_step", "run_dir"])
            .drop_duplicates(subset=key_cols, keep="last")
            .reset_index(drop=True)
        )
    if not curve_df.empty:
        winners = final_df[key_cols + ["run_dir"]].copy()
        curve_df = curve_df.merge(winners, on=key_cols + ["run_dir"], how="inner")
        curve_df = curve_df.sort_values(key_cols + ["global_step"]).reset_index(drop=True)
    return final_df, curve_df


def export_visual_datasets(results_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_df, curve_df = _load_runs(results_dir)

    outputs: list[Path] = []
    targets = [
        ("final_run_dataset.csv", final_df, "csv"),
        ("final_run_dataset.parquet", final_df, "parquet"),
        ("eval_curve_dataset.csv", curve_df, "csv"),
        ("eval_curve_dataset.parquet", curve_df, "parquet"),
    ]

    for filename, df, kind in targets:
        path = output_dir / filename
        if kind == "csv":
            df.to_csv(path, index=False)
        else:
            df.to_parquet(path, index=False)
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/dissertation/mcts_poison_runs")
    parser.add_argument("--output-dir", default="results/dissertation/visual_data")
    args = parser.parse_args()

    outputs = export_visual_datasets(Path(args.results_dir), Path(args.output_dir))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
