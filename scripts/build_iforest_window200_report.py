import _bootstrap  # noqa: F401

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from adsl.utils import ensure_dir


def _infer_mcts_condition(config: dict) -> str:
    if not config["corruption"]["enabled"]:
        return "clean"
    if config["controller"]["enabled"]:
        return "attack_defended"
    return "attack_none"


def _run_stamp(path: Path) -> str:
    name = path.name
    parts = name.rsplit("_", 1)
    return parts[-1] if len(parts) == 2 else ""


def _load_metrics_curve(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    return df.sort_values("global_step").groupby("global_step", as_index=False).tail(1)


def _load_final_metrics(metrics_path: Path) -> dict:
    return _load_metrics_curve(metrics_path).iloc[-1].to_dict()


def _evaluation_auc(metrics_path: Path, attack_start_step: int) -> float:
    curve = _load_metrics_curve(metrics_path)
    post_attack = curve[curve["global_step"] >= attack_start_step].copy()
    if len(post_attack) < 2:
        return float("nan")
    return float(np.trapezoid(post_attack["eval_return_mean"].to_numpy(), post_attack["global_step"].to_numpy()))


def _time_to_threshold(metrics_path: Path, clean_curve: pd.DataFrame, attack_start_step: int) -> float:
    curve = _load_metrics_curve(metrics_path)
    merged = curve.merge(clean_curve, on="global_step", suffixes=("", "_clean"))
    threshold = merged["eval_return_mean_clean"] - merged["eval_return_std_clean"].fillna(0.0)
    post_attack = merged[merged["global_step"] >= attack_start_step].copy()
    crossed = post_attack[post_attack["eval_return_mean"] < threshold.loc[post_attack.index]]
    if crossed.empty:
        return float(post_attack["global_step"].max() + merged["global_step"].diff().median())
    return float(crossed.iloc[0]["global_step"])


def _common_record(run_dir: Path, config: dict, condition: str) -> dict:
    final = _load_final_metrics(run_dir / "metrics.csv")
    return {
        "run_dir": run_dir.resolve(),
        "run_name": final.get("run_name", run_dir.name),
        "env_id": config["env"]["id"],
        "seed": int(config["seed"]),
        "schedule": config["corruption"]["schedule"],
        "poison_type": config["corruption"]["type"],
        "condition": condition,
        "attack_start_step": int(config["corruption"].get("start_step", 0)),
        "window_length": int(config["detector"]["window_length"]),
        "detector_threshold": float(config["detector"]["trigger_threshold"]),
        "final_return": float(final.get("eval_return_mean", float("nan"))),
        "evaluation_auc": _evaluation_auc(run_dir / "metrics.csv", int(config["corruption"].get("start_step", 0))),
        "harmful_accept_rate": float(final.get("harmful_accept_rate", float("nan"))),
        "blocked_updates": float(final.get("blocked_updates", float("nan"))),
        "flagged_windows": float(final.get("flagged_windows", float("nan"))),
        "flagged_harmful_windows": float(final.get("flagged_harmful_windows", float("nan"))),
        "interventions_accept": float(final.get("interventions_accept", float("nan"))),
        "interventions_attenuate": float(final.get("interventions_attenuate", float("nan"))),
        "interventions_block": float(final.get("interventions_block", float("nan"))),
        "interventions_sanitize": float(final.get("interventions_sanitize", float("nan"))),
        "sanitize_clean_replay_uses": float(final.get("sanitize_clean_replay_uses", float("nan"))),
        "attenuate_clean_replay_uses": float(final.get("attenuate_clean_replay_uses", float("nan"))),
        "completed": int(float(final.get("global_step", 0))) >= int(config["training"]["total_steps"]),
        "run_stamp": _run_stamp(run_dir),
    }


def _collect_mcts_reference_runs(results_root: Path) -> dict[tuple, dict]:
    winners: dict[tuple, dict] = {}
    for cfg_path in results_root.glob("*/config.json"):
        raw = json.loads(cfg_path.read_text())
        condition = _infer_mcts_condition(raw)
        if condition not in {"clean", "attack_none"}:
            continue
        run_dir = cfg_path.parent
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        record = _common_record(run_dir, raw, condition)
        key = (record["env_id"], record["schedule"], record["poison_type"], condition, record["seed"])
        current = winners.get(key)
        if current is None or record["run_stamp"] >= current["run_stamp"]:
            winners[key] = record
    if len(winners) != 180:
        raise ValueError(f"Expected 180 clean/attack_none MCTS reference runs, found {len(winners)}")
    return winners


def _collect_iforest_runs(results_root: Path, total_steps: int) -> dict[tuple, dict]:
    winners: dict[tuple, dict] = {}
    for cfg_path in results_root.glob("*/config.json"):
        raw = json.loads(cfg_path.read_text())
        run_dir = cfg_path.parent
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        final = _load_final_metrics(metrics_path)
        if int(float(final.get("global_step", 0))) < total_steps:
            continue
        record = _common_record(run_dir, raw, "detector_only")
        key = (record["env_id"], record["schedule"], record["poison_type"], record["seed"])
        current = winners.get(key)
        if current is None or record["run_stamp"] >= current["run_stamp"]:
            winners[key] = record
    if len(winners) != 90:
        raise ValueError(f"Expected 90 complete IForest runs, found {len(winners)}")
    return winners


def _materialize_selected_runs(selected_root: Path, runs: dict[tuple, dict]) -> Path:
    selected_root.mkdir(parents=True, exist_ok=True)
    for child in selected_root.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            for nested in child.iterdir():
                if nested.is_symlink() or nested.is_file():
                    nested.unlink()
            child.rmdir()
    for record in sorted(runs.values(), key=lambda row: Path(row["run_dir"]).name):
        link_path = selected_root / Path(record["run_dir"]).name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(record["run_dir"], link_path)
    return selected_root


def _build_clean_curve(clean_runs: list[dict]) -> pd.DataFrame:
    curves = []
    for record in clean_runs:
        curve = _load_metrics_curve(Path(record["run_dir"]) / "metrics.csv")[["global_step", "eval_return_mean"]]
        curves.append(curve.rename(columns={"eval_return_mean": Path(record["run_dir"]).name}))
    merged = curves[0]
    for curve in curves[1:]:
        merged = merged.merge(curve, on="global_step")
    clean_cols = [c for c in merged.columns if c != "global_step"]
    merged["eval_return_mean_clean"] = merged[clean_cols].mean(axis=1)
    merged["eval_return_std_clean"] = merged[clean_cols].std(axis=1)
    return merged[["global_step", "eval_return_mean_clean", "eval_return_std_clean"]]


def _export_detector_visuals(runs: dict[tuple, dict], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_rows = []
    curve_frames = []
    for record in sorted(runs.values(), key=lambda row: (row["env_id"], row["schedule"], row["poison_type"], row["seed"])):
        config = json.loads((Path(record["run_dir"]) / "config.json").read_text())
        metrics = _load_metrics_curve(Path(record["run_dir"]) / "metrics.csv")
        shared = {
            "run_dir": str(record["run_dir"]),
            "run_name": record["run_name"],
            "env_id": record["env_id"],
            "seed": record["seed"],
            "schedule": record["schedule"],
            "poison_type": record["poison_type"],
            "condition": "detector_only",
            "attack_start_step": record["attack_start_step"],
            "window_length": record["window_length"],
            "detector_threshold": 0.5,
            "controller_mode": "none",
            "mcts_simulations": 0,
            "mcts_horizon": 0,
            "attenuate_replay_mode": config["controller"].get("attenuate_replay_mode", "weighted_mix"),
            "sanitize_replay_mode": config["controller"].get("sanitize_replay_mode", "clean_only_replacement"),
            "attenuate_clean_ratio": float(config["controller"].get("attenuate_clean_ratio", 0.5)),
            "target_steps": int(config["training"]["total_steps"]),
            "batch_size": int(config["training"]["batch_size"]),
            "replay_size": int(config["training"]["replay_size"]),
            "completed": True,
        }
        final_rows.append(
            {
                **shared,
                "final_return": record["final_return"],
                "evaluation_auc": record["evaluation_auc"],
                "accepted_updates": record.get("accepted_updates", np.nan),
                "blocked_updates": record["blocked_updates"],
                "sanitized_transitions": record.get("sanitized_transitions", np.nan),
                "flagged_windows": record["flagged_windows"],
                "flagged_harmful_windows": record["flagged_harmful_windows"],
                "interventions_accept": record["interventions_accept"],
                "interventions_attenuate": record["interventions_attenuate"],
                "interventions_block": record["interventions_block"],
                "interventions_sanitize": record["interventions_sanitize"],
                "sanitize_clean_replay_uses": record["sanitize_clean_replay_uses"],
                "attenuate_clean_replay_uses": record["attenuate_clean_replay_uses"],
                "harmful_accept_rate": record["harmful_accept_rate"],
            }
        )
        curve = metrics.copy()
        for key, value in shared.items():
            curve[key] = value
        curve_frames.append(curve)
    final_df = pd.DataFrame(final_rows)
    curve_df = pd.concat(curve_frames, ignore_index=True)
    outputs = []
    for filename, df in [
        ("final_run_dataset.csv", final_df),
        ("eval_curve_dataset.csv", curve_df),
    ]:
        path = output_dir / filename
        df.to_csv(path, index=False)
        outputs.append(path)
    for filename, df in [
        ("final_run_dataset.parquet", final_df),
        ("eval_curve_dataset.parquet", curve_df),
    ]:
        path = output_dir / filename
        df.to_parquet(path, index=False)
        outputs.append(path)
    return outputs


def _markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *body])


def _write_manifest(output_root: Path, iforest_runs: dict[tuple, dict], mcts_reference: dict[tuple, dict]) -> Path:
    manifest = {
        "window_length": 200,
        "target_steps": 200000,
        "selected_iforest_runs": len(iforest_runs),
        "selected_mcts_reference_runs": len(mcts_reference),
        "source_iforest_root": "results/dissertation/iforest_parameterized_runs/window200_iforest_baseline_iforest_steps200000_window200_thresholddefault_20260507T214909",
        "source_mcts_root": "results/dissertation/window200_artifacts/window200_selected_runs",
        "selection_rule": "latest completed run per logical cell",
        "iforest_runs": [
            {
                "env_id": row["env_id"],
                "schedule": row["schedule"],
                "poison_type": row["poison_type"],
                "seed": row["seed"],
                "run_dir": str(row["run_dir"]),
                "run_stamp": row["run_stamp"],
            }
            for row in sorted(iforest_runs.values(), key=lambda row: (row["env_id"], row["schedule"], row["poison_type"], row["seed"]))
        ],
    }
    path = output_root / "iforest_window200_selection_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def build_iforest_report(iforest_root: Path, mcts_root: Path, output_root: Path, total_steps: int) -> list[Path]:
    ensure_dir(output_root)
    iforest_runs = _collect_iforest_runs(iforest_root, total_steps)
    mcts_reference = _collect_mcts_reference_runs(mcts_root)
    selected_root = _materialize_selected_runs(output_root / "iforest_window200_selected_runs", iforest_runs)
    manifest_path = _write_manifest(output_root, iforest_runs, mcts_reference)
    visual_outputs = _export_detector_visuals(iforest_runs, output_root / "visual_data_iforest_window200")

    support_rows = []
    ttt_rows = []
    for env_id in ["HalfCheetah-v4", "Walker2d-v4", "Hopper-v4"]:
        for schedule in ["random_sparse", "bursty"]:
            for poison_type in ["reward_poisoning", "action_perturbation", "observation_corruption"]:
                clean = [
                    row for row in mcts_reference.values()
                    if row["env_id"] == env_id and row["schedule"] == schedule and row["poison_type"] == poison_type and row["condition"] == "clean"
                ]
                attack = [
                    row for row in mcts_reference.values()
                    if row["env_id"] == env_id and row["schedule"] == schedule and row["poison_type"] == poison_type and row["condition"] == "attack_none"
                ]
                detector = [
                    row for row in iforest_runs.values()
                    if row["env_id"] == env_id and row["schedule"] == schedule and row["poison_type"] == poison_type
                ]
                if not clean or not attack or not detector:
                    continue
                clean_curve = _build_clean_curve(clean)
                attack_ttt = [
                    _time_to_threshold(Path(row["run_dir"]) / "metrics.csv", clean_curve, row["attack_start_step"])
                    for row in attack
                ]
                detector_ttt = [
                    _time_to_threshold(Path(row["run_dir"]) / "metrics.csv", clean_curve, row["attack_start_step"])
                    for row in detector
                ]
                for row, value in zip(detector, detector_ttt):
                    ttt_rows.append({"run_dir": str(row["run_dir"]), "time_to_threshold": value})
                harm_reduction = 1.0 - (
                    np.mean([row["harmful_accept_rate"] for row in detector]) / max(np.mean([row["harmful_accept_rate"] for row in attack]), 1e-8)
                )
                return_lift = (
                    np.mean([row["final_return"] for row in detector]) - np.mean([row["final_return"] for row in attack])
                ) / max(abs(np.mean([row["final_return"] for row in attack])), 1e-8)
                auc_lift = (
                    np.mean([row["evaluation_auc"] for row in detector]) - np.mean([row["evaluation_auc"] for row in attack])
                ) / max(abs(np.mean([row["evaluation_auc"] for row in attack])), 1e-8)
                ttt_lift = (np.mean(detector_ttt) - np.mean(attack_ttt)) / max(np.mean(attack_ttt), 1e-8)
                support_rows.append(
                    {
                        "env_id": env_id,
                        "schedule": schedule,
                        "poison_type": poison_type,
                        "harm_reduction": round(harm_reduction, 4),
                        "return_lift": round(return_lift, 4),
                        "evaluation_auc_lift": round(auc_lift, 4),
                        "ttt_lift": round(ttt_lift, 4),
                        "supports_h1_cell": harm_reduction >= 0.35,
                        "supports_h2_cell": return_lift > 0.0,
                        "supports_h3_cell": auc_lift > 0.0,
                    }
                )

    support_df = pd.DataFrame(support_rows).sort_values(["env_id", "schedule", "poison_type"])
    support_csv = output_root / "poisoned_cell_support_matrix.csv"
    support_df.to_csv(support_csv, index=False)

    h1_mean = float(support_df["harm_reduction"].mean())
    h2_frac = float(support_df["supports_h2_cell"].mean())
    h3_frac = float(support_df["supports_h3_cell"].mean())
    hypothesis_df = pd.DataFrame(
        [
            {
                "hypothesis": "H1",
                "criterion": "mean_harmful_update_reduction >= 0.35",
                "observed_value": h1_mean,
                "support": h1_mean >= 0.35,
            },
            {
                "hypothesis": "H2",
                "criterion": "positive_final_return_lift_fraction >= 0.20",
                "observed_value": h2_frac,
                "support": h2_frac >= 0.20,
            },
            {
                "hypothesis": "H3",
                "criterion": "positive_evaluation_auc_lift_fraction >= 0.25",
                "observed_value": h3_frac,
                "support": h3_frac >= 0.25,
            },
        ]
    )
    hypothesis_csv = output_root / "hypothesis_support_summary.csv"
    hypothesis_df.to_csv(hypothesis_csv, index=False)

    mcts_hypothesis_csv = Path("results/dissertation/window200_artifacts/hypothesis_support_summary.csv")
    mcts_hyp = pd.read_csv(mcts_hypothesis_csv).rename(
        columns={"observed_value": "mcts_observed_value", "support": "mcts_support"}
    )
    iforest_hyp = hypothesis_df.rename(
        columns={"observed_value": "iforest_observed_value", "support": "iforest_support"}
    )
    comparison_df = iforest_hyp.merge(mcts_hyp[["hypothesis", "mcts_observed_value", "mcts_support"]], on="hypothesis", how="left")
    comparison_csv = output_root / "iforest_vs_mcts_window200_summary.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    report_lines = [
        "# Detector-Only Window-200 Report",
        "",
        "This report summarizes the completed `window_length=200`, `200,000`-step Isolation Forest detector-only comparison baseline.",
        "",
        "## Sources",
        "",
        f"- detector-only source: `{iforest_root}`",
        f"- MCTS comparison source: `{mcts_root}`",
        "",
        "## Detector-Only Matrix Status",
        "",
        f"- completed detector-only logical runs: `{len(iforest_runs)}`",
        f"- completed clean/attack-none MCTS reference runs: `{len(mcts_reference)}`",
        "",
        "## Hypothesis Outcomes Under Detector-Only Baseline",
        "",
        f"- `H1`: mean harmful accepted update reduction across poisoned cells = `{h1_mean * 100:.1f}%` ; support = `{'yes' if h1_mean >= 0.35 else 'no'}`",
        f"- `H2`: poisoned cells with positive final-return lift = `{h2_frac * 100:.1f}%` ; support = `{'yes' if h2_frac >= 0.20 else 'no'}`",
        f"- `H3`: poisoned cells with positive evaluation-AUC lift = `{h3_frac * 100:.1f}%` ; support = `{'yes' if h3_frac >= 0.25 else 'no'}`",
        "",
        "## Detector-Only vs MCTS Window-200",
        "",
        _markdown_table(comparison_df.round(4)),
        "",
        "## Cell-Level Detector-Only Support Matrix",
        "",
        _markdown_table(support_df.round(4)),
        "",
        "## Key Outputs",
        "",
        "- `iforest_window200_final_report.md`",
        "- `iforest_window200_selection_manifest.json`",
        "- `hypothesis_support_summary.csv`",
        "- `poisoned_cell_support_matrix.csv`",
        "- `iforest_vs_mcts_window200_summary.csv`",
        "- `visual_data_iforest_window200/`",
        "- `iforest_window200_selected_runs/`",
        "",
    ]
    report_path = output_root / "iforest_window200_final_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")

    readme_path = output_root / "README_iforest_window200.md"
    readme_path.write_text(
        "\n".join(
            [
                "# IForest Window-200 Artifacts",
                "",
                "This directory contains the completed detector-only Isolation Forest baseline analysis for the `window_length=200`, `200,000`-step study.",
                "",
                "## Included Artifacts",
                "",
                "- `iforest_window200_final_report.md`",
                "- `iforest_window200_selection_manifest.json`",
                "- `hypothesis_support_summary.csv`",
                "- `poisoned_cell_support_matrix.csv`",
                "- `iforest_vs_mcts_window200_summary.csv`",
                "- `visual_data_iforest_window200/`",
                "- `iforest_window200_selected_runs/`",
                "",
                "## Comparison Convention",
                "",
                "- detector-only runs are compared against the completed `window=200` MCTS clean and attack-none selected runs",
                "- H1/H2/H3 criteria match the existing `results/dissertation/window200_artifacts/hypothesis_support_summary.csv` thresholds",
                "",
            ]
        )
        + "\n"
    )

    return [manifest_path, report_path, readme_path, support_csv, hypothesis_csv, comparison_csv, *visual_outputs, selected_root]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iforest-root",
        default="results/dissertation/iforest_parameterized_runs/window200_iforest_baseline_iforest_steps200000_window200_thresholddefault_20260507T214909",
    )
    parser.add_argument("--mcts-root", default="results/dissertation/window200_artifacts/window200_selected_runs")
    parser.add_argument("--output-root", default="results/dissertation/iforest_window200_artifacts")
    parser.add_argument("--total-steps", type=int, default=200000)
    args = parser.parse_args()

    outputs = build_iforest_report(Path(args.iforest_root), Path(args.mcts_root), Path(args.output_root), args.total_steps)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
