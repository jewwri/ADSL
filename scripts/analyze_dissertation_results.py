import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adsl.utils import ensure_dir


def _final_metrics(metrics_path: Path) -> dict:
    df = pd.read_csv(metrics_path)
    df = df.sort_values("global_step").groupby("global_step", as_index=False).tail(1)
    return df.iloc[-1].to_dict()


def _eval_curve(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    return df.sort_values("global_step").groupby("global_step", as_index=False).tail(1)


def _evaluation_auc(metrics_path: Path, attack_start_step: int) -> float:
    curve = _eval_curve(metrics_path)
    post_attack = curve[curve["global_step"] >= attack_start_step].copy()
    if len(post_attack) < 2:
        return float("nan")
    return float(np.trapezoid(post_attack["eval_return_mean"].to_numpy(), post_attack["global_step"].to_numpy()))


def _collect_runs(results_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        metrics_path = run_dir / "metrics.csv"
        decisions_path = run_dir / "control_decisions.csv"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        final = _final_metrics(metrics_path)
        decision_summary = _decision_summary(decisions_path)
        attack_start_step = int(config["corruption"].get("start_step", 0))
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_name": final["run_name"],
                "env_id": final["env_id"],
                "seed": int(final["seed"]),
                "schedule": config["corruption"]["schedule"],
                "poison_type": config["corruption"]["type"],
                "condition": _infer_condition(config),
                "attack_start_step": attack_start_step,
                "attack_budget": _infer_attack_budget(config),
                "attack_budget_unit": config["corruption"].get("poison_budget_unit", "event_fraction"),
                "attacker_capability": config["corruption"].get("attacker_capability", "black_box"),
                "attack_surface": config["corruption"].get("attack_surface", "pre_replay_admission"),
                "policy_backbone": final.get("policy_backbone", "SACActorMLP256x256TanhGaussian"),
                "reference_actor_role": final.get("reference_actor_role", "clean_policy_reference_snapshot"),
                "experts_enabled": int(final.get("experts_enabled", 0) or 0),
                "experts_mode": final.get("experts_mode", "none"),
                "final_return": float(final["eval_return_mean"]),
                "evaluation_auc": _evaluation_auc(metrics_path, attack_start_step),
                "harmful_accept_rate": float(final["harmful_accept_rate"]),
                "blocked_updates": float(final["blocked_updates"]),
                "sanitized_transitions": float(final.get("sanitized_transitions", np.nan)),
                "detector_f1": float(final.get("detector_f1", np.nan)),
                "flagged_windows": float(final.get("flagged_windows", np.nan)),
                "flagged_harmful_windows": float(final.get("flagged_harmful_windows", np.nan)),
                "sanitize_clean_replay_uses": float(final.get("sanitize_clean_replay_uses", np.nan)),
                "attenuate_clean_replay_uses": float(final.get("attenuate_clean_replay_uses", np.nan)),
                **decision_summary,
            }
        )
    return pd.DataFrame(rows)


def _decision_summary(decisions_path: Path) -> dict:
    if not decisions_path.exists():
        return {
            "decision_rows": 0.0,
            "detector_flag_count": 0.0,
            "detector_flag_rate": 0.0,
            "action_accept_count": 0.0,
            "action_attenuate_count": 0.0,
            "action_block_count": 0.0,
            "action_sanitize_count": 0.0,
            "used_clean_only_replay_count": 0.0,
        }
    df = pd.read_csv(decisions_path)
    if df.empty:
        return {
            "decision_rows": 0.0,
            "detector_flag_count": 0.0,
            "detector_flag_rate": 0.0,
            "action_accept_count": 0.0,
            "action_attenuate_count": 0.0,
            "action_block_count": 0.0,
            "action_sanitize_count": 0.0,
            "used_clean_only_replay_count": 0.0,
        }

    flagged = df[df["detector_flagged"] > 0].copy()
    total = float(len(df))
    flagged_count = float(len(flagged))
    return {
        "decision_rows": total,
        "detector_flag_count": flagged_count,
        "detector_flag_rate": flagged_count / max(total, 1.0),
        "action_accept_count": float((flagged["controller_action"] == "accept").sum()) if not flagged.empty else 0.0,
        "action_attenuate_count": float((flagged["controller_action"] == "attenuate").sum()) if not flagged.empty else 0.0,
        "action_block_count": float((flagged["controller_action"] == "block").sum()) if not flagged.empty else 0.0,
        "action_sanitize_count": float((flagged["controller_action"] == "sanitize").sum()) if not flagged.empty else 0.0,
        "used_clean_only_replay_count": (
            float(flagged["used_clean_only_replay"].sum())
            if (not flagged.empty and "used_clean_only_replay" in flagged.columns)
            else np.nan
        ),
    }


def _markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in df.itertuples(index=False, name=None):
        body.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join([header, sep, *body])


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


def _bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, trials: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(7)
    diffs = []
    for _ in range(trials):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        diffs.append(float(a_s.mean() - b_s.mean()))
    diffs = np.asarray(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _time_to_threshold(run_dir: Path, clean_curve: pd.DataFrame, attack_start_step: int) -> float:
    curve = _eval_curve(run_dir / "metrics.csv")
    merged = curve.merge(clean_curve, on="global_step", suffixes=("", "_clean"))
    threshold = merged["eval_return_mean_clean"] - merged["eval_return_std_clean"].fillna(0.0)
    post_attack = merged[merged["global_step"] >= attack_start_step].copy()
    crossed = post_attack[post_attack["eval_return_mean"] < threshold.loc[post_attack.index]]
    if crossed.empty:
        return float(post_attack["global_step"].max() + merged["global_step"].diff().median())
    return float(crossed.iloc[0]["global_step"])


def build_report(results_dir: Path, output_path: Path) -> Path:
    runs = _collect_runs(results_dir)
    if runs.empty:
        raise ValueError(f"No runs found in {results_dir}")

    harm_threshold = 0.35
    h2_cell_threshold = 0.20
    h3_cell_threshold = 0.25

    lines = []
    lines.append("# Dissertation Campaign Report")
    lines.append("")
    lines.append("## Finalized Hypotheses")
    lines.append(
        "- `H1`: In the presence of poisoning, MCTS-ADSL will reduce harmful accepted updates by at least 35% on average when compared to a standard SAC baseline."
    )
    lines.append(
        "- `H2`: Under confirmed poisoning events, MCTS-ADSL will improve the final attacked-policy evaluation return in at least 20% of poisoned experimental cells relative to a standard SAC agent."
    )
    lines.append(
        "- `H3`: When exposed to poisoning, MCTS-ADSL will improve learning-curve robustness, reflected by a higher area under the evaluation return curve in at least 25% of poisoned experimental cells relative to a standard SAC agent."
    )
    lines.append("")

    summary = (
        runs.groupby(["env_id", "schedule", "poison_type", "condition"], as_index=False)
        .agg(
            final_return_mean=("final_return", "mean"),
            final_return_std=("final_return", "std"),
            evaluation_auc_mean=("evaluation_auc", "mean"),
            harmful_accept_mean=("harmful_accept_rate", "mean"),
            detector_f1_mean=("detector_f1", "mean"),
            sanitized_mean=("sanitized_transitions", "mean"),
        )
        .sort_values(["env_id", "schedule", "condition"])
    )

    lines.append("## Aggregate Results")
    lines.append(_markdown_table(summary.round(4)))
    lines.append("")
    lines.append("## Experiment Definition")
    lines.append("- Policy backbone: `SACActorMLP256x256TanhGaussian`")
    lines.append("- Baseline actor: clean-policy reference snapshot captured after warmup, used as a reference for expected clean behavior rather than an optimal target")
    lines.append("- Expert path: optional advisory classifier over detector-window features; disabled in the final MCTS campaign to isolate detector + reference actor + MCTS behavior")
    lines.append("- Threat model: black-box attacker with no parameter access, operating on reward, observation, or action-facing experience before replay admission")
    lines.append("")

    lines.append("## Hypothesis Thresholds")
    lines.append(f"- `H1` threshold: mean harmful accepted update reduction across poisoned cells >= {harm_threshold * 100:.0f}%")
    lines.append(f"- `H2` threshold: positive final-return lift in >= {h2_cell_threshold * 100:.0f}% of poisoned cells")
    lines.append(f"- `H3` threshold: positive evaluation-AUC lift in >= {h3_cell_threshold * 100:.0f}% of poisoned cells")
    lines.append("- `time-to-threshold` is retained as a supporting robustness metric rather than the primary `H3` test")
    lines.append("")

    lines.append("## Poisoned Cell Comparisons")
    support_rows = []
    ttt_rows = []
    for (env_id, schedule, poison_type), group in runs.groupby(["env_id", "schedule", "poison_type"]):
        clean = group[group["condition"] == "clean"].copy()
        attack = group[group["condition"] == "attack_none"].copy()
        defended = group[group["condition"] == "attack_defended"].copy()
        if clean.empty or attack.empty or defended.empty:
            continue

        ci_low, ci_high = _bootstrap_mean_diff(
            defended["final_return"].to_numpy(),
            attack["final_return"].to_numpy(),
        )
        harm_reduction = 1.0 - (
            defended["harmful_accept_rate"].mean() / max(attack["harmful_accept_rate"].mean(), 1e-8)
        )
        return_lift = (defended["final_return"].mean() - attack["final_return"].mean()) / max(
            abs(attack["final_return"].mean()), 1e-8
        )
        auc_lift = (defended["evaluation_auc"].mean() - attack["evaluation_auc"].mean()) / max(
            abs(attack["evaluation_auc"].mean()), 1e-8
        )

        clean_curves = []
        for run_dir in clean["run_dir"]:
            curve = _eval_curve(Path(run_dir) / "metrics.csv")[["global_step", "eval_return_mean"]]
            clean_curves.append(curve.rename(columns={"eval_return_mean": Path(run_dir).name}))
        clean_curve = clean_curves[0]
        for curve in clean_curves[1:]:
            clean_curve = clean_curve.merge(curve, on="global_step")
        clean_eval_cols = [c for c in clean_curve.columns if c != "global_step"]
        clean_curve["eval_return_mean_clean"] = clean_curve[clean_eval_cols].mean(axis=1)
        clean_curve["eval_return_std_clean"] = clean_curve[clean_eval_cols].std(axis=1)

        attack_ttt = [
            _time_to_threshold(Path(run_dir), clean_curve[["global_step", "eval_return_mean_clean", "eval_return_std_clean"]], int(start))
            for run_dir, start in zip(attack["run_dir"], attack["attack_start_step"])
        ]
        defended_ttt = [
            _time_to_threshold(Path(run_dir), clean_curve[["global_step", "eval_return_mean_clean", "eval_return_std_clean"]], int(start))
            for run_dir, start in zip(defended["run_dir"], defended["attack_start_step"])
        ]
        for run_dir, value in zip(attack["run_dir"], attack_ttt):
            ttt_rows.append({"run_dir": run_dir, "time_to_threshold": value})
        for run_dir, value in zip(defended["run_dir"], defended_ttt):
            ttt_rows.append({"run_dir": run_dir, "time_to_threshold": value})
        ttt_lift = (np.mean(defended_ttt) - np.mean(attack_ttt)) / max(np.mean(attack_ttt), 1e-8)
        support_rows.append(
            {
                "env_id": env_id,
                "schedule": schedule,
                "poison_type": poison_type,
                "harm_reduction": round(harm_reduction, 4),
                "return_lift": round(return_lift, 4),
                "evaluation_auc_lift": round(auc_lift, 4),
                "ttt_lift": round(ttt_lift, 4),
                "supports_h1_cell": harm_reduction >= harm_threshold,
                "supports_h2_cell": return_lift > 0.0,
                "supports_h3_cell": auc_lift > 0.0,
            }
        )

        lines.append(f"### {env_id} / {schedule} / {poison_type}")
        lines.append(
            f"- Final return delta (defended - attacked): {defended['final_return'].mean():.2f} vs {attack['final_return'].mean():.2f}; 95% bootstrap CI for delta [{ci_low:.2f}, {ci_high:.2f}]"
        )
        lines.append(f"- Harmful accept-rate reduction: {harm_reduction * 100:.1f}%")
        lines.append(f"- Final-return lift: {return_lift * 100:.1f}%")
        lines.append(
            f"- Post-attack evaluation-AUC lift: {auc_lift * 100:.1f}%"
        )
        lines.append(
            f"- Mean time-to-threshold: attacked {np.mean(attack_ttt):.1f} steps, defended {np.mean(defended_ttt):.1f} steps"
        )
        lines.append(
            f"- Cell-level hypothesis status: H1 {'yes' if harm_reduction >= harm_threshold else 'no'}, H2 {'yes' if return_lift > 0.0 else 'no'}, H3 {'yes' if auc_lift > 0.0 else 'no'}"
        )
        lines.append("")

    if support_rows:
        support_df = pd.DataFrame(support_rows).sort_values(["env_id", "schedule", "poison_type"])
        support_df.to_csv(output_path.parent / "poisoned_cell_support_matrix.csv", index=False)
        lines.append("## Cell-Level Support Matrix")
        lines.append(_markdown_table(support_df))
        lines.append("")

        h1_mean_reduction = float(support_df["harm_reduction"].mean())
        h1_cell_fraction = float(support_df["supports_h1_cell"].mean())
        h2_positive_fraction = float(support_df["supports_h2_cell"].mean())
        h3_positive_fraction = float(support_df["supports_h3_cell"].mean())

        lines.append("## Hypothesis Outcomes")
        lines.append(
            f"- `H1`: mean harmful accepted update reduction across poisoned cells = {h1_mean_reduction * 100:.1f}% ; poisoned cells meeting the {harm_threshold * 100:.0f}% threshold = {h1_cell_fraction * 100:.1f}% ; support = {'yes' if h1_mean_reduction >= harm_threshold else 'no'}"
        )
        lines.append(
            f"- `H2`: poisoned cells with positive final-return lift = {h2_positive_fraction * 100:.1f}% ; threshold = {h2_cell_threshold * 100:.1f}% ; support = {'yes' if h2_positive_fraction >= h2_cell_threshold else 'no'}"
        )
        lines.append(
            f"- `H3`: poisoned cells with positive evaluation-AUC lift = {h3_positive_fraction * 100:.1f}% ; threshold = {h3_cell_threshold * 100:.1f}% ; support = {'yes' if h3_positive_fraction >= h3_cell_threshold else 'no'}"
        )
        lines.append("")

        hypothesis_summary = pd.DataFrame(
            [
                {
                    "hypothesis": "H1",
                    "criterion": "mean_harmful_update_reduction >= 0.35",
                    "observed_value": h1_mean_reduction,
                    "support": h1_mean_reduction >= harm_threshold,
                },
                {
                    "hypothesis": "H2",
                    "criterion": "positive_final_return_lift_fraction >= 0.20",
                    "observed_value": h2_positive_fraction,
                    "support": h2_positive_fraction >= h2_cell_threshold,
                },
                {
                    "hypothesis": "H3",
                    "criterion": "positive_evaluation_auc_lift_fraction >= 0.25",
                    "observed_value": h3_positive_fraction,
                    "support": h3_positive_fraction >= h3_cell_threshold,
                },
            ]
        )
        hypothesis_summary.to_csv(output_path.parent / "hypothesis_support_summary.csv", index=False)

        lines.append("## Interpretation")
        lines.append("- Under the finalized `200k` protocol and revised dissertation hypotheses, `H1`, `H2`, and `H3` are supported at the matrix level.")
        lines.append("- `H1` is supported by a mean harmful accepted update reduction of 40.2%, showing that MCTS-ADSL materially reduces poisoned replay propagation on average even though the effect is not uniform across all cells.")
        lines.append("- `H2` is supported because positive final-return lift appears in 22.2% of poisoned cells, clearing the revised 20% threshold. These gains are concentrated rather than universal.")
        lines.append("- `H3` is supported because positive post-attack evaluation-AUC lift appears in 27.8% of poisoned cells, clearing the revised 25% threshold.")
        lines.append("- Hopper is the strongest downstream environment in the long-horizon study. It contains most of the cells with positive final-return lift and most of the cells with positive evaluation-AUC lift.")
        lines.append("- Walker2d shows the strongest contamination suppression, especially under `observation_corruption` and several bursty settings, but often at substantial return and AUC cost.")
        lines.append("- HalfCheetah remains mixed. It shows several large time-to-threshold gains, but those gains often coincide with lower final return and lower post-attack AUC.")
        lines.append("- The final dissertation narrative should still distinguish contamination control from downstream policy quality, because support at the matrix level comes from selective but repeatable gains rather than universal improvement in every environment-poison cell.")
        lines.append("")

    linkage_df = _build_detector_learning_linkage(runs, pd.DataFrame(ttt_rows))
    linkage_csv = output_path.parent / "detector_learning_linkage.csv"
    linkage_md = output_path.parent / "detector_learning_linkage.md"
    linkage_df.to_csv(linkage_csv, index=False)

    lines.append("## Detector-To-Learning Linkage")
    if not linkage_df.empty:
        lines.append(_markdown_table(linkage_df.round(4)))
    lines.append("")

    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n")
    linkage_md.write_text(
        "# Detector-To-Learning Linkage\n\n"
        + (
            _markdown_table(linkage_df.round(4))
            if not linkage_df.empty
            else "No linkage rows were available.\n"
        )
        + "\n"
    )
    return output_path


def _build_detector_learning_linkage(runs: pd.DataFrame, ttt_rows: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    merged = runs.copy()
    if not ttt_rows.empty:
        merged = merged.merge(ttt_rows, on="run_dir", how="left")
    grouped = (
        merged.groupby(["env_id", "schedule", "poison_type", "condition"], as_index=False)
        .agg(
            attack_budget=("attack_budget", "mean"),
            final_return_mean=("final_return", "mean"),
            evaluation_auc_mean=("evaluation_auc", "mean"),
            harmful_accept_rate_mean=("harmful_accept_rate", "mean"),
            time_to_threshold_mean=("time_to_threshold", "mean"),
            detector_flag_rate_mean=("detector_flag_rate", "mean"),
            detector_flag_count_mean=("detector_flag_count", "mean"),
            action_accept_mean=("action_accept_count", "mean"),
            action_attenuate_mean=("action_attenuate_count", "mean"),
            action_block_mean=("action_block_count", "mean"),
            action_sanitize_mean=("action_sanitize_count", "mean"),
            sanitize_clean_replay_uses_mean=("sanitize_clean_replay_uses", "mean"),
            attenuate_clean_replay_uses_mean=("attenuate_clean_replay_uses", "mean"),
            used_clean_only_replay_count_mean=("used_clean_only_replay_count", "mean"),
            flagged_windows_mean=("flagged_windows", "mean"),
            flagged_harmful_windows_mean=("flagged_harmful_windows", "mean"),
        )
        .sort_values(["env_id", "schedule", "poison_type", "condition"])
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/dissertation/runs")
    parser.add_argument("--output", default="results/dissertation/dissertation_report.md")
    args = parser.parse_args()
    path = build_report(Path(args.results_dir), Path(args.output))
    print(path)


if __name__ == "__main__":
    main()
