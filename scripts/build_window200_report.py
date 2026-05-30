import _bootstrap  # noqa: F401

import argparse
import csv
import json
import os
import re
from pathlib import Path

from analyze_dissertation_results import build_report
from export_visual_datasets import export_visual_datasets


RUN_STAMP_RE = re.compile(r"_(\d{8}T\d{6})$")


def _condition_from_config(raw: dict) -> str:
    if raw.get("controller", {}).get("enabled"):
        return "attack_defended"
    if raw.get("corruption", {}).get("enabled"):
        return "attack_none"
    return "clean"


def _is_complete(metrics_path: Path, total_steps: int) -> bool:
    if not metrics_path.exists():
        return False
    try:
        with metrics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        max_step = max((int(float(row.get("global_step") or 0)) for row in rows), default=0)
    except Exception:
        return False
    return max_step >= total_steps


def _run_stamp(path: Path) -> str:
    match = RUN_STAMP_RE.search(path.name)
    if match:
        return match.group(1)
    return ""


def _collect_window200_runs(results_root: Path, total_steps: int) -> tuple[dict, list[dict]]:
    winners: dict[tuple, dict] = {}
    all_candidates: list[dict] = []

    for cfg_path in results_root.rglob("config.json"):
        try:
            raw = json.loads(cfg_path.read_text())
        except Exception:
            continue

        detector = raw.get("detector", {})
        training = raw.get("training", {})
        if detector.get("window_length") != 200 or training.get("total_steps") != total_steps:
            continue

        run_dir = cfg_path.parent
        metrics_path = run_dir / "metrics.csv"
        if not _is_complete(metrics_path, total_steps):
            continue

        record = {
            "run_dir": run_dir.resolve(),
            "env_id": raw["env"]["id"],
            "schedule": raw["corruption"]["schedule"],
            "poison_type": raw["corruption"]["type"],
            "condition": _condition_from_config(raw),
            "seed": int(raw["seed"]),
            "window_length": int(detector["window_length"]),
            "trigger_threshold": float(detector["trigger_threshold"]),
            "run_stamp": _run_stamp(run_dir),
        }
        key = (
            record["env_id"],
            record["schedule"],
            record["poison_type"],
            record["condition"],
            record["seed"],
        )
        current = winners.get(key)
        if current is None or record["run_stamp"] >= current["run_stamp"]:
            winners[key] = record
        all_candidates.append(record)

    return winners, all_candidates


def _materialize_selected_runs(selected_root: Path, winners: dict[tuple, dict]) -> Path:
    selected_root.mkdir(parents=True, exist_ok=True)
    for child in selected_root.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            for nested in child.iterdir():
                if nested.is_symlink() or nested.is_file():
                    nested.unlink()
            child.rmdir()

    for record in sorted(winners.values(), key=lambda row: row["run_dir"].name):
        link_path = selected_root / record["run_dir"].name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(record["run_dir"], link_path)
    return selected_root


def _write_selection_manifest(output_root: Path, winners: dict[tuple, dict], candidates: list[dict]) -> Path:
    manifest = {
        "window_length": 200,
        "target_steps": 200000,
        "selected_runs": len(winners),
        "complete_candidates": len(candidates),
        "duplicate_candidates_removed": len(candidates) - len(winners),
        "source_root": "results/dissertation/parameterized_runs",
        "selection_rule": "latest completed run per (env_id, schedule, poison_type, condition, seed)",
        "runs": [
            {
                "env_id": row["env_id"],
                "schedule": row["schedule"],
                "poison_type": row["poison_type"],
                "condition": row["condition"],
                "seed": row["seed"],
                "run_dir": str(row["run_dir"]),
                "trigger_threshold": row["trigger_threshold"],
                "run_stamp": row["run_stamp"],
            }
            for row in sorted(
                winners.values(),
                key=lambda row: (row["env_id"], row["schedule"], row["poison_type"], row["condition"], row["seed"]),
            )
        ],
    }
    path = output_root / "window200_selection_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def _write_readmes(output_root: Path, selected_root: Path, winners: dict[tuple, dict], candidates: list[dict]) -> list[Path]:
    duplicate_count = len(candidates) - len(winners)
    thresholds = sorted({row["trigger_threshold"] for row in winners.values()})
    threshold_text = ", ".join(f"{value:.2f}" for value in thresholds)

    artifact_readme = output_root / "README_window200.md"
    artifact_readme.write_text(
        "\n".join(
            [
                "# Window-200 Experiment Artifacts",
                "",
                "This directory contains the deduplicated `window_length=200` rerun analysis for the `200,000`-step dissertation matrix.",
                "",
                "## Selection Rule",
                "",
                "- Source root: `results/dissertation/parameterized_runs`",
                "- Included only completed runs with `window_length=200` and `total_steps=200000`",
                "- Selected exactly one completed run per `(env_id, schedule, poison_type, condition, seed)` cell",
                "- Tie-break rule: latest timestamped run directory wins",
                "",
                "## Selection Summary",
                "",
                f"- Completed candidate runs discovered: `{len(candidates)}`",
                f"- Final unique logical cells selected: `{len(winners)}`",
                f"- Duplicate completed runs removed during deduplication: `{duplicate_count}`",
                f"- Detector thresholds present in the selected set: `{threshold_text}`",
                "",
                "## Key Outputs",
                "",
                "- `window200_final_report.md`",
                "- `window200_selection_manifest.json`",
                "- `hypothesis_support_summary.csv`",
                "- `poisoned_cell_support_matrix.csv`",
                "- `visual_data_window200/`",
                "- `window200_selected_runs/`",
                "",
                "## Notes",
                "",
                "- Existing `window=50` reports and READMEs were left untouched.",
                "- The selected runs directory is a symlinked staging view over the chosen source runs, not a second copy of the raw artifacts.",
                "",
            ]
        )
        + "\n"
    )

    docs_readme = Path("docs/README_WINDOW200.md")
    docs_readme.write_text(
        "\n".join(
            [
                "# Window-200 Rerun Notes",
                "",
                "This document tracks the long-horizon rerun that preserved the `200,000`-step dissertation matrix while changing the detector window length from `50` to `200`.",
                "",
                "## Purpose",
                "",
                "- evaluate whether the shorter `window=50` detector was over-granular for long-horizon training",
                "- preserve the original attack schedules, environments, poison types, seeds, and total training horizon",
                "- compare the resulting `window=200` matrix against the original `window=50` MCTS dissertation run",
                "",
                "## Shared Experimental Constants",
                "",
                "- environments: `HalfCheetah-v4`, `Walker2d-v4`, `Hopper-v4`",
                "- schedules: `random_sparse`, `bursty`",
                "- poison types: `reward_poisoning`, `action_perturbation`, `observation_corruption`",
                "- conditions: `clean`, `attack_none`, `attack_defended`",
                "- seeds: `0-4`",
                "- total steps: `200000`",
                "",
                "## Detector Configuration Change",
                "",
                "- changed detector `window_length` from `50` to `200`",
                "- preserved the original schedule-specific thresholds in the run configs",
                "- `random_sparse`: `0.12`",
                "- `bursty`: `0.14`",
                "",
                "## Reporting Location",
                "",
                f"- artifact README: `{artifact_readme}`",
                f"- selected runs: `{selected_root}`",
                f"- final report: `{output_root / 'window200_final_report.md'}`",
                "",
            ]
        )
        + "\n"
    )
    return [artifact_readme, docs_readme]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/dissertation/parameterized_runs")
    parser.add_argument("--output-root", default="results/dissertation/window200_artifacts")
    parser.add_argument("--total-steps", type=int, default=200000)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    winners, candidates = _collect_window200_runs(results_root, args.total_steps)
    if len(winners) != 270:
        raise ValueError(f"Expected 270 unique completed window=200 cells, found {len(winners)}")

    selected_root = _materialize_selected_runs(output_root / "window200_selected_runs", winners)
    manifest = _write_selection_manifest(output_root, winners, candidates)
    report = build_report(selected_root, output_root / "window200_final_report.md")
    visual_outputs = export_visual_datasets(selected_root, output_root / "visual_data_window200")
    readmes = _write_readmes(output_root, selected_root, winners, candidates)

    print(manifest)
    print(report)
    for path in visual_outputs:
        print(path)
    for path in readmes:
        print(path)


if __name__ == "__main__":
    main()
