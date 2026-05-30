from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig, load_experiment_config
from .detection import compute_window_features
from .experts import train_supervised_classifier
from .pipelines import run_experiment
from .utils import ensure_dir


def main_experiment() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    run_dir = run_experiment(config)
    print(f"Run complete: {run_dir}")


def _config_from_ablation(row: dict[str, str]) -> ExperimentConfig:
    env_id = row["environment"]
    rl_algorithm = row["rl_algorithm"]
    config = ExperimentConfig(name=row["ablation_id"], seed=0)
    config.env.id = env_id
    config.training.total_steps = 20_000 if env_id == "CartPole-v1" else 50_000
    config.detector.window_length = int(row["window_length"])
    config.detector.trigger_threshold = float(row["detector_threshold"])
    config.experts.mode = row["expert_mode"]
    config.controller.mode = row["controller_mode"]
    config.controller.harm_threshold = float(row["harm_threshold"])
    config.corruption.type = row["corruption_type"]
    config.corruption.schedule = row["attack_schedule"]
    config.corruption.severity = row["corruption_severity"]
    config.corruption.enabled = row["corruption_type"] != "clean"
    if row["class_regime"] == "realistic":
        config.corruption.random_sparse_p = 0.02
    if rl_algorithm == "SAC":
        config.training.batch_size = 256
    return config


def main_ablation() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.matrix).open()))
    if args.phase:
        rows = [row for row in rows if row["phase"] == args.phase]
    if args.limit is not None:
        rows = rows[: args.limit]

    for row in rows:
        config = _config_from_ablation(row)
        run_dir = run_experiment(config)
        print(f"{row['ablation_id']} -> {run_dir}")


def main_train_experts() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    labels = df[args.label_column].astype(str).to_numpy()
    features = df.drop(columns=[args.label_column]).to_numpy(dtype=np.float32)
    classes = sorted(set(labels.tolist()))
    model = train_supervised_classifier(features, labels, classes)
    outdir = ensure_dir(args.output_dir)
    import torch

    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "input_dim": features.shape[1],
        },
        outdir / "expert_classifier.pt",
    )
    print(f"Saved expert classifier to {outdir / 'expert_classifier.pt'}")


def main_dashboard() -> None:
    from .dashboard import build_dashboard

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = build_dashboard(args.results_dir, args.output)
    print(f"Dashboard written to {path}")


def main_export_databricks() -> None:
    from .databricks import export_databricks_ready_table

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = export_databricks_ready_table(args.results_dir, args.output)
    print(f"Export written to {path}")
