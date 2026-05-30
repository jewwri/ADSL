import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


WINDOW_STAT_COLS = [
    "window_reward_mean",
    "window_reward_std",
    "window_reward_min",
    "window_reward_max",
    "window_action_norm_mean",
    "window_action_norm_std",
    "window_state_shift_mean",
    "window_state_shift_std",
    "window_state_shift_max",
]
WINDOW_TEMPORAL_COLS = [
    "window_temporal_abs_mean",
    "window_temporal_abs_max",
]
WINDOW_ALL_COLS = WINDOW_STAT_COLS + WINDOW_TEMPORAL_COLS
STEP_COLS = [
    "step_step_reward_abs",
    "step_step_action_norm",
    "step_step_state_shift",
    "step_step_state_abs_mean",
]


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_windows(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in results_dir.rglob("detector_windows.csv"):
        frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError(f"No detector_windows.csv files found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / max(1e-8, precision + recall)


def _train_eval(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    x_t = torch.as_tensor(train_x, dtype=torch.float32)
    y_t = torch.as_tensor(train_y.reshape(-1, 1), dtype=torch.float32)
    x_test = torch.as_tensor(test_x, dtype=torch.float32)

    model = BinaryMLP(train_x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(40):
        logits = model(x_t)
        loss = loss_fn(logits, y_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(x_test)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return (probs >= 0.5).astype(np.int64)


def _evaluate_group(group: pd.DataFrame) -> list[dict]:
    rows = []
    seeds = sorted(group["seed"].unique().tolist())
    for held_out in seeds:
        train = group[group["seed"] != held_out]
        test = group[group["seed"] == held_out]
        if train.empty or test.empty:
            continue
        y_train = train["label"].to_numpy(dtype=np.int64)
        y_test = test["label"].to_numpy(dtype=np.int64)

        feature_sets = {
            "single_step": STEP_COLS,
            "statistical_only": WINDOW_STAT_COLS,
            "temporal_only": WINDOW_TEMPORAL_COLS,
            "hybrid": WINDOW_ALL_COLS,
        }
        for name, cols in feature_sets.items():
            pred = _train_eval(
                train[cols].to_numpy(dtype=np.float32),
                y_train,
                test[cols].to_numpy(dtype=np.float32),
            )
            rows.append(
                {
                    "held_out_seed": held_out,
                    "feature_set": name,
                    "f1": _f1_score(y_test, pred),
                }
            )
    return rows


def build_report(results_dir: Path, output_path: Path) -> Path:
    df = _load_windows(results_dir)
    rows = []
    for (env_id, schedule, poison_type), group in df.groupby(["env_id", "schedule", "poison_type"]):
        eval_rows = _evaluate_group(group)
        for row in eval_rows:
            row["env_id"] = env_id
            row["schedule"] = schedule
            row["poison_type"] = poison_type
            rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "# Detector Hypotheses Report\n\nNot enough held-out seed folds were available to evaluate H1/H2.\n"
        )
        return output_path

    summary = (
        results.groupby(["env_id", "schedule", "poison_type", "feature_set"], as_index=False)
        .agg(mean_f1=("f1", "mean"), std_f1=("f1", "std"))
        .sort_values(["env_id", "schedule", "poison_type", "feature_set"])
    )

    h1 = []
    h2 = []
    for (env_id, schedule, poison_type), group in summary.groupby(["env_id", "schedule", "poison_type"]):
        wide = group.set_index("feature_set")["mean_f1"].to_dict()
        h1_gain = wide["hybrid"] - wide["single_step"]
        h2_gain = wide["hybrid"] - max(wide["statistical_only"], wide["temporal_only"])
        h1.append(
            {
                "env_id": env_id,
                "schedule": schedule,
                "poison_type": poison_type,
                "hybrid_minus_single_step": round(h1_gain, 4),
                "supports_h1": h1_gain >= (0.10 if schedule == "bursty" else 0.05),
            }
        )
        h2.append(
            {
                "env_id": env_id,
                "schedule": schedule,
                "poison_type": poison_type,
                "hybrid_minus_best_subset": round(h2_gain, 4),
                "supports_h2": h2_gain >= 0.05,
            }
        )

    out = []
    out.append("# Detector Hypotheses Report")
    out.append("")
    out.append("## H1")
    out.append("Sequence-window detection should outperform a single-step baseline by at least 0.10 F1 under bursty schedules and 0.05 F1 under random-sparse schedules.")
    out.append("")
    out.append(summary.round(4).to_csv(index=False))
    out.append("")
    out.append("## H1 Support")
    out.append(pd.DataFrame(h1).to_csv(index=False))
    out.append("")
    out.append("## H2")
    out.append("Hybrid statistical-plus-temporal features should outperform statistical-only and temporal-only detectors by at least 0.05 F1.")
    out.append("")
    out.append("## H2 Support")
    out.append(pd.DataFrame(h2).to_csv(index=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/dissertation/detector_runs")
    parser.add_argument("--output", default="results/dissertation/detector_hypotheses_report.md")
    args = parser.parse_args()
    print(build_report(Path(args.results_dir), Path(args.output)))


if __name__ == "__main__":
    main()
