from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dashboard import collect_metrics
from .utils import ensure_dir


def export_databricks_ready_table(results_dir: str | Path, output_path: str | Path) -> Path:
    df = collect_metrics(results_dir)
    out = Path(output_path)
    ensure_dir(out.parent)
    if df.empty:
        df = pd.DataFrame(columns=["run_name", "env_id", "seed", "global_step", "eval_return_mean"])
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    return out

