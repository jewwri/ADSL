from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

from .utils import ensure_dir


def collect_metrics(results_dir: str | Path) -> pd.DataFrame:
    frames = []
    for path in Path(results_dir).rglob("metrics.csv"):
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_dashboard(results_dir: str | Path, output_path: str | Path) -> Path:
    df = collect_metrics(results_dir)
    out = Path(output_path)
    ensure_dir(out.parent)
    if df.empty:
        out.write_text("<html><body><h1>No results found</h1></body></html>")
        return out

    figures = []
    if {"global_step", "eval_return_mean", "run_name"}.issubset(df.columns):
        fig = px.line(df, x="global_step", y="eval_return_mean", color="run_name", title="Return")
        figures.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    if {"harmful_accept_rate", "benign_block_rate", "run_name"}.issubset(df.columns):
        latest = df.sort_values("global_step").groupby("run_name").tail(1)
        fig2 = px.scatter(
            latest,
            x="benign_block_rate",
            y="harmful_accept_rate",
            color="run_name",
            title="Control Tradeoff",
        )
        figures.append(fig2.to_html(full_html=False, include_plotlyjs=False))

    table_html = df.tail(25).to_html(index=False)
    html = "<html><body><h1>ADSL Dashboard</h1>" + "".join(figures) + table_html + "</body></html>"
    out.write_text(html)
    return out

