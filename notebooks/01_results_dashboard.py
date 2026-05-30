# Databricks notebook source
# MAGIC %md
# MAGIC # ADSL Results Dashboard

# COMMAND ----------

import os
import pandas as pd
import plotly.express as px

RESULTS_PATH = os.environ.get("ADSL_RESULTS_PATH", "/dbfs/FileStore/adsl/results")

frames = []
for root, _, files in os.walk(RESULTS_PATH):
    for name in files:
        if name == "metrics.csv":
            frames.append(pd.read_csv(os.path.join(root, name)))

if not frames:
    raise ValueError(f"No metrics.csv files found under {RESULTS_PATH}")

df = pd.concat(frames, ignore_index=True)
display(df)

# COMMAND ----------

if "global_step" in df.columns and "eval_return_mean" in df.columns:
    fig = px.line(
        df,
        x="global_step",
        y="eval_return_mean",
        color="run_name",
        title="Evaluation Return Over Training",
    )
    fig.show()

# COMMAND ----------

summary_cols = [
    c for c in [
        "run_name",
        "env_id",
        "seed",
        "detector_precision",
        "detector_recall",
        "detector_f1",
        "harmful_accept_rate",
        "benign_block_rate",
        "eval_return_mean",
    ] if c in df.columns
]

display(df[summary_cols].drop_duplicates())

