# Reproducibility Guide

## Canonical Artifact Boundary

This repository is intentionally scoped to the dissertation experiments listed in
`experiments/canonical_experiments.json`.
Column-level definitions are in `docs/DATA_DICTIONARY.md`.

Canonical retained result families:

- `results/dissertation/mcts_poison_runs_200k/`
- `results/dissertation/telemetry_runs/`
- `results/dissertation/iforest_parameterized_runs/window200_iforest_baseline_iforest_steps200000_window200_thresholddefault_20260507T214909/`
- derived reports and visual datasets listed in the manifest

Everything under `results/ignored_archive/` is retained locally for provenance, but ignored by Git. It contains superseded smoke runs, old pilot material, generated caches, and prior result trees that are not part of the final doctoral analysis.

## Environment

Use Python `>=3.10`. Install the project and optional extras needed for the retained experiments:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[dev,detectors,mujoco]"
```

MuJoCo-backed Gymnasium environments must be available for rerunning the full experiments.

## Validate The Checked-In Artifact

Run the research artifact validator:

```bash
python3 scripts/validate_research_artifact.py
```

Expected checks:

- `mcts_200k`: `270` rows
- `telemetry_full_matrix`: `270` rows across the two telemetry summaries
- `iforest_window200_baseline`: `90` rows
- every retained run reaches `global_step=200000`
- the expected environment, poison type, schedule, condition, and seed matrix is complete
- derived output paths exist
- `results/ignored_archive/` exists for noncanonical provenance material

Run code-level validation:

```bash
pytest -q
python3 -m compileall src scripts tests
```

## Reproduce The 200k MCTS Matrix

```bash
python3 scripts/run_dissertation_campaign.py \
  --output-root results/dissertation/mcts_poison_runs_200k \
  --total-steps 200000
```

The runner skips completed cells by inspecting existing run configs and final metrics.

## Future 200k-Capped Early-Stopping Runs

The retained dissertation artifacts are fixed-horizon `200000`-step runs. For a future third run, keep `200000` as the maximum horizon and enable plateau-based early stopping:

```bash
python3 scripts/run_parameterized_campaign.py \
  --backend mcts \
  --output-root results/dissertation/third_run \
  --campaign-name window200_early_stop \
  --total-steps 200000 \
  --window-length 200 \
  --early-stopping \
  --early-stopping-min-steps 100000 \
  --early-stopping-patience-evals 25 \
  --early-stopping-min-delta 0.01 \
  --early-stopping-smoothing-window 5
```

The same flags are available on `scripts/run_dissertation_campaign.py` and `scripts/run_iforest_detector_baseline.py`.
Early-stopped campaign directories and per-run directories include `isolated_earlystop`; the default early-stopped output roots are separate from the retained fixed-horizon result roots.

## Rebuild 200k Derived Data

```bash
python3 scripts/analyze_dissertation_results.py \
  --results-dir results/dissertation/mcts_poison_runs_200k \
  --output results/dissertation/mcts_final_report_200k.md

python3 scripts/export_visual_datasets.py \
  --results-dir results/dissertation/mcts_poison_runs_200k \
  --output-dir results/dissertation/visual_data_200k
```

## Reproduce The Isolation Forest Baseline

```bash
python3 scripts/run_iforest_detector_baseline.py \
  --output-root results/dissertation/iforest_parameterized_runs/window200_iforest_baseline_iforest_steps200000_window200_thresholddefault_20260507T214909 \
  --total-steps 200000 \
  --window-length 200
```

## Rebuild Isolation Forest Comparison Data

```bash
python3 scripts/build_iforest_window200_report.py \
  --iforest-root results/dissertation/iforest_parameterized_runs/window200_iforest_baseline_iforest_steps200000_window200_thresholddefault_20260507T214909 \
  --mcts-root results/dissertation/window200_artifacts/window200_selected_runs \
  --output-root results/dissertation/iforest_window200_artifacts \
  --total-steps 200000
```

## Rebuild Window-200 MCTS Artifacts

The raw `parameterized_runs` source tree is archived locally because the final checked-in artifact only needs the selected derived outputs. To rebuild the window-200 report from raw reruns, restore that archived source tree or rerun:

```bash
python3 scripts/run_parameterized_campaign.py \
  --backend mcts \
  --output-root results/dissertation/parameterized_runs \
  --campaign-name window200_rerun \
  --total-steps 200000 \
  --window-length 200
```

Then build:

```bash
python3 scripts/build_window200_report.py \
  --results-root results/dissertation/parameterized_runs \
  --output-root results/dissertation/window200_artifacts \
  --total-steps 200000
```

## Archival Policy

Do not commit:

- `.venv/`
- `__pycache__/`
- `*.egg-info/`
- `.DS_Store`
- `databricks/`
- `results/ignored_archive/`

Do commit:

- source code required to run or analyze the retained experiments
- tests
- docs that describe the retained experiments
- canonical summary/result/report artifacts listed in `experiments/canonical_experiments.json`
