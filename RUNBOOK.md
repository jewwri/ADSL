# Runbook

## Retained Result Families

The checked-in result tree is intentionally limited to:

- `results/dissertation/mcts_poison_runs_200k/`
- `results/dissertation/telemetry_runs/`
- `results/dissertation/iforest_parameterized_runs/window200_iforest_baseline_iforest_steps200000_window200_thresholddefault_20260507T214909/`
- derived reports and visual datasets for those runs

The canonical manifest is `experiments/canonical_experiments.json`.

## Validate The Artifact

```bash
python3 scripts/validate_research_artifact.py
pytest -q
python3 -m compileall src scripts tests
```

## Reproduce Or Resume The 200k MCTS Matrix

```bash
python3 scripts/run_dissertation_campaign.py \
  --output-root results/dissertation/mcts_poison_runs_200k \
  --total-steps 200000
```

The runner is resume-safe and skips completed cells by matching `config.json` plus final `global_step`.

## Rebuild The 200k MCTS Report Data

```bash
python3 scripts/analyze_dissertation_results.py \
  --results-dir results/dissertation/mcts_poison_runs_200k \
  --output results/dissertation/mcts_final_report_200k.md

python3 scripts/export_visual_datasets.py \
  --results-dir results/dissertation/mcts_poison_runs_200k \
  --output-dir results/dissertation/visual_data_200k
```

## Rebuild Window-200 MCTS Artifacts

```bash
python3 scripts/build_window200_report.py \
  --results-root results/dissertation/parameterized_runs \
  --output-root results/dissertation/window200_artifacts \
  --total-steps 200000
```

The source `parameterized_runs` tree is archived because only the derived `window200_artifacts` are part of the active checked-in result set.

## Rebuild Isolation Forest Comparison Artifacts

```bash
python3 scripts/build_iforest_window200_report.py \
  --iforest-root results/dissertation/iforest_parameterized_runs/window200_iforest_baseline_iforest_steps200000_window200_thresholddefault_20260507T214909 \
  --mcts-root results/dissertation/window200_artifacts/window200_selected_runs \
  --output-root results/dissertation/iforest_window200_artifacts \
  --total-steps 200000
```

## Validation

```bash
pytest -q
python3 -m compileall src scripts tests
```
