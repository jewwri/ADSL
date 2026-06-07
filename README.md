# ADSL Dissertation Experiments

## Experiment Scope

All retained experiments use:

- environments: `HalfCheetah-v4`, `Walker2d-v4`, `Hopper-v4`
- poison types: `reward_poisoning`, `action_perturbation`, `observation_corruption`
- schedules: `random_sparse`, `bursty`
- seeds: `0-4`
- horizon: `200000` environment steps

The MCTS matrices compare `clean`, `attack_none`, and `attack_defended`. The Isolation Forest baseline is detector-only.

## Active Results

The machine-readable source of truth is `experiments/canonical_experiments.json`.

- `results/dissertation/mcts_poison_runs_200k/`
- `results/dissertation/telemetry_runs/`
- `results/dissertation/iforest_parameterized_runs/window200_iforest_baseline_iforest_steps200000_window200_thresholddefault_20260507T214909/`
- `results/dissertation/visual_data_200k/`
- `results/dissertation/window200_artifacts/`
- `results/dissertation/iforest_window200_artifacts/`

## Active Code

- `src/adsl/`: SAC training, corruption, detection, MCTS control, telemetry, and Isolation Forest pipelines
- `scripts/run_dissertation_campaign.py`: `200k` MCTS matrix runner
- `scripts/run_parameterized_campaign.py`: parameterized MCTS or Isolation Forest campaign runner
- `scripts/run_window200_worker_pool.py`: coordinated window-200 MCTS worker pool
- `scripts/run_iforest_detector_baseline.py`: detector-only Isolation Forest baseline runner
- `scripts/analyze_dissertation_results.py`: MCTS report builder
- `scripts/export_visual_datasets.py`: visual dataset exporter
- `scripts/build_window200_report.py`: window-200 MCTS report builder
- `scripts/build_iforest_window200_report.py`: Isolation Forest comparison report builder

## Documentation

- `RUNBOOK.md`
- `docs/REPRODUCIBILITY.md`
- `docs/DATA_DICTIONARY.md`
- `docs/METHODOLOGY_DRAFT.md`
- `docs/README_WINDOW200.md`
- `docs/ARCHITECTURE.md`
- `docs/DISSERTATION_READY.md`
