# Window-200 Rerun Notes

This document tracks the long-horizon rerun that preserved the `200,000`-step dissertation matrix while changing the detector window length from `50` to `200`.

## Purpose

- evaluate whether the shorter `window=50` detector was over-granular for long-horizon training
- preserve the original attack schedules, environments, poison types, seeds, and total training horizon
- compare the resulting `window=200` matrix against the original `window=50` MCTS dissertation run

## Shared Experimental Constants

- environments: `HalfCheetah-v4`, `Walker2d-v4`, `Hopper-v4`
- schedules: `random_sparse`, `bursty`
- poison types: `reward_poisoning`, `action_perturbation`, `observation_corruption`
- conditions: `clean`, `attack_none`, `attack_defended`
- seeds: `0-4`
- total steps: `200000`

## Detector Configuration Change

- changed detector `window_length` from `50` to `200`
- preserved the original schedule-specific thresholds in the run configs
- `random_sparse`: `0.12`
- `bursty`: `0.14`

## Reporting Location

- artifact README: `results/dissertation/window200_artifacts/README_window200.md`
- selected runs: `results/dissertation/window200_artifacts/window200_selected_runs`
- final report: `results/dissertation/window200_artifacts/window200_final_report.md`

## Current Interpretation

- `window=200` supports the contamination-control claims: mean harmful accepted update reduction is `64.2%`, and `12 / 18` poisoned cells reduce harmful accepted updates by at least `50%`.
- `window=200` also supports the degradation-delay claim: `12 / 18` poisoned cells show positive time-to-threshold lift.
- Final return and evaluation-AUC should be discussed as diagnostics, not primary supported claims. Positive final-return lift occurs in `1 / 18` poisoned cells, and positive evaluation-AUC lift occurs in `3 / 18` poisoned cells.
