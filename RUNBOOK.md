# Runbook for the Active MCTS ADSL Campaign

## Immediate Goal

Complete the full `270`-run dissertation matrix for the MCTS-based ADSL defense while keeping the run path consistent and resume-safe.

## Active Matrix

- environments: `HalfCheetah-v4`, `Walker2d-v4`, `Hopper-v4`
- schedules: `random_sparse`, `bursty`
- poison types: `reward_poisoning`, `action_perturbation`, `observation_corruption`
- conditions: `clean`, `attack_none`, `attack_defended`
- seeds: `0-4`

## Standard Command

```bash
python3 scripts/run_dissertation_campaign.py --output-root results/dissertation/mcts_poison_runs
```

The runner skips completed cells by matching `config.json` plus final `global_step`, so it is safe to rerun after interruption.

## Pre-Run Validation

Before launching the full matrix, verify:

- the target MuJoCo environment versions are available
- `pytest -q` passes
- `python3 -m compileall src scripts tests` passes
- a short defended smoke run writes `mcts_traces.csv`

## Research-Critical Checks

Before accepting a configuration as dissertation-valid, confirm:

- detector flags occur after attack onset
- MCTS traces are written for flagged defended windows
- root actions include more than trivial `accept` behavior under attack
- sanitized transitions increase in defended attacked conditions
- harmful accepted updates are lower than `attack_none`

## Runtime Guidance

- do not edit training logic while workers are active
- do not change thresholds mid-campaign
- keep the output root fixed to preserve resume behavior
- treat interrupted defended runs as rerunnable, not reusable, unless they reached full `total_steps`

## Reporting Packet

Use the following artifacts during and after the run:

- [docs/METHODOLOGY_DRAFT.md](/Users/jewellwright/Documents/ADSL/docs/METHODOLOGY_DRAFT.md:1)
- [results/dissertation/mcts_interim_results.md](/Users/jewellwright/Documents/ADSL/results/dissertation/mcts_interim_results.md:1)
- `results/dissertation/mcts_poison_runs/summary.csv`

## Common Failure Modes

- detector thresholds too high, causing MCTS never to trigger after attack onset
- detector thresholds too low, causing excessive sanitization in benign windows
- defended return drop despite strong harmful-update reduction
- environment-dependent divergence, especially under observation corruption
- partial result directories left after interrupted runs
