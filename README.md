# ADSL Research Repository

This repository contains the active `ADSL` dissertation experiment stack for poisoned continuous-control reinforcement learning. The current research path centers on:

- baseline SAC training in MuJoCo continuous-control environments
- simulated `reward_poisoning`, `action_perturbation`, and `observation_corruption`
- window-based anomaly detection with `window_length=50`
- post-detection Monte Carlo Tree Search (MCTS) look-ahead validation
- replay-buffer gating, sanitization, attenuation, and blocking
- dissertation reporting around detection quality and learning robustness

## Current Experiment Scope

The primary dissertation campaign is:

- environments: `HalfCheetah-v4`, `Walker2d-v4`, `Hopper-v4`
- schedules: `random_sparse`, `bursty`
- poison types: `reward_poisoning`, `action_perturbation`, `observation_corruption`
- conditions: `clean`, `attack_none`, `attack_defended`
- seeds: `0-4`

The defended path uses a detector-triggered MCTS validator that estimates policy deviation from a clean baseline actor before allowing poisoned experience to influence replay or updates.
That baseline actor is a clean-policy reference snapshot captured after warmup, not a fixed optimal policy target.

The actor used in the dissertation campaign is the generic SAC actor from [src/adsl/rl.py](/Users/jewellwright/Documents/ADSL/src/adsl/rl.py:18): a tanh-squashed Gaussian policy with a `256 x 256` MLP backbone.

The optional expert path uses the classifier in [src/adsl/experts.py](/Users/jewellwright/Documents/ADSL/src/adsl/experts.py:18), trained on detector-window features and dominant window corruption labels. In the final MCTS matrix, `experts.enabled=false` so the main results isolate the detector + clean-policy reference + MCTS intervention stack.

## Repository Layout

- `src/adsl/`: core training, detection, corruption, control, and logging code
- `scripts/`: experiment runners, analysis entrypoints, and campaign orchestration
- `docs/`: architecture, dissertation-path notes, and methodology drafts
- `results/`: generated reports and experiment outputs
- `databricks/`: optional remote execution scaffolding

## Recommended First Steps

1. Create a Python environment and install dependencies from `pyproject.toml`.
2. Read [docs/DISSERTATION_READY.md](/Users/jewellwright/Documents/ADSL/docs/DISSERTATION_READY.md:1).
3. Read [docs/METHODOLOGY_DRAFT.md](/Users/jewellwright/Documents/ADSL/docs/METHODOLOGY_DRAFT.md:1).
4. Launch or resume the dissertation campaign:

```bash
python3 scripts/run_dissertation_campaign.py \
  --output-root results/dissertation/mcts_poison_runs_200k \
  --total-steps 200000
```

The runner is resume-safe and skips completed cells automatically.

For parameterized reruns without touching the dissertation script itself, use:

```bash
python3 scripts/run_parameterized_campaign.py \
  --backend mcts \
  --output-root results/dissertation/mcts_poison_runs_window200 \
  --total-steps 200000 \
  --window-length 200 \
  --detector-threshold 0.2
```

Or for the isolated detector-only baseline:

```bash
python3 scripts/run_parameterized_campaign.py \
  --backend iforest \
  --output-root results/dissertation/iforest_detector_runs_window200 \
  --total-steps 200000 \
  --window-length 200 \
  --detector-threshold 0.5 \
  --gate-mode sanitize
```

## Reporting

Current primary reporting artifacts:

- final `200k` results: [mcts_final_report.md](/Users/jewellwright/Documents/ADSL/results/dissertation/mcts_final_report.md:1)
- dissertation-path notes: [DISSERTATION_READY.md](/Users/jewellwright/Documents/ADSL/docs/DISSERTATION_READY.md:1)
- methodology draft: [METHODOLOGY_DRAFT.md](/Users/jewellwright/Documents/ADSL/docs/METHODOLOGY_DRAFT.md:1)
- architecture summary: [ARCHITECTURE.md](/Users/jewellwright/Documents/ADSL/docs/ARCHITECTURE.md:1)
