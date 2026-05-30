# Dissertation-Ready Continuous-Control Path

## Scope

This hardened path is intentionally limited to:

- continuous-control SAC training
- `HalfCheetah-v4`, `Walker2d-v4`, and `Hopper-v4`
- `reward_poisoning`, `action_perturbation`, and `observation_corruption`
- `random_sparse` and `bursty` schedules
- `5` seeds per cell
- detector window length `50`

## Novelty-Critical Design

The defended path is built around a detector-triggered MCTS look-ahead validator:

- the detector flags a suspicious transition window
- ADSL launches an MCTS look-ahead over intervention choices
- each MCTS branch performs a shadow SAC update and scores deviation from a clean-policy reference actor
- the selected action controls whether flagged experience is accepted, attenuated, blocked, or sanitized

This is the current dissertation novelty path and supersedes earlier proxy-controller variants.

Additional clarifications:

- attacker capability is modeled as black-box with no parameter access
- poisoning acts on reward, observation-facing, or action-facing experience before replay admission
- the clean baseline actor is a reference for expected clean behavior, not an optimal target policy
- the detector is intentionally permissive and only triggers deeper validation
- MCTS is the decision authority after a detector flag
- the MCTS transition is a short-horizon shadow learning update used to estimate downstream training harm rather than a conventional planning transition
- the final campaign uses the generic SAC actor from `src/adsl/rl.py`
- the optional expert classifier is disabled in the retained final matrix

## Replay Sanitization

Replay sanitization is implemented as replay-admission control and clean-batch substitution:

- blocked or sanitized harmful transitions are withheld from replay insertion
- defended update batches can be replaced with clean-only replay or mixed with clean replay under weighted attenuation
- withheld transitions are counted in `sanitized_transitions`
- `sanitize_clean_replay_uses` and `attenuate_clean_replay_uses` log how often those paths actually used clean-only replay

## Finalized Hypotheses

- `H1`: MCTS-ADSL reduces harmful accepted updates by at least `35%` on average relative to the SAC attack baseline.
- `H2`: MCTS-ADSL improves final attacked-policy evaluation return in at least `20%` of poisoned experimental cells.
- `H3`: MCTS-ADSL improves post-attack evaluation-return AUC in at least `25%` of poisoned experimental cells.

`time-to-threshold` remains a supporting robustness metric rather than the primary `H3` criterion.

## Final Campaign Status

The full `270`-run, `200k`-step MCTS dissertation matrix is complete. The primary report is:

- `results/dissertation/mcts_final_report_200k.md`

Current dissertation-safe interpretation:

- the long-horizon `200k` study supports the revised `H1`, `H2`, and `H3` criteria at the full-matrix level
- contamination control and downstream return must be interpreted separately
- Hopper provides the strongest downstream behavior, with the largest share of positive return and AUC cells
- Walker2d shows the strongest contamination suppression, but often at large return cost
- HalfCheetah remains mixed, with several time-to-threshold gains but weaker final-return and AUC outcomes
- reward poisoning remains the most interpretable poison family for positive downstream effects, but not as a universal win

The visual-ready final `200k` datasets are:

- `results/dissertation/visual_data_200k/final_run_dataset.csv`
- `results/dissertation/visual_data_200k/final_run_dataset.parquet`
- `results/dissertation/visual_data_200k/eval_curve_dataset.csv`
- `results/dissertation/visual_data_200k/eval_curve_dataset.parquet`
