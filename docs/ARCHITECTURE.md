# ADSL Architecture

## Core Pipeline

1. Collect transitions from an RL environment.
2. Apply optional corruption injectors to observations, rewards, or actions.
3. Build sliding windows over transitions.
4. Compute anomaly features and a detector risk score.
5. Trigger MCTS only for flagged windows.
6. Define MCTS state as learner snapshot + suspicious window summary + replay context.
7. Roll forward shadow SAC updates under candidate intervention actions.
8. Score policy deviation relative to the clean-policy reference actor using baseline deviation, predicted return drop, and detector risk.
9. Accept, attenuate, sanitize, or block replay admission and weighting.
10. Record structured outputs for experiments and dashboards, including harmful-update control, final-return outcomes, evaluation-return AUC, and supporting robustness metrics.

## Main Modules

- `adsl.rl`: RL backbones and training loops
- `adsl.corruption`: corruption schedules and injectors
- `adsl.detection`: feature extraction and anomaly scoring
- `adsl.control`: baseline reference, MCTS look-ahead, and intervention logic
- `adsl.data`: replay buffer, clean replay sampling, and sliding windows
- `adsl.pipelines`: experiment orchestration
- `adsl.dashboard`: result aggregation and reporting
