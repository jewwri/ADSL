# Data Dictionary

This dictionary covers the canonical dissertation artifacts declared in `experiments/canonical_experiments.json`.

## Experimental Factors

- `env_id`: Gymnasium MuJoCo task. Retained values are `HalfCheetah-v4`, `Hopper-v4`, and `Walker2d-v4`.
- `poison_type`: corruption family, one of `reward_poisoning`, `action_perturbation`, or `observation_corruption`.
- `schedule`: attack schedule, either `random_sparse` or `bursty`.
- `condition`: MCTS matrix condition, one of `clean`, `attack_none`, or `attack_defended`.
- `seed`: integer random seed, retained values `0` through `4`.
- `global_step`: final training step. Canonical retained summaries must reach `200000`.
- `target_steps`: configured maximum training horizon. Future early-stopped runs can have `global_step < target_steps`.

## Common Run Columns

- `run_name`: encoded run identifier containing environment, poison type, schedule, condition, and seed.
- `run_dir`: path to the run directory that produced the row.
- `eval_return_mean`: mean evaluation return at the logged step.
- `accepted_updates`: number of training updates accepted by the pipeline.
- `blocked_updates`: legacy blocked-update count from old artifacts; corrected ADSL accept/sanitize runs emit `0`.
- `sanitized_transitions`: number of harmful transitions withheld or sanitized before replay influence.
- `flagged_windows`: number of detector-flagged transition windows.
- `flagged_harmful_windows`: number of detector-flagged windows containing harmful corruption.
- `captured_suspicious_windows`: number of detector-flagged windows captured for expert-training evidence.
- `captured_harmful_windows`: number of captured windows that contain injected corruption according to experiment labels.
- `attack_steps`: number of poisoned attack steps observed by the run.
- `harmful_accept_rate`: fraction of harmful attack updates accepted.
- `benign_block_rate`: legacy benign-block rate from old artifacts; corrected ADSL accept/sanitize runs emit `0`.
- `detector_precision`, `detector_recall`, `detector_f1`: detector quality against poisoned-window labels.
- `early_stopping_enabled`: whether plateau-based stopping was enabled for the run.
- `early_stopping_min_steps`, `early_stopping_patience_evals`, `early_stopping_min_delta`, `early_stopping_smoothing_window`: stopping policy configuration emitted with telemetry-backed metric rows.
- `early_stopped`: whether the run stopped before `target_steps`.
- `stop_reason`: reason for early stopping, currently `plateau` when triggered.
- `early_stopping_best_smoothed_return`: best smoothed evaluation return seen by the stopping monitor.
- `early_stopping_stale_evals`: number of consecutive evaluation points without meaningful smoothed-return improvement.

## MCTS Intervention Columns

- `interventions_accept`: detector-flagged windows where ADSL selected `accept`.
- `interventions_attenuate`: legacy column retained for old artifacts; corrected ADSL runs emit `0`.
- `interventions_block`: legacy column retained for old artifacts; corrected ADSL runs emit `0`.
- `interventions_sanitize`: detector-flagged windows where ADSL selected `sanitize`.
- `sanitize_clean_replay_uses`: count of sanitize actions that used clean-only replay.
- `attenuate_clean_replay_uses`: legacy column retained for old artifacts; corrected ADSL runs emit `0`.
- `sanitize_replay_mode`: sanitize implementation mode, retained as configuration evidence.
- `policy_backbone`: actor architecture identifier.
- `reference_actor_role`: role of the clean actor snapshot used for MCTS deviation scoring.
- `experts_enabled`, `experts_mode`: retained to document that the final matrix disables expert intervention.

## Attack Model Columns

- `attack_budget`: configured poisoning budget.
- `attack_budget_unit`: unit for the budget, typically `event_fraction`.
- `attacker_capability`: attacker knowledge/access assumption.
- `attack_surface`: point where corruption enters the training stream.

## Telemetry Columns

Telemetry summaries include all common run columns plus:

- `timestamp_utc`: timestamp of the final telemetry row.
- `run_started_utc`: run start timestamp.
- `wall_time_elapsed_s`: wall-clock runtime.
- `process_cpu_user_s`, `process_cpu_system_s`, `process_cpu_time_s`: process CPU accounting.
- `process_cpu_util_percent`, `process_cpu_util_normalized_percent`: process CPU utilization.
- `process_rss_mb`, `process_vms_mb`: process memory usage.
- `process_thread_count`: process thread count.
- `system_cpu_percent`, `system_memory_percent`: host utilization snapshot.
- `gpu_memory_allocated_mb`, `gpu_memory_reserved_mb`: GPU memory counters when available.
- `detector_runtime_ms_total`, `detector_runtime_ms_mean`, `detector_runtime_calls`: detector runtime telemetry.
- `mcts_runtime_ms_total`, `mcts_runtime_ms_mean`, `mcts_runtime_calls`: MCTS runtime telemetry.

## Isolation Forest Columns

Isolation Forest summaries include telemetry-style runtime columns plus:

- `detector_backend`: detector implementation name.
- `detector_gate_mode`: gate action used by the Isolation Forest sanitize-gated ablation.
- `detector_threshold`: risk threshold for Isolation Forest gating.
- `detector_fit_runtime_ms_total`: cumulative Isolation Forest fitting runtime.

## Derived Visual Dataset Columns

`results/dissertation/visual_data_200k/final_run_dataset.csv` joins final metrics with configuration metadata:

- `experiment_name`: configured experiment name.
- `attack_start_step`: configured attack onset.
- `window_length`: detector window length.
- `detector_threshold`: detector trigger threshold.
- `controller_mode`: controller mode, usually `mcts` for defended rows and `none` otherwise.
- `mcts_simulations`: MCTS simulation count.
- `mcts_horizon`: MCTS rollout horizon.
- `target_steps`: configured training horizon.
- `batch_size`: SAC update batch size.
- `replay_size`: replay buffer capacity.
- `completed`: whether the run reached its target horizon.
- `final_return`: final evaluation return, copied from the final metrics row.
- `evaluation_auc`: post-attack area under the evaluation-return curve.
