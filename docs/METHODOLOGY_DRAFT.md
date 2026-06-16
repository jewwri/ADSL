# Methodology Draft for the MCTS-ADSL Dissertation Study

## 1. Establish Baseline And Simulate Poisoning

The study begins by training a standard SAC agent in continuous-control MuJoCo environments under a clean warmup period. The current environment set is `HalfCheetah-v4`, `Walker2d-v4`, and `Hopper-v4`. The policy backbone is a tanh-squashed Gaussian SAC actor with a two-layer `256 x 256` MLP trunk, paired with the corresponding double-Q SAC critic with two-layer `256 x 256` MLPs.

The clean warmup serves two purposes: it provides nominal policy behavior against which later defended behavior can be compared, and it gives the detector access to non-poisoned transition windows for baseline feature estimation. Importantly, the captured clean actor is treated as a reference for expected clean-policy behavior, not as a fixed optimal target.

After warmup, poisoning is injected online into the training stream using one of three poisoning families:

- `reward_poisoning`
- `action_perturbation`
- `observation_corruption`

Each poisoning family is evaluated under two schedules:

- `random_sparse`
- `bursty`

The threat model is explicitly black-box. The attacker has no access to model parameters, gradients, or replay internals. Instead, the attacker can perturb reward, observation-facing, or action-facing experience after an environment step has been generated and before the resulting transition is admitted into replay. This preserves a realistic training-time threat model in which the learner observes corrupted experience rather than a corrupted environment implementation.

Poisoning budget is explicit in configuration and logs. In the current dissertation matrix, `random_sparse` uses a budget of `0.08` corrupted events per step in expectation, while `bursty` uses a burst budget of `40 / 200 = 0.20` of steps within the attack window. Those schedule parameters are written to the run config and logged into control traces and final metrics.

## 2. Construct The ADSL Ensemble

ADSL is constructed as a multi-stage defense ensemble rather than a single detector. Its current defended path contains four coordinated components:

- a window-based anomaly detector
- a baseline reference module
- an MCTS look-ahead controller
- a replay-buffer intervention layer

An optional fifth component exists in the codebase: an expert classifier over detector-window features. This expert path is advisory rather than authoritative, and it is disabled in the final dissertation MCTS matrix to isolate detector + reference actor + MCTS behavior.

The detector operates on windows of `50` transitions. It extracts reward statistics, action statistics, state-shift features, and temporal-delta features and produces a scalar anomaly risk. The detector is intentionally permissive: it is used as a trigger for deeper validation rather than the final decision maker.

The baseline reference module stores observations from the clean training prefix and captures a clean-policy actor snapshot after baseline warmup. This reference actor provides the expected clean-behavior comparison used later by the MCTS controller; it is not assumed to be optimal, frozen globally, or privileged beyond representing the learner before poisoning begins.

When expert mode is enabled, the expert classifier is a supervised two-layer `64 x 64` MLP over detector-window features. It is trained on labeled detector windows collected from the same experimental cell, where the target label is the dominant corruption label within the flagged window. Its output is a probability distribution over the configured classes, such as `clean` and the current poison type. In the final MCTS campaign, this expert prediction is logged only when enabled and is not used to override MCTS.

## 3. Integrate ADSL With The Replay Buffer

ADSL is integrated directly into the replay-admission path. Every transition is appended to a sliding window buffer. Once the window is full, the detector evaluates the entire transition sequence. If the window is not flagged, the transition proceeds normally. If the window is flagged, the controller is invoked before poisoned experience is allowed to propagate through replay reuse.

The replay buffer supports two sampling modes:

- standard sampling from all admitted transitions
- clean-only sampling from transitions marked as uncorrupted

This clean-only replay path is necessary for two reasons. First, it allows replay sanitization to substitute or blend clean transitions when flagged experience is judged harmful. Second, it gives the MCTS look-ahead a stable comparison batch when evaluating intervention choices.

Replay intervention semantics are explicit:

- `sanitize` uses `clean_only_replacement`: flagged replay influence is captured for expert evidence, withheld from learner replay, and replaced with clean-only replay samples when clean replay is available
- `accept` keeps the transition in learner replay when detector/controller confidence is insufficient for sanitation
- `accept` leaves replay admission and sampling unchanged

The implementation logs whether clean replay was actually available, how often `sanitize` used clean-only replay, and how many suspicious windows were captured for expert training.

## 4. Implement Gating And Weighting

The defended gating mechanism is a post-detection MCTS look-ahead. After the detector flags a suspicious window, ADSL constructs a look-ahead problem with:

- state = current learner state + suspicious window summary + replay context
- actions = `accept`, `sanitize`
- transition = one short-horizon shadow learning update under the selected intervention
- reward/score = weighted combination of deviation from clean baseline behavior, predicted return drop, and detector risk

For each candidate intervention, MCTS simulates a short-horizon shadow SAC update using a copy of the current learner. This transition should be read as a training-dynamics transition rather than a conventional environment-planning transition. The resulting candidate actor is compared against the clean reference actor on stored clean-prefix observations. The scoring function uses the controller weights in config, with the current default weighting:

- baseline-behavior deviation: `0.55`
- predicted return drop: `0.25`
- detector risk: `0.20`

The chosen action governs how the training pipeline handles the suspicious experience:

- `accept`: admit the transition and train normally
- `accept`: allow low-confidence suspicious windows to continue training
- `sanitize`: capture suspicious evidence, withhold suspicious replay influence, and replace the learner update with clean-only replay when available

This design allows the detector threshold to remain lower than a one-stage detector-controller system could tolerate, because the final intervention is validated by look-ahead rather than by anomaly score alone.

## 5. Evaluate Detection And Learning Robustness

The experimental matrix compares three conditions:

- `clean`
- `attack_none`
- `attack_defended`

Evaluation is organized around both detection and downstream robustness. Detection reporting includes detector F1 over poisoned windows. Robustness reporting includes:

- final attacked-policy evaluation return
- post-attack area under the evaluation return curve
- harmful accepted update rate
- sanitized transition count
- time-to-threshold relative to the clean baseline
- detector flag count and flag rate
- intervention frequencies for `accept` and `sanitize`
- how often `sanitize` actually used clean-only replay
- captured suspicious-window counts for expert training

The primary hypothesis tests are:

- `H1`: mean harmful accepted updates decrease by at least `35%` relative to the standard SAC baseline
- `H2`: final attacked-policy evaluation return improves in at least `20%` of poisoned experimental cells
- `H3`: post-attack evaluation-return AUC improves in at least `25%` of poisoned experimental cells

Each experimental cell uses `5` seeds and trains for `200,000` environment steps in the finalized dissertation matrix.

## 6. Analyze And Refine ADSL Behavior

Analysis is not limited to whether the defense helps. It also asks how it helps and where it fails. For each environment, schedule, and poison type, the study compares:

- detector quality in attacked conditions
- how often MCTS chooses `accept` or `sanitize`
- how often flagged windows lead to clean-only replay usage
- whether strong contamination reduction is accompanied by return gains
- whether evaluation-return AUC improves even when final return does not
- whether intervention is overly conservative in environments where SAC remains stable under attack

This final analysis step is essential because a defense can appear successful by reducing harmful replay contamination yet still underperform on final return if the intervention policy is too conservative. The dissertation interpretation must therefore distinguish between contamination control, final-return outcomes, learning-curve robustness, and environment-dependent tradeoffs. To support that distinction, the analysis emits per-environment and per-poison-type detector-to-learning linkage summaries that connect detector flags and intervention choices to harmful accepted updates, final return, evaluation-return AUC, and time-to-threshold.
