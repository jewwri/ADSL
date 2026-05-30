# Experimental Design for MCTS-ADSL

## 1. Experimental Objective

The purpose of the current campaign is to evaluate whether `ADSL` can:

1. establish a clean baseline policy in continuous-control SAC training,
2. detect suspicious transition windows under poisoned online experience,
3. trigger an MCTS look-ahead validator after detection,
4. use that look-ahead to gate replay admission and update weighting,
5. preserve learning robustness relative to undefended SAC.

The key thesis is now explicitly systems-oriented: poisoning defense should be judged by whether it prevents harmful replay reuse and policy deviation, not by detector quality alone.

## 2. Current Hypothesis Frame

The live dissertation campaign is organized around three system-facing claims:

- `H1`: ADSL will reduce harmful accepted updates by at least `80%` relative to standard SAC.
- `H2`: ADSL will improve attacked-policy evaluation return by at least `10%` on average relative to standard SAC.
- `H3`: ADSL will increase time-to-threshold by at least `25%` relative to standard SAC under confirmed poisoning events.

## 3. Research Questions

1. How accurately can the detector identify anomalous windows before updates are applied?
2. How accurately can the expert layer distinguish among reward poisoning, observation corruption, and action perturbation?
3. Does intervention at update time prevent degradation more effectively than passive anomaly flagging?
4. What is the tradeoff between protection and over-intervention?
5. How sensitive is the system to window length, detection threshold, harm threshold, and attack schedule?
6. How does the method behave when corruption is rare, bursty, or adaptive?

## 3. Experimental Factors

### 3.1 RL Backbone

The current campaign uses continuous-control `SAC` only.

### 3.2 Environments

- `HalfCheetah-v4`
- `Walker2d-v4`
- `Hopper-v4`

### 3.3 Corruption Families

Include three poisoning families:

#### Reward Poisoning

Examples:

- reward sign flipping
- additive adversarial bias
- reward clipping or inflation
- delayed reward distortion

Operational injection:

\[
r_t^{\text{corr}} = r_t + \delta_t
\]
or
\[
r_t^{\text{corr}} = -r_t
\]
for corrupted intervals.

#### Action Perturbation

Examples:

- random replacement
- targeted replacement with worst-case action
- Gaussian perturbation in continuous control

Operational injection:

\[
a_t^{\text{exec}} = a_t + \epsilon_t
\]
or action substitution under a corruption mask.

#### Observation Corruption

Examples:

- Gaussian noise
- feature masking
- adversarial feature shift
- partial permutation of observation dimensions

Operational injection:

\[
s_t^{\text{corr}} = s_t + \nu_t
\]

### 3.4 Attack Schedules

Evaluate each poisoning family under:

#### Random Sparse

- timestep-level low-probability corruption after warmup

#### Bursty

- contiguous attack windows after warmup

### 3.5 Conditions

- `clean`
- `attack_none`
- `attack_defended`

### 3.6 Seeds And Windowing

- seeds: `0-4`
- detector window length: `50`

## 4. Method Blocks

The methodology is implemented in six blocks:

1. establish a clean baseline and simulate poisoning
2. construct the ADSL ensemble
3. integrate ADSL with replay
4. implement gating and weighting
5. evaluate detection and learning robustness
6. analyze and refine behavior

These blocks are expanded in [docs/METHODOLOGY_DRAFT.md](/Users/jewellwright/Documents/ADSL/docs/METHODOLOGY_DRAFT.md:1).
- Detector-only pipeline with no update intervention
- Monolithic corruption classifier with no experts
- Heuristic filter baseline such as TD-error thresholding

### Useful Stronger Baselines

- Replay filtering by anomaly score
- Reward clipping defense
- Observation denoising preprocessor
- Robust loss or adversarial training baseline

The point of the baseline set is to isolate whether the gains come from detection, diagnosis, or the control loop itself.

## 7. Evaluation Metrics

### 7.1 Detection Metrics

Measure on window labels:

- true positive rate
- false positive rate
- precision
- recall
- F1
- AUROC
- AUPRC
- expected calibration error if risk is probabilistic

### 7.2 Classification Metrics

Measure on flagged anomalous windows:

- overall accuracy
- macro-F1
- weighted F1
- per-class precision and recall
- confusion matrix
- confidence calibration

### 7.3 RL Performance Metrics

Measure at training and evaluation time:

- average episodic return
- rolling return variance
- worst-case return during attack windows
- area under learning curve
- recovery time after attack burst
- final return
- degradation relative to benign training

### 7.4 Control Metrics

Measure controller behavior directly:

- harmful update acceptance rate
- benign update blocking rate
- overall intervention rate
- modification rate
- average trust coefficient if using soft control
- compute overhead per environment step or update

## 8. Experimental Protocol

### 8.1 Training Horizon

Choose a fixed training budget per environment.

Suggested starting values:

- CartPole: 100k to 300k environment steps
- LunarLander: 300k to 1M steps
- MuJoCo tasks: 500k to 2M steps

### 8.2 Random Seeds

Each configuration should be run with multiple seeds.

Minimum:

- 5 seeds for pilot experiments

Preferred:

- 10 seeds for final reporting

### 8.3 Evaluation Frequency

At fixed intervals:

- run evaluation episodes without exploration noise if appropriate
- log detector and controller statistics
- record corruption-specific outcomes during the most recent interval

Suggested interval:

- every 5k or 10k environment steps for small environments
- every 25k or 50k steps for larger environments

### 8.4 Train / Validation / Test Splits for Corruption Models

If detector and classifier are trained using stored windows:

- Train split: 70 percent
- Validation split: 15 percent
- Test split: 15 percent

Split by trajectory or episode, not by individual window, to avoid leakage.

### 8.5 Threshold Selection

Thresholds should not be tuned on the test split.

Use validation to set:

- anomaly threshold \(\eta\)
- harm threshold \(\kappa\)
- trust attenuation schedule if soft control is used

Recommended strategy:

- choose \(\eta\) by maximizing validation F1 or by constraining FPR
- choose \(\kappa\) by minimizing downstream validation degradation

## 9. Suggested Experiment Sequence

Run experiments in the following order.

### Phase 1: Pipeline Sanity Checks

- Single environment: `CartPole-v1`
- Single corruption type at a time
- Random sparse schedule
- Balanced labels

Goal:

- verify corruption injection
- verify detector learns usable signal
- verify shadow controller logic executes correctly

### Phase 2: Detector and Expert Validation

- Add bursty schedule
- Add all three corruption types
- Compare monolithic vs expert classifier
- Sweep window length and detector threshold

Goal:

- establish the detection and diagnosis story before full RL claims

### Phase 3: Full-System Comparison

- Compare undefended RL, detector-only, expert-only, controller-only, and full ANS-RL
- Use balanced and realistic corruption regimes
- Measure RL robustness and control tradeoffs

Goal:

- demonstrate the value of the integrated closed-loop system

### Phase 4: Generalization and Stress Testing

- Move to continuous control
- Add persistent and escalating attacks
- Stress-test under low corruption prevalence

Goal:

- assess transferability and operational realism

## 10. Statistical Analysis

Use confidence intervals across seeds for all primary metrics.

Recommended tests:

- mean and standard deviation across seeds
- 95 percent confidence intervals by bootstrap or t-interval
- paired comparisons across matched seeds where possible

Primary comparisons should be reported against:

- no defense baseline
- detector-only baseline
- monolithic classifier baseline

## 11. Primary Tables and Figures

Prepare the following outputs.

### Tables

- detection performance by corruption type and schedule
- classification performance by model type
- RL robustness across baselines
- intervention tradeoff table with harmful accept and benign block rates

### Figures

- training curves under clean vs corrupted conditions
- rolling return under bursty attacks
- ROC and PR curves for detector
- confusion matrices for corruption experts
- sensitivity plots over \(\eta\), \(\kappa\), and window length

## 12. Threats to Validity

The experimental design should acknowledge:

- synthetic corruption may not capture all real-world failure modes
- detector performance may depend on environment-specific dynamics
- shadow-update cost may be prohibitive in larger settings
- thresholds may require retuning across tasks
- rare corruption settings may produce unstable estimates for minority classes

## 13. Minimum Viable Experimental Package

If you need to start with a lean setup, use this first:

- Environment: `CartPole-v1`
- RL algorithm: DQN
- Window lengths: 8, 16, 32
- Corruption types: reward poisoning, observation corruption, action perturbation
- Schedules: random sparse and bursty
- Baselines: no defense, detector-only, monolithic classifier, full ANS-RL
- Seeds: 5
- Metrics: F1, AUROC, episodic return, return variance, harmful accept rate, benign block rate

This setup is enough to generate the first meaningful ablation results.

## 14. Recommended Logging Schema

For every run, log at minimum:

- run_id
- seed
- environment
- RL algorithm
- corruption type
- corruption schedule
- corruption severity
- detector features
- window length
- detector threshold
- expert architecture
- controller mode
- harm threshold
- train steps
- eval return mean
- eval return std
- detector precision
- detector recall
- detector F1
- classifier macro-F1
- harmful accept rate
- benign block rate
- runtime

## 15. Decision Rule for Success

The method should be considered successful if the full system:

1. Improves RL robustness relative to no-defense and detector-only baselines.
2. Maintains acceptable false positive and benign-block rates.
3. Preserves performance close to the undefended learner in clean settings.
4. Continues to show benefit under realistic low corruption prevalence.
