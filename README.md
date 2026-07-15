# ICS-NAD-GYM

ICS-NAD-GYM is a Gymnasium environment and research pipeline for sequential intrusion
response using real Industrial Control System (ICS) network-flow telemetry. It converts
the ICS-NAD dataset from a static classification benchmark into a stateful decision
problem in which an agent must choose whether to `ALLOW`, `ALERT`, or `BLOCK` each flow.

The repository contains:

- A leak-resistant preprocessing and train/validation/test split pipeline.
- A Gymnasium environment with asymmetric security costs, detection latency, and
  stateful source quarantine.
- Ten supervised ML/DL baselines evaluated both as classifiers and simulator policies.
- A recurrent PPO baseline using `sb3-contrib`.
- Offline, label-informed LLM risk scoring of flow archetypes.
- Potential-based reward shaping with no LLM calls in the RL training loop.
- Multi-seed evaluation, bootstrap confidence intervals, and paired statistical tests.

## Research question

Can offline LLM-scored traffic archetypes improve the generalization of a recurrent RL
intrusion-response policy without changing the underlying operational objective?

The project focuses on operational response rather than row-level classification alone.
A model with a strong F1 score can still perform poorly after false-block costs,
detection delay, and quarantine effects are included.

## Environment

Each episode is a contiguous, time-ordered window from one ICS-NAD capture file. Capture
files are never mixed inside an episode. The observation contains 49 normalized numeric
flow features and a six-way protocol one-hot vector. Address identifiers and absolute
timestamps are excluded to reduce memorization shortcuts.

The action space is discrete:

| ID | Action | Meaning |
|---:|---|---|
| 0 | `ALLOW` | Permit the flow without raising an incident |
| 1 | `ALERT` | Flag the flow for investigation |
| 2 | `BLOCK` | Block the flow and start a temporary source quarantine |

The base reward is deliberately asymmetric:

| Traffic | `ALLOW` | `ALERT` | `BLOCK` |
|---|---:|---:|---:|
| Benign | +1 | -5 | -10 |
| Attack | -20 | +10 | +15 |

Correct quarantine receives a per-step bonus of `+2`; false quarantine receives a
per-step penalty of `-3`. Detecting a new attack run also receives a latency bonus of
`5 / (1 + flows_since_onset)`.

## LLM-guided reward shaping

The training split is clustered into 25 flow archetypes using k-means. Each archetype is
summarized using selected traffic statistics and its historical attack rate, then scored
offline by a local 4-bit `Qwen/Qwen3-14B` model. The resulting risk scores define a
potential function over observations.

Shaping uses:

```text
F(s, s') = gamma * Phi(s') - Phi(s),  gamma = 0.99
```

This is potential-based reward shaping, so it preserves the optimal policy of the base
reward formulation when its assumptions hold. LLM inference is cached and is not called
inside the RL training loop.

Because the current LLM prompt includes each archetype's historical attack rate, this
implementation should be described as **label-informed LLM shaping**, not zero-shot LLM
domain knowledge.

## Current results

The latest matched-budget experiment compares five pairs of RecurrentPPO policies trained
for two million steps per run.

| Test result | RL only | RL + LLM shaping |
|---|---:|---:|
| Mean F1 across 5 seeds | 0.849 | 0.872 |
| Mean simulator reward | 2828.26 | 3309.57 |

LLM shaping improved test F1 on four of five seeds. The paired mean F1 difference was
`+0.0232`, with a 95% CI of `[-0.0408, 0.0872]` and Wilcoxon `p = 0.3125`. The observed
effect is therefore promising but **not statistically significant** at the conventional
0.05 threshold.

For validation-tuned supervised policies, AdaBoost produced the highest current mean test
reward (`3279.28`), while ANN produced the highest F1 among those policies (`0.937`). This
is further evidence that classifier F1 and operational reward measure different behavior.

Machine-readable results are available in:

- [`experiments/multiseed_eval_final.json`](experiments/multiseed_eval_final.json)
- [`experiments/classifier_threshold_tuning.json`](experiments/classifier_threshold_tuning.json)

Detailed phase reports are in [`docs/`](docs/).

## Repository layout

```text
src/
  icsnad_gym/   Dataset preparation, splitting, scaling, environment, and rewards
  baselines/    Supervised ML/DL models and training pipeline
  llm/          Flow archetypes, local LLM client, and offline risk scoring
  rl/           RecurrentPPO training, policy evaluation, and statistics
docs/           Methodology, phase reports, results, and research framing
experiments/    Small, machine-readable result summaries
```

Large datasets, model caches, virtual environments, TensorBoard logs, and trained model
checkpoints are intentionally excluded from Git.

## Requirements

The experiments were developed with Python 3.12. Create an isolated environment and
install the research dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  gymnasium numpy polars pyarrow scipy scikit-learn joblib \
  torch stable-baselines3 sb3-contrib tensorboard \
  xgboost lightgbm pydantic
```

The LLM scoring stage additionally requires a CUDA-capable system and:

```bash
python -m pip install transformers accelerate bitsandbytes
```

## Data setup

Download the ICS-NAD dataset separately and arrange the relevant files as follows:

```text
datasets/ICS-NAD/
  Labeled_CSV/
    ABB/
    Schneider/
    Siemens_LONG_0723/
  Time_of_Attack/
```

The dataset is not redistributed by this repository. See the original ICS-NAD
publication under [Citation](#citation).

### Important path configuration

This is currently a research snapshot. Several modules retain absolute paths from the
original experiment machine:

```text
/home/user5/Desktop/MS THESIS
```

Before running the pipeline on another checkout, replace those path constants with the
absolute path to your local repository. Locate every affected source file with:

```bash
rg -l '/home/user5/Desktop/MS THESIS' src
```

Portable project-root configuration is planned but is not implemented in this snapshot.

## Reproducing the pipeline

Run all commands from the repository root:

```bash
export PYTHONPATH="$PWD/src"
```

### 1. Build the harmonized dataset

```bash
python -m icsnad_gym.build_dataset
python -m icsnad_gym.splits
python -m icsnad_gym.scalers
```

The split is file-level whenever multiple captures exist for an attack type. A
time-ordered intra-file split with boundary gaps is used only when an attack type has a
single capture.

### 2. Validate the environment

```bash
python -m icsnad_gym.sanity_check
```

The sanity check compares random and fixed-action policies against an oracle and requires
the oracle to dominate before training results are trusted.

### 3. Train supervised baselines

```bash
python -m baselines.train_supervised_baselines
```

### 4. Build and score LLM archetypes

```bash
python -m llm.archetypes
python -m llm.reward_shaper
```

The second command downloads and runs Qwen3-14B unless it is already present in the local
model cache.

### 5. Train matched RL policies

RL-only example:

```bash
python -m rl.train_recurrent_ppo \
  --run-name recurrentppo_v2_seed1 \
  --total-timesteps 2000000 \
  --seed 1
```

LLM-shaped example:

```bash
python -m rl.train_recurrent_ppo \
  --run-name recurrentppo_llm_seed1 \
  --total-timesteps 2000000 \
  --seed 1 \
  --use-llm-shaping
```

Training is checkpointed and resumes from the newest checkpoint or final model in the
same run directory.

### 6. Evaluate policies

```bash
python -m rl.evaluate_multiseed
python -m rl.tune_classifier_thresholds
```

The multi-seed evaluator uses identical held-out episode sequences for paired policies,
reports bootstrap F1 intervals, and applies paired Wilcoxon and t-tests across seeds.

## Limitations

- The five-seed shaping effect is not yet statistically significant.
- Evaluation currently uses one primary dataset; cross-dataset transfer remains future work.
- The LLM prompt contains label-derived attack rates and needs a no-label ablation before
  making a clean zero-shot domain-knowledge claim.
- Dataset files and trained checkpoints are not included in the repository.
- Source paths are not yet portable across machines.

## Citation

This project is built on the ICS-NAD dataset. Please cite the original dataset papers:

- [A Dataset Collected in Real-World Industrial Control Systems for Network Attack Detection](https://doi.org/10.1038/s41597-026-06738-x), *Scientific Data*, 2026.
- [ICS-NAD: A Dataset Collected in Multiple Real-World Industrial Control Systems for Network Attack Detection](https://doi.org/10.1109/CAC67268.2025.11487720), CAC 2025.

The reward-shaping formulation follows Ng, Harada, and Russell's policy-invariance result:

- [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), ICML 1999.

## Project status

This repository is an active MS thesis research artifact. Results should be interpreted as
experimental evidence rather than a production-ready intrusion prevention system.
