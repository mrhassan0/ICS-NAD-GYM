"""Final multi-seed evaluation: RecurrentPPO (no-shaping) vs RecurrentPPO+LLM-shaping,
across every seed whose pair of final_model.zip files exists. Produces, per run:
  - aggregate precision/recall/F1 (and reward mean) over many held-out episodes
  - an episode-level bootstrap 95% CI on F1
and across seeds:
  - paired t-based CI on the test-F1 difference (shaping - no-shaping)
  - paired Wilcoxon signed-rank test and one-sample paired t-test on the same diffs

Only evaluates seeds where BOTH final_model.zip files already exist, so this can be
re-run as more seeds finish training without editing anything.
"""

import json

import torch
torch.set_num_threads(2)

import numpy as np
from pathlib import Path
from sb3_contrib import RecurrentPPO

from icsnad_gym.env import ICSNADGym
from icsnad_gym.reward import ALLOW, ALERT, BLOCK
from rl.stats_utils import bootstrap_f1_ci, paired_diff_ci_and_test, f1_from_confusion

EXPERIMENTS_DIR = Path("/home/user5/Desktop/MS THESIS/experiments")
N_EPISODES = 40
EVAL_SEED = 123  # fixed across every run so paired comparisons see identical episodes

RUN_NAME_MAP = {
    0: ("recurrentppo_v2_big", "recurrentppo_llm_shaped"),
    1: ("recurrentppo_v2_seed1", "recurrentppo_llm_seed1"),
    2: ("recurrentppo_v2_seed2", "recurrentppo_llm_seed2"),
    3: ("recurrentppo_v2_seed3", "recurrentppo_llm_seed3"),
    4: ("recurrentppo_v2_seed4", "recurrentppo_llm_seed4"),
}


def ready_seeds():
    ready = {}
    for seed, (noshape_name, shape_name) in RUN_NAME_MAP.items():
        noshape_path = EXPERIMENTS_DIR / noshape_name / "final_model.zip"
        shape_path = EXPERIMENTS_DIR / shape_name / "final_model.zip"
        if noshape_path.exists() and shape_path.exists():
            ready[seed] = (noshape_path, shape_path)
    return ready


def run_rl_policy_detailed(env, model, n_episodes: int):
    """Like evaluate_baseline.run_rl_policy but also returns per-episode
    confusion tuples and per-episode reward, needed for bootstrap CIs."""
    episode_rewards = []
    episode_confusions = []  # (tp, fp, fn, tn) per episode
    for _ in range(n_episodes):
        obs, info = env.reset()
        lstm_states = None
        episode_start = np.array([True])
        done = False
        total = 0.0
        tp = fp = fn_ = tn = 0
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            is_attack = info["is_attack"]
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            if is_attack and int(action) in (ALERT, BLOCK):
                tp += 1
            elif is_attack and int(action) == ALLOW:
                fn_ += 1
            elif not is_attack and int(action) in (ALERT, BLOCK):
                fp += 1
            else:
                tn += 1
            episode_start = np.array([False])
            done = terminated or truncated
        episode_rewards.append(total)
        episode_confusions.append((tp, fp, fn_, tn))
    return episode_rewards, episode_confusions


def evaluate_run(model_path, split: str):
    model = RecurrentPPO.load(str(model_path), device="cpu")
    env = ICSNADGym(split=split, episode_length=1024, seed=EVAL_SEED)
    rewards, confusions = run_rl_policy_detailed(env, model, N_EPISODES)
    agg = np.asarray(confusions, dtype=np.float64).sum(axis=0)
    tp, fp, fn, tn = agg
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = f1_from_confusion(tp, fp, fn)
    ci = bootstrap_f1_ci(confusions, n_boot=2000, alpha=0.05, seed=EVAL_SEED)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_ci_lo": ci["lo"],
        "f1_ci_hi": ci["hi"],
        "n_episodes": N_EPISODES,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    ready = ready_seeds()
    print(f"Seeds ready for evaluation (both final_model.zip present): {sorted(ready)}")
    missing = sorted(set(RUN_NAME_MAP) - set(ready))
    if missing:
        print(f"Not yet ready (will be skipped): {missing}")

    results = {}
    for seed, (noshape_path, shape_path) in sorted(ready.items()):
        for split in ["val", "test"]:
            print(f"\n--- seed {seed} no-shaping split={split} ---", flush=True)
            results[f"seed{seed}_noshaping_{split}"] = evaluate_run(noshape_path, split)
            print(results[f"seed{seed}_noshaping_{split}"])

            print(f"--- seed {seed} shaping split={split} ---", flush=True)
            results[f"seed{seed}_shaping_{split}"] = evaluate_run(shape_path, split)
            print(results[f"seed{seed}_shaping_{split}"])

    # Across-seed paired analysis on test F1 (the headline comparison)
    seeds_sorted = sorted(ready)
    noshape_test_f1 = [results[f"seed{s}_noshaping_test"]["f1"] for s in seeds_sorted]
    shape_test_f1 = [results[f"seed{s}_shaping_test"]["f1"] for s in seeds_sorted]
    paired = paired_diff_ci_and_test(noshape_test_f1, shape_test_f1)
    paired["seeds"] = seeds_sorted

    out = {"per_run": results, "paired_test_f1": paired}
    out_path = EXPERIMENTS_DIR / "multiseed_eval_final.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(f"\nPaired test-F1 (shaping - noshaping) across seeds {seeds_sorted}:")
    print(f"  diffs = {paired['diffs']}")
    print(f"  mean diff = {paired['mean_diff']:.4f}  95% CI [{paired['ci_lo']:.4f}, {paired['ci_hi']:.4f}]")
    print(f"  Wilcoxon: stat={paired['wilcoxon_stat']}, p={paired['wilcoxon_p']}")
    print(f"  Paired t-test: stat={paired['paired_ttest_stat']}, p={paired['paired_ttest_p']}")


if __name__ == "__main__":
    main()
