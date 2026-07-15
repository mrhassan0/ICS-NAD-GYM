"""Strengthens the classifier-as-policy comparison from evaluate_classifiers_in_sim.py:
instead of a hard predict() -> {ALLOW, BLOCK} mapping, use predict_proba() and two
tuned thresholds to get the same 3-way action space the RL policy has:

    p(attack) <  t_low                -> ALLOW
    t_low <= p(attack) < t_high       -> ALERT
    p(attack) >= t_high               -> BLOCK

Thresholds are grid-searched on the VAL split to maximize mean simulator reward
(the same currency the RL arms are judged in, not just F1), then the CHOSEN
thresholds (fixed, not re-tuned) are evaluated fresh on the TEST split with more
episodes and a bootstrap F1 CI - the same val-tune/test-evaluate discipline the
RL arms follow, so no arm gets to tune on its own eval data.
"""

import json
import itertools

import torch
torch.set_num_threads(2)

import numpy as np
import joblib

from icsnad_gym.env import ICSNADGym, PROTOCOL_CATEGORIES, _protocol_onehot
from icsnad_gym.reward import ALLOW, ALERT, BLOCK
from icsnad_gym.scalers import NUMERIC_FEATURES
from rl.stats_utils import bootstrap_f1_ci, f1_from_confusion

MODELS_DIR = "/home/user5/Desktop/MS THESIS/data/baseline_models"
EXPERIMENTS_DIR = "/home/user5/Desktop/MS THESIS/experiments"
CLASSIFIER_NAMES = [
    "GBDT", "RandomForest", "AdaBoost", "LightGBM", "XGBoost",
    "ELM", "ANN", "KNN", "SVC", "GaussianNB",
]

EVAL_SEED = 123
THRESHOLD_GRID = [0.2, 0.35, 0.5, 0.65, 0.8]
N_TUNE_EPISODES = 8
N_EVAL_EPISODES = 40


def _batch_raw_features(df) -> np.ndarray:
    numeric = df.select(NUMERIC_FEATURES).fill_null(0.0).to_numpy().astype(np.float32)
    proto_mat = np.zeros((len(df), len(PROTOCOL_CATEGORIES)), dtype=np.float32)
    for i, p in enumerate(df["protocol"].to_list()):
        proto_mat[i] = _protocol_onehot(p)
    return np.concatenate([numeric, proto_mat], axis=1)


def actions_from_probs(probs: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    actions = np.full(len(probs), ALLOW, dtype=np.int64)
    actions[probs >= t_low] = ALERT
    actions[probs >= t_high] = BLOCK
    return actions


def run_thresholded_policy(env, model, scaler, t_low, t_high, n_episodes, seed=EVAL_SEED):
    """Replays n_episodes deterministically (reset(seed=...) only on the first
    one, exactly like evaluate_multiseed.py) and scores the given threshold pair."""
    total_rewards = []
    episode_confusions = []
    for ep in range(n_episodes):
        if ep == 0:
            obs, info = env.reset(seed=seed)
        else:
            obs, info = env.reset()

        raw = _batch_raw_features(env._episode_df)
        scaled = scaler.transform(raw)
        probs = model.predict_proba(scaled)[:, 1]
        actions = actions_from_probs(probs, t_low, t_high)

        done = False
        total = 0.0
        tp = fp = fn_ = tn = 0
        step_idx = 0
        while not done:
            action = int(actions[step_idx])
            is_attack = info["is_attack"]
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if is_attack and action in (ALERT, BLOCK):
                tp += 1
            elif is_attack and action == ALLOW:
                fn_ += 1
            elif not is_attack and action in (ALERT, BLOCK):
                fp += 1
            else:
                tn += 1
            step_idx += 1
            done = terminated or truncated
        total_rewards.append(total)
        episode_confusions.append((tp, fp, fn_, tn))
    return total_rewards, episode_confusions


def tune_thresholds(name, model, scaler):
    env = ICSNADGym(split="val", episode_length=1024, seed=EVAL_SEED)
    best = None
    for t_low, t_high in itertools.combinations(THRESHOLD_GRID, 2):
        rewards, _ = run_thresholded_policy(env, model, scaler, t_low, t_high, N_TUNE_EPISODES)
        mean_reward = float(np.mean(rewards))
        if best is None or mean_reward > best["mean_reward"]:
            best = {"t_low": t_low, "t_high": t_high, "mean_reward": mean_reward}
    return best


def evaluate_final(name, model, scaler, t_low, t_high, split):
    env = ICSNADGym(split=split, episode_length=1024, seed=EVAL_SEED)
    rewards, confusions = run_thresholded_policy(env, model, scaler, t_low, t_high, N_EVAL_EPISODES)
    agg = np.asarray(confusions, dtype=np.float64).sum(axis=0)
    tp, fp, fn, tn = agg
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = f1_from_confusion(tp, fp, fn)
    ci = bootstrap_f1_ci(confusions, n_boot=2000, alpha=0.05, seed=EVAL_SEED)
    return {
        "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
        "precision": precision, "recall": recall, "f1": f1,
        "f1_ci_lo": ci["lo"], "f1_ci_hi": ci["hi"], "n_episodes": N_EVAL_EPISODES,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    scaler = joblib.load(f"{MODELS_DIR}/scaler.joblib")
    results = {}
    for name in CLASSIFIER_NAMES:
        print(f"\n=== {name} ===", flush=True)
        model = joblib.load(f"{MODELS_DIR}/{name}.joblib")
        if not hasattr(model, "predict_proba"):
            print(f"  skipping {name}: no predict_proba available", flush=True)
            continue

        best = tune_thresholds(name, model, scaler)
        print(f"  tuned on val: t_low={best['t_low']}, t_high={best['t_high']}, "
              f"val_mean_reward={best['mean_reward']:.2f}", flush=True)

        test_metrics = evaluate_final(name, model, scaler, best["t_low"], best["t_high"], "test")
        print(f"  test: {test_metrics}", flush=True)

        # binary-threshold reference (t_low=t_high=0.5, i.e. the original
        # evaluate_classifiers_in_sim.py behavior) for a same-methodology comparison
        binary_metrics = evaluate_final(name, model, scaler, 0.5, 0.5, "test")
        print(f"  test (binary @0.5 reference): {binary_metrics}", flush=True)

        results[name] = {
            "tuned_thresholds": {"t_low": best["t_low"], "t_high": best["t_high"]},
            "val_tuning_mean_reward": best["mean_reward"],
            "test_tuned": test_metrics,
            "test_binary_reference": binary_metrics,
        }

    out_path = f"{EXPERIMENTS_DIR}/classifier_threshold_tuning.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")

    print(f"\n{'model':15s} {'t_low':>6s} {'t_high':>6s} {'reward':>10s} {'f1':>8s} {'f1_lo':>8s} {'f1_hi':>8s} "
          f"{'bin_reward':>11s} {'bin_f1':>8s}")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["test_tuned"]["f1"]):
        t = r["test_tuned"]; b = r["test_binary_reference"]
        print(f"{name:15s} {r['tuned_thresholds']['t_low']:6.2f} {r['tuned_thresholds']['t_high']:6.2f} "
              f"{t['mean_reward']:10.2f} {t['f1']:8.3f} {t['f1_ci_lo']:8.3f} {t['f1_ci_hi']:8.3f} "
              f"{b['mean_reward']:11.2f} {b['f1']:8.3f}")


if __name__ == "__main__":
    main()
