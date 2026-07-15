"""Evaluate the trained RecurrentPPO baseline against val/test episodes, alongside the
Phase 2 sanity-floor policies (random / always-ALLOW / always-ALERT / always-BLOCK /
oracle) for direct comparison on the same splits.
"""

import random

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from icsnad_gym.env import ICSNADGym
from icsnad_gym.reward import ALLOW, ALERT, BLOCK


def run_baseline_policy(env, policy_fn, n_episodes: int):
    totals, tp, fp, fn_, tn = [], 0, 0, 0, 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        total = 0.0
        is_attack = info["is_attack"]
        while not done:
            action = policy_fn(is_attack)
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
            is_attack = info["is_attack"]
            done = terminated or truncated
        totals.append(total)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn_) if (tp + fn_) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"mean_reward": float(np.mean(totals)), "std_reward": float(np.std(totals)),
            "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn_, "tn": tn}


def run_rl_policy(env, model, n_episodes: int):
    totals, tp, fp, fn_, tn = [], 0, 0, 0, 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        lstm_states = None
        episode_start = np.array([True])
        done = False
        total = 0.0
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
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
        totals.append(total)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn_) if (tp + fn_) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"mean_reward": float(np.mean(totals)), "std_reward": float(np.std(totals)),
            "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn_, "tn": tn}


def main():
    model = RecurrentPPO.load("/home/user5/Desktop/MS THESIS/experiments/recurrentppo_baseline_v1/final_model.zip", device="cpu")

    for split in ["val", "test"]:
        print(f"\n{'='*20} split={split} {'='*20}")
        env = ICSNADGym(split=split, episode_length=1024, seed=123)

        baselines = {
            "always_ALLOW": lambda is_attack: ALLOW,
            "always_ALERT": lambda is_attack: ALERT,
            "oracle": lambda is_attack: BLOCK if is_attack else ALLOW,
        }
        results = {}
        for name, fn in baselines.items():
            random.seed(123)
            results[name] = run_baseline_policy(env, fn, n_episodes=15)

        results["RecurrentPPO"] = run_rl_policy(env, model, n_episodes=15)

        print(f"{'policy':15s} {'reward':>10s} {'std':>8s} {'precision':>10s} {'recall':>8s} {'f1':>8s}")
        for name, r in results.items():
            print(f"{name:15s} {r['mean_reward']:10.2f} {r['std_reward']:8.2f} {r['precision']:10.3f} {r['recall']:8.3f} {r['f1']:8.3f}")


if __name__ == "__main__":
    main()
