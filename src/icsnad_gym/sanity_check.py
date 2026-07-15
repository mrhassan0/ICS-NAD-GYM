"""Mandatory sanity-floor check, per the approved plan: run before trusting the reward
function or training loop. Random / always-ALLOW / always-ALERT / always-BLOCK / oracle
policies must clearly separate in total episode reward, in the expected order, before any
RL training result from this environment should be trusted.
"""

import random

import numpy as np

from icsnad_gym.env import ICSNADGym
from icsnad_gym.reward import ALLOW, ALERT, BLOCK


def run_policy(env, policy_fn, n_episodes: int) -> list[float]:
    totals = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        total = 0.0
        is_attack = info.get("is_attack", False)  # first-step context filled after first step
        while not done:
            action = policy_fn(is_attack)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            is_attack = info["is_attack"]
            done = terminated or truncated
        totals.append(total)
    return totals


def main(n_episodes: int = 20, episode_length: int = 1024, seed: int = 0):
    env = ICSNADGym(split="train", episode_length=episode_length, seed=seed)

    policies = {
        "random": lambda is_attack: random.choice([ALLOW, ALERT, BLOCK]),
        "always_ALLOW": lambda is_attack: ALLOW,
        "always_ALERT": lambda is_attack: ALERT,
        "always_BLOCK": lambda is_attack: BLOCK,
        "oracle": lambda is_attack: BLOCK if is_attack else ALLOW,
    }

    results = {}
    for name, fn in policies.items():
        totals = run_policy(env, fn, n_episodes)
        results[name] = (float(np.mean(totals)), float(np.std(totals)))

    print(f"{'policy':15s} {'mean_reward':>14s} {'std':>10s}")
    for name, (mean, std) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:15s} {mean:14.2f} {std:10.2f}")

    # Basic ordering assertion: oracle should clearly beat everything else.
    oracle_mean = results["oracle"][0]
    others_max = max(m for k, (m, s) in results.items() if k != "oracle")
    print(f"\noracle beats best non-oracle policy by: {oracle_mean - others_max:.2f}")
    assert oracle_mean > others_max, "SANITY FAILURE: oracle does not dominate — reward function is suspect"
    print("SANITY CHECK PASSED: oracle dominates all baseline policies")


if __name__ == "__main__":
    main()
