"""Answers a real gap in the 3-way comparison: Phase 4 evaluated the 10 supervised
classifiers as static row-level predictors on held-out samples (F1/precision/recall only).
None of them were ever run *inside* ICSNADGym as an actual sequential policy — meaning
there was no reward number for them, only F1, which isn't the same currency the RL arms
are measured in. This wraps each trained classifier as a policy (predict attack -> BLOCK,
predict benign -> ALLOW) and runs it through the exact same simulator episodes used to
evaluate the RL models, so reward and F1 are both available for every model, on equal footing.

Predictions are batched per episode (all ~1024 rows at once), not called one row at a time
in the step loop — the flow sequence within an episode doesn't depend on the agent's
actions, so there's no need to pay Python-level per-call overhead 1024 times per episode.
A first version did exactly that and was still running after 46 minutes of CPU time with
zero output; this version finishes the same workload via ~30 batched predict() calls total
instead of ~300,000 individual ones.
"""

import numpy as np
import polars as pl
import joblib

from icsnad_gym.env import ICSNADGym, PROTOCOL_CATEGORIES, _protocol_onehot
from icsnad_gym.reward import ALLOW, BLOCK
from icsnad_gym.scalers import NUMERIC_FEATURES

MODELS_DIR = "/home/user5/Desktop/MS THESIS/data/baseline_models"
CLASSIFIER_NAMES = [
    "GBDT", "RandomForest", "AdaBoost", "LightGBM", "XGBoost",
    "ELM", "ANN", "KNN", "SVC", "GaussianNB",
]


def _batch_raw_features(df: pl.DataFrame) -> np.ndarray:
    """Vectorized equivalent of ICSNADGym._raw_features, for a whole episode's rows at once."""
    numeric = df.select(NUMERIC_FEATURES).fill_null(0.0).to_numpy().astype(np.float32)
    proto_mat = np.zeros((len(df), len(PROTOCOL_CATEGORIES)), dtype=np.float32)
    for i, p in enumerate(df["protocol"].to_list()):
        proto_mat[i] = _protocol_onehot(p)
    return np.concatenate([numeric, proto_mat], axis=1)


def run_classifier_policy(env: ICSNADGym, model, scaler, n_episodes: int) -> dict:
    totals, tp, fp, fn_, tn = [], 0, 0, 0, 0
    for _ in range(n_episodes):
        obs, info = env.reset()

        # Batch-predict the whole episode's flows in one shot.
        raw = _batch_raw_features(env._episode_df)
        scaled = scaler.transform(raw)
        preds = model.predict(scaled)

        done = False
        total = 0.0
        step_idx = 0
        while not done:
            pred = int(preds[step_idx])
            action = BLOCK if pred == 1 else ALLOW
            is_attack = info["is_attack"]
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if is_attack and pred == 1:
                tp += 1
            elif is_attack and pred == 0:
                fn_ += 1
            elif not is_attack and pred == 1:
                fp += 1
            else:
                tn += 1
            step_idx += 1
            done = terminated or truncated
        totals.append(total)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn_) if (tp + fn_) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"mean_reward": float(np.mean(totals)), "std_reward": float(np.std(totals)),
            "precision": precision, "recall": recall, "f1": f1}


def main():
    scaler = joblib.load(f"{MODELS_DIR}/scaler.joblib")

    for split in ["val", "test"]:
        print(f"\n{'='*20} split={split} {'='*20}", flush=True)
        env = ICSNADGym(split=split, episode_length=1024, seed=123)
        print(f"{'model':15s} {'reward':>10s} {'std':>8s} {'precision':>10s} {'recall':>8s} {'f1':>8s}", flush=True)
        for name in CLASSIFIER_NAMES:
            model = joblib.load(f"{MODELS_DIR}/{name}.joblib")
            r = run_classifier_policy(env, model, scaler, n_episodes=15)
            print(f"{name:15s} {r['mean_reward']:10.2f} {r['std_reward']:8.2f} "
                  f"{r['precision']:10.3f} {r['recall']:8.3f} {r['f1']:8.3f}", flush=True)


if __name__ == "__main__":
    main()
