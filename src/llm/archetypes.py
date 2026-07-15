"""Cluster train-split flows into K statistical archetypes (k-means), each summarized by
its centroid feature statistics and dominant ground-truth label distribution. The LLM
scores each archetype's risk once (offline, cached) — the RL environment only ever needs
to look up which archetype a flow is closest to, never call the LLM in the training hot loop.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from baselines.data import load_split
from icsnad_gym.scalers import NUMERIC_FEATURES
from icsnad_gym.env import PROTOCOL_CATEGORIES

ARCHETYPE_DIR = Path("/home/user5/Desktop/MS THESIS/data/archetypes")
N_CLUSTERS = 25
TRAIN_SAMPLE_ROWS = 150_000

FEATURE_NAMES = NUMERIC_FEATURES + [f"protocol_{c}" for c in PROTOCOL_CATEGORIES]

# A handful of features that most directly characterize an attack signature, used to build
# a human-readable archetype description for the LLM prompt (rather than dumping all 55 dims).
DESCRIPTIVE_FEATURES = [
    "sSynRate", "sAckRate", "sFinRate", "sRstRate", "sFragmentRate",
    "duration", "sPackets", "sBytesAvg", "sLoad",
]


def build_archetypes():
    ARCHETYPE_DIR.mkdir(parents=True, exist_ok=True)

    # load_split returns (X, y) with y = binary is_attack; reload with labels for archetype
    # description purposes by re-deriving from the same underlying loader's label column.
    X, y = load_split("train", max_rows=TRAIN_SAMPLE_ROWS)

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit(X_s)
    labels = km.labels_

    archetypes = []
    for c in range(N_CLUSTERS):
        mask = labels == c
        n = int(mask.sum())
        if n == 0:
            continue
        centroid_raw = X[mask].mean(axis=0)
        attack_rate = float(y[mask].mean())
        stats = {name: float(val) for name, val in zip(FEATURE_NAMES, centroid_raw)}
        archetypes.append({
            "cluster_id": c,
            "n_samples": n,
            "attack_rate": attack_rate,
            "descriptive_stats": {k: round(stats[k], 4) for k in DESCRIPTIVE_FEATURES},
        })

    (ARCHETYPE_DIR / "archetypes.json").write_text(json.dumps(archetypes, indent=2))
    np.save(ARCHETYPE_DIR / "centroids.npy", km.cluster_centers_)
    import joblib
    joblib.dump(scaler, ARCHETYPE_DIR / "scaler.joblib")
    joblib.dump(km, ARCHETYPE_DIR / "kmeans.joblib")

    print(f"Built {len(archetypes)} archetypes from {len(X):,} rows")
    for a in sorted(archetypes, key=lambda a: -a["attack_rate"])[:5]:
        print(f"  cluster {a['cluster_id']}: n={a['n_samples']}, attack_rate={a['attack_rate']:.3f}, "
              f"sSynRate={a['descriptive_stats']['sSynRate']:.2f}, sFragmentRate={a['descriptive_stats']['sFragmentRate']:.2f}")
    return archetypes


if __name__ == "__main__":
    build_archetypes()
