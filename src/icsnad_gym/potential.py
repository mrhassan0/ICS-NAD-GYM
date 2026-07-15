"""Potential function Φ(s) for LLM-based reward shaping: a weighted similarity of the
current flow to the K LLM-scored archetypes (Phase 5), used as a potential-based shaping
term F(s,s') = γΦ(s') − Φ(s) — provably policy-invariant (Ng, Harada & Russell, 1999), so
any measured effect on training is attributable to the LLM's guidance quality, not a
changed reward optimum. Archetype lookup only (no LLM calls) — safe for the RL hot loop.
"""

import json
from pathlib import Path

import joblib
import numpy as np

ARCHETYPE_DIR = Path("/home/user5/Desktop/MS THESIS/data/archetypes")
_SOFTMAX_TEMPERATURE = 50.0  # controls how sharply Φ(s) favors the nearest archetype(s)


class ArchetypePotential:
    """Loads the fitted archetype scaler/centroids/LLM risk scores once; call `.phi(raw_features)`
    per step. `raw_features` must be in the same NUMERIC_FEATURES + protocol-one-hot order
    used by src/baselines/data.py and src/llm/archetypes.py."""

    def __init__(self):
        self.scaler = joblib.load(ARCHETYPE_DIR / "scaler.joblib")
        self.centroids = np.load(ARCHETYPE_DIR / "centroids.npy")
        archetypes = json.loads((ARCHETYPE_DIR / "archetypes.json").read_text())
        # centroids.npy rows are ordered by KMeans cluster index; archetypes.json entries
        # carry their own cluster_id — align risk scores to centroid row order explicitly.
        risk_by_cluster = {a["cluster_id"]: a.get("llm_risk", 5) for a in archetypes}
        self.risks = np.array([risk_by_cluster.get(c, 5) for c in range(len(self.centroids))], dtype=np.float32)

    def phi(self, raw_features: np.ndarray) -> float:
        x = self.scaler.transform(raw_features.reshape(1, -1))
        dists = np.linalg.norm(self.centroids - x, axis=1)
        # Standard softmax numerical-stability trick: subtract the min distance before
        # exponentiating. Without this, exp(-dist/T) underflows to 0 for every cluster when
        # dist >> T (observed: distances ~190-350 with the original T=2 gave exp(-96) for
        # every cluster — weights became floating-point noise, not a meaningful signal).
        shifted = dists - dists.min()
        weights = np.exp(-shifted / _SOFTMAX_TEMPERATURE)
        weights /= weights.sum()
        return float(np.dot(weights, self.risks))
