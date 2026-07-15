"""ICSNADGym: a Gymnasium environment over real ICS-NAD flow telemetry.

Episodes are contiguous, time-ordered windows of flows sampled from a single capture file
(never synthetic cross-file mixes — different files are temporally/physically unrelated
sessions). The agent classifies/responds to each flow in sequence; BLOCK triggers a
stateful quarantine, which is what makes this a genuinely sequential decision problem
rather than a per-flow classifier.
"""

import json
import random
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

from icsnad_gym.potential import ArchetypePotential
from icsnad_gym.quarantine import QuarantineState
from icsnad_gym.reward import (
    ALLOW, ALERT, BLOCK, base_reward, detection_latency_bonus,
    potential_based_shaping, quarantine_ongoing_reward,
)
from icsnad_gym.scalers import NUMERIC_FEATURES, brand_of_parquet

DATA_DIR = Path("/home/user5/Desktop/MS THESIS/data")
PARQUET_DIR = DATA_DIR / "parquet"

PROTOCOL_CATEGORIES = ["IPV4-TCP", "IPV4-UDP", "IPV4-ICMP", "ARP", "IPV6", "OTHER"]
LOAD_COLUMNS = NUMERIC_FEATURES + ["protocol", "state", "sAddress"]
FILE_CACHE_SIZE = 5  # bounded per-process LRU: avoids unbounded memory growth over long runs


def _protocol_onehot(proto: str) -> np.ndarray:
    vec = np.zeros(len(PROTOCOL_CATEGORIES), dtype=np.float32)
    idx = PROTOCOL_CATEGORIES.index(proto) if proto in PROTOCOL_CATEGORIES else PROTOCOL_CATEGORIES.index("OTHER")
    vec[idx] = 1.0
    return vec


class ICSNADGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, split: str = "train", episode_length: int = 4096,
                 quarantine_duration: int = 50, attack_window_bias: float = 0.7, seed: int | None = None,
                 use_llm_shaping: bool = False):
        super().__init__()
        self.split = split
        self.episode_length = episode_length
        self.attack_window_bias = attack_window_bias
        self.quarantine = QuarantineState(duration=quarantine_duration)
        self._rng = random.Random(seed)

        # Config-toggleable, default off: existing sanity checks / RL-only baseline behavior
        # (Phase 2/3) are unaffected unless a caller explicitly opts in for the RL+LLM arm.
        self.use_llm_shaping = use_llm_shaping
        self._llm_potential = ArchetypePotential() if use_llm_shaping else None

        self.splits = json.loads((DATA_DIR / "splits.json").read_text())
        self.scalers = json.loads((DATA_DIR / "scalers.json").read_text())
        self._eligible = self._build_eligible_files()
        if not self._eligible:
            raise ValueError(f"No eligible files found for split={split!r}")

        n_features = len(NUMERIC_FEATURES) + len(PROTOCOL_CATEGORIES)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._episode_df: pl.DataFrame | None = None
        self._brand: str | None = None
        self._ptr = 0
        self._flows_since_onset = 0
        self._current_run_detected = True  # no active run at episode start
        self._file_cache: OrderedDict[str, pl.DataFrame] = OrderedDict()

    def _build_eligible_files(self) -> list[dict]:
        eligible = []
        for fname, split in self.splits["file_level"].items():
            if split == self.split:
                n = pl.read_parquet(PARQUET_DIR / fname, columns=["state"]).height
                eligible.append({"file": fname, "row_range": (0, n)})
        for fname, meta in self.splits["intra_file"].items():
            lo, hi = meta["row_ranges"][self.split]
            eligible.append({"file": fname, "row_range": (lo, hi)})
        return eligible

    def _get_cached_file(self, fname: str) -> pl.DataFrame:
        if fname in self._file_cache:
            self._file_cache.move_to_end(fname)
            return self._file_cache[fname]
        df = pl.read_parquet(PARQUET_DIR / fname, columns=LOAD_COLUMNS)
        df = df.sort("startOffset", maintain_order=True)
        self._file_cache[fname] = df
        if len(self._file_cache) > FILE_CACHE_SIZE:
            self._file_cache.popitem(last=False)
        return df

    def _load_window(self) -> pl.DataFrame:
        choice = self._rng.choice(self._eligible)
        fname, (lo, hi) = choice["file"], choice["row_range"]
        n_available = hi - lo
        L = min(self.episode_length, n_available)

        full = self._get_cached_file(fname)[lo:hi]

        is_attack = (full["state"] != "BENIGN").to_numpy()
        attack_idxs = np.nonzero(is_attack)[0]

        if len(attack_idxs) > 0 and self._rng.random() < self.attack_window_bias:
            center = int(self._rng.choice(attack_idxs))
            start = max(0, min(n_available - L, center - L // 2))
        else:
            start = self._rng.randint(0, max(0, n_available - L))

        self._brand = brand_of_parquet(fname)
        return full[start:start + L]

    def _raw_features(self, row: dict) -> np.ndarray:
        """Unnormalized NUMERIC_FEATURES + protocol one-hot, matching the exact feature
        construction used to fit the archetype scaler (src/llm/archetypes.py via
        src/baselines/data.py) — the archetype potential function expects this space, not
        the per-brand normalized observation the agent sees."""
        vals = np.array([row[c] if row[c] is not None else 0.0 for c in NUMERIC_FEATURES], dtype=np.float32)
        return np.concatenate([vals, _protocol_onehot(row["protocol"])]).astype(np.float32)

    def _observe(self, row: dict) -> np.ndarray:
        scaler = self.scalers[self._brand]
        vals = np.array(
            [(row[c] if row[c] is not None else 0.0) - scaler["mean"][c] for c in NUMERIC_FEATURES],
            dtype=np.float32,
        )
        stds = np.array([scaler["std"][c] for c in NUMERIC_FEATURES], dtype=np.float32)
        vals = vals / stds
        proto_vec = _protocol_onehot(row["protocol"])
        return np.concatenate([vals, proto_vec]).astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        self._episode_df = self._load_window()
        self.quarantine.reset()
        self._ptr = 0
        self._flows_since_onset = 0
        self._current_run_detected = True
        row = self._episode_df.row(self._ptr, named=True)
        info = {"is_attack": bool(row["state"] != "BENIGN"), "source": row["sAddress"]}
        return self._observe(row), info

    def step(self, action: int):
        row = self._episode_df.row(self._ptr, named=True)
        is_attack = row["state"] != "BENIGN"
        source = row["sAddress"]

        if self.quarantine.is_quarantined(source):
            reward = quarantine_ongoing_reward(self.quarantine.was_correct_block(source))
        else:
            reward = base_reward(is_attack, action)
            if action == BLOCK:
                self.quarantine.start(source, was_correct=is_attack)

        # Detection-latency bonus: reward fast detection of a fresh attack run.
        if is_attack:
            if not self._current_run_detected:
                self._flows_since_onset += 1
                if action in (ALERT, BLOCK):
                    reward += detection_latency_bonus(self._flows_since_onset)
                    self._current_run_detected = True
        else:
            self._current_run_detected = True  # run ended; next attack row starts a fresh onset
            self._flows_since_onset = 0

        self.quarantine.tick()

        # LLM-shaped potential term F(s,s') = γΦ(s') − Φ(s), added on top of the base
        # reward above — never replaces it, so the shaping is provably policy-invariant
        # (Ng/Harada/Russell 1999) regardless of archetype/LLM quality.
        if self.use_llm_shaping:
            phi_s = self._llm_potential.phi(self._raw_features(row))

        self._ptr += 1
        terminated = False
        truncated = self._ptr >= len(self._episode_df)

        if truncated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"is_attack": False, "source": None}
            if self.use_llm_shaping:
                reward += potential_based_shaping(phi_s, 0.0)
        else:
            next_row = self._episode_df.row(self._ptr, named=True)
            obs = self._observe(next_row)
            if self.use_llm_shaping:
                phi_s_next = self._llm_potential.phi(self._raw_features(next_row))
                reward += potential_based_shaping(phi_s, phi_s_next)
            # info describes the row matching `obs` (the NEXT decision), not the row just
            # acted on above — a policy that reads info to choose its next action must see
            # ground truth for the row it's about to act on, not the one already scored.
            info = {"is_attack": bool(next_row["state"] != "BENIGN"), "source": next_row["sAddress"]}

        return obs, float(reward), terminated, truncated, info
