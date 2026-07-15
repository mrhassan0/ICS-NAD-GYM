"""Load harmonized ICS-NAD data per split (train/val/test), respecting the exact same
file-level/intra-file boundaries as ICSNADGym (Phase 1/2), for a leak-free supervised
baseline comparison. Binary target: state != BENIGN (matches the RL environment's
is_attack framing, so results are directly comparable to Phase 3).
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from icsnad_gym.env import PROTOCOL_CATEGORIES
from icsnad_gym.scalers import NUMERIC_FEATURES

DATA_DIR = Path("/home/user5/Desktop/MS THESIS/data")
PARQUET_DIR = DATA_DIR / "parquet"
LOAD_COLUMNS = NUMERIC_FEATURES + ["protocol", "state"]


def _protocol_onehot_matrix(protocols: list[str]) -> np.ndarray:
    mat = np.zeros((len(protocols), len(PROTOCOL_CATEGORIES)), dtype=np.float32)
    for i, p in enumerate(protocols):
        idx = PROTOCOL_CATEGORIES.index(p) if p in PROTOCOL_CATEGORIES else PROTOCOL_CATEGORIES.index("OTHER")
        mat[i, idx] = 1.0
    return mat


def load_split(split: str, max_rows: int | None = None, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y) for the given split. If max_rows is set, stratified-subsamples by
    binary label to keep classical ML training/eval tractable at this data scale."""
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    frames = []
    for fname, s in splits["file_level"].items():
        if s == split:
            frames.append(pl.read_parquet(PARQUET_DIR / fname, columns=LOAD_COLUMNS))
    for fname, meta in splits["intra_file"].items():
        lo, hi = meta["row_ranges"][split]
        df = pl.read_parquet(PARQUET_DIR / fname, columns=list(set(LOAD_COLUMNS + ["startOffset"])))
        df = df.sort("startOffset", maintain_order=True)[lo:hi].select(LOAD_COLUMNS)
        frames.append(df)

    full = pl.concat(frames, how="vertical_relaxed")
    # A handful of rows have null `state` (traced in Phase 1 to CSV parse-error recovery on
    # 2 corrupted source files) — treat as BENIGN rather than propagate null through the
    # boolean comparison, which would otherwise crash the int64 cast below.
    full = full.with_columns(pl.col("state").fill_null("BENIGN"))
    y_full = (full["state"] != "BENIGN").to_numpy().astype(np.int64)

    if max_rows is not None and len(full) > max_rows:
        rng = np.random.RandomState(seed)
        pos_idx = np.nonzero(y_full == 1)[0]
        neg_idx = np.nonzero(y_full == 0)[0]
        # Preserve the true class ratio in the subsample rather than forcing balance,
        # so evaluation metrics reflect real-world class imbalance.
        pos_frac = len(pos_idx) / len(y_full)
        n_pos = min(len(pos_idx), int(max_rows * pos_frac))
        n_neg = min(len(neg_idx), max_rows - n_pos)
        chosen = np.concatenate([
            rng.choice(pos_idx, n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int),
            rng.choice(neg_idx, n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int),
        ])
        rng.shuffle(chosen)
        full = full[chosen.tolist()]
        y_full = y_full[chosen]

    numeric = full.select(NUMERIC_FEATURES).fill_null(0.0).to_numpy().astype(np.float32)
    proto_onehot = _protocol_onehot_matrix(full["protocol"].to_list())
    X = np.concatenate([numeric, proto_onehot], axis=1)
    return X, y_full
