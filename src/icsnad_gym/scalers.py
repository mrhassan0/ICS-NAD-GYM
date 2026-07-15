"""Fit per-brand z-score scalers on the TRAIN split only, over the numeric flow features
that make up the RL/supervised observation space (identity fields like IPs/MACs/absolute
timestamps are excluded per the approved plan's observation-space design, so the policy
can't shortcut by memorizing fixed addresses)."""

import json
from pathlib import Path

import polars as pl

PARQUET_DIR = Path("/home/user5/Desktop/MS THESIS/data/parquet")
SPLITS_PATH = Path("/home/user5/Desktop/MS THESIS/data/splits.json")
SCALERS_PATH = Path("/home/user5/Desktop/MS THESIS/data/scalers.json")

# Numeric observation features: excludes identity (sAddress/rAddress/sMACs/rMACs/sIPs/rIPs),
# absolute time (startDate/endDate/start/end), protocol (one-hot, not z-scored), and state/
# label_uncertain (not observation features).
NUMERIC_FEATURES = [
    "startOffset", "endOffset", "duration", "sPackets", "rPackets",
    "sBytesSum", "rBytesSum", "sBytesMax", "rBytesMax", "sBytesMin", "rBytesMin", "sBytesAvg", "rBytesAvg",
    "sLoad", "rLoad", "sPayloadSum", "rPayloadSum", "sPayloadMax", "rPayloadMax",
    "sPayloadMin", "rPayloadMin", "sPayloadAvg", "rPayloadAvg",
    "sInterPacketAvg", "rInterPacketAvg", "sttl", "rttl",
    "sAckRate", "rAckRate", "sUrgRate", "rUrgRate", "sFinRate", "rFinRate",
    "sPshRate", "rPshRate", "sSynRate", "rSynRate", "sRstRate", "rRstRate",
    "sWinTCP", "rWinTCP", "sFragmentRate", "rFragmentRate",
    "sAckDelayMax", "rAckDelayMax", "sAckDelayMin", "rAckDelayMin", "sAckDelayAvg", "rAckDelayAvg",
]


def brand_of_parquet(fname: str) -> str:
    if fname.startswith("Siemens"):
        return "Siemens"
    if fname.startswith("Schneider"):
        return "Schneider"
    return "ABB"


def _train_rows_for_file(fname: str, splits: dict) -> pl.DataFrame | None:
    path = PARQUET_DIR / fname
    if fname in splits["file_level"]:
        if splits["file_level"][fname] != "train":
            return None
        return pl.read_parquet(path, columns=NUMERIC_FEATURES)
    if fname in splits["intra_file"]:
        lo, hi = splits["intra_file"][fname]["row_ranges"]["train"]
        df = pl.read_parquet(path, columns=NUMERIC_FEATURES).sort("startOffset", maintain_order=True)
        return df[lo:hi]
    return None


def fit_scalers() -> dict:
    splits = json.loads(SPLITS_PATH.read_text())
    all_files = sorted(PARQUET_DIR.glob("*.parquet"))

    brand_frames: dict[str, list[pl.DataFrame]] = {"ABB": [], "Schneider": [], "Siemens": []}
    for f in all_files:
        df = _train_rows_for_file(f.name, splits)
        if df is not None:
            brand_frames[brand_of_parquet(f.name)].append(df)

    scalers = {}
    for brand, frames in brand_frames.items():
        if not frames:
            continue
        full = pl.concat(frames)
        means, stds = {}, {}
        for col in NUMERIC_FEATURES:
            series = full[col].fill_null(0.0)
            m = series.mean()
            s = series.std()
            means[col] = float(m) if m is not None else 0.0
            stds[col] = float(s) if s is not None and s > 1e-9 else 1.0  # avoid div-by-zero on constant columns
        scalers[brand] = {"mean": means, "std": stds, "n_train_rows": len(full)}

    SCALERS_PATH.write_text(json.dumps(scalers, indent=2))
    return scalers


if __name__ == "__main__":
    s = fit_scalers()
    for brand, d in s.items():
        print(f"{brand}: fit on {d['n_train_rows']:,} train rows")
    print(f"\nWritten to {SCALERS_PATH}")
