"""Train/val/test split assignment for the harmonized ICS-NAD Parquet corpus.

Policy (per the approved thesis plan): file-level holdout wherever an attack type has
multiple files (avoids leaking the same capture session across train/eval), falling back
to a time-ordered intra-file split only for attack types that exist in exactly one file
(confirmed: only NMAP and ICMPREDIRECT, out of 20 canonical attack types).

File-level assignment per label's (deterministically sorted) file list:
  - 1 file:      handled separately via intra-file time split
  - 2 files:     [train, test]
  - 3 files:     [train, train, test]
  - >=4 files:   [train, ..., val, test]  (last two held out, rest train)
"""

import hashlib
import json
from pathlib import Path

import polars as pl

PARQUET_DIR = Path("/home/user5/Desktop/MS THESIS/data/parquet")
SPLITS_PATH = Path("/home/user5/Desktop/MS THESIS/data/splits.json")

INTRA_FILE_SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
INTRA_FILE_GAP_FRACTION = 0.02  # small buffer between segments to reduce boundary leakage


def _shuffle_key(fname: str) -> str:
    # Deterministic but brand-name-independent ordering: plain alphabetical sort would put
    # "ABB_*" first and "Siemens_*" last for every multi-brand label, so Siemens (or whichever
    # brand sorts last) would *never* land in train across the whole corpus. Hash-based
    # ordering decorrelates split assignment from brand name while staying reproducible.
    return hashlib.md5(fname.encode()).hexdigest()


def _assign_file_level(files: list[str]) -> dict[str, str]:
    files = sorted(files, key=_shuffle_key)
    n = len(files)
    if n == 2:
        return {files[0]: "train", files[1]: "test"}
    if n == 3:
        return {files[0]: "train", files[1]: "train", files[2]: "test"}
    # n >= 4
    assignment = {f: "train" for f in files[:-2]}
    assignment[files[-2]] = "val"
    assignment[files[-1]] = "test"
    return assignment


def build_splits() -> dict:
    files = sorted(PARQUET_DIR.glob("*.parquet"))
    label_to_files: dict[str, list[str]] = {}
    for f in files:
        labels = pl.read_parquet(f, columns=["state"])["state"].unique().to_list()
        for lab in labels:
            if lab and lab.upper() != "BENIGN":
                label_to_files.setdefault(lab, []).append(f.name)

    file_split: dict[str, str] = {}
    intra_file: dict[str, dict] = {}
    conflicts = []

    for label, fnames in label_to_files.items():
        if len(fnames) == 1:
            fname = fnames[0]
            df = pl.read_parquet(PARQUET_DIR / fname, columns=["startOffset"]).sort("startOffset", maintain_order=True)
            n = len(df)
            gap = int(n * INTRA_FILE_GAP_FRACTION)
            train_end = int(n * INTRA_FILE_SPLIT_FRACTIONS["train"])
            val_end = train_end + gap + int(n * INTRA_FILE_SPLIT_FRACTIONS["val"])
            intra_file[fname] = {
                "label": label,
                "fractions": INTRA_FILE_SPLIT_FRACTIONS,
                "n_rows": n,
                "row_ranges": {
                    "train": [0, train_end],
                    "val": [train_end + gap, val_end],
                    "test": [val_end + gap, n],
                },
                "note": "row indices after sorting by startOffset ascending; small gaps left between segments",
            }
            continue
        assignment = _assign_file_level(fnames)
        for fname, split in assignment.items():
            if fname in file_split and file_split[fname] != split:
                conflicts.append((fname, file_split[fname], split, label))
            file_split[fname] = split

    if conflicts:
        raise ValueError(f"Split assignment conflicts (file claimed by multiple labels differently): {conflicts}")

    # Any file with only BENIGN rows (shouldn't happen given every file was captured for a
    # specific attack, but guard anyway) defaults to train.
    for f in files:
        if f.name not in file_split and f.name not in intra_file:
            file_split[f.name] = "train"

    manifest = {"file_level": file_split, "intra_file": intra_file}
    SPLITS_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def _brand_of(fname: str) -> str:
    if fname.startswith("Siemens"):
        return "Siemens"
    if fname.startswith("Schneider"):
        return "Schneider"
    return "ABB"


if __name__ == "__main__":
    m = build_splits()
    from collections import Counter
    print("file-level split counts:", Counter(m["file_level"].values()))
    print("intra-file split files:", list(m["intra_file"].keys()))

    print("\nBrand representation by split (file counts):")
    brand_split = Counter()
    for fname, split in m["file_level"].items():
        brand_split[(_brand_of(fname), split)] += 1
    for brand in ["ABB", "Schneider", "Siemens"]:
        row = {s: brand_split.get((brand, s), 0) for s in ["train", "val", "test"]}
        flag = "  <-- WARNING: zero train files" if row["train"] == 0 else ""
        print(f"  {brand:10s} {row}{flag}")

    print(f"\nWritten to {SPLITS_PATH}")
