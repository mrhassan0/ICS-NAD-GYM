"""Parse Time_of_Attack log files and use them as a corroboration check (not primary
reconstruction) for the signature/rename-based label harmonization.

Each log lists ~19 repeated short trials with an attack onset offset *relative to each
trial*, e.g. "The 7 time of attack begins at 3 s". The absolute trial boundaries within a
capture file are not stated anywhere in the dataset and can't be reliably reconstructed
(captures vary in length, "part1" files are partial), so exact per-row timestamp alignment
isn't attempted. Instead this module counts contiguous attack bursts in the harmonized
`state` column and checks that the burst count is in the same ballpark as the number of
trials logged — an independent sanity check on the signature/rename reconstruction, not a
replacement for it.
"""

import re
from pathlib import Path

import polars as pl

_LINE_RE = re.compile(r"The (\d+) time of attack begins at (\d+(?:\.\d+)?) s")


def parse_time_of_attack_file(path: Path) -> list[tuple[int, float]]:
    """Return [(trial_index, onset_offset_seconds), ...] for one log file."""
    trials = []
    for line in path.read_text().splitlines():
        m = _LINE_RE.match(line.strip())
        if m:
            trials.append((int(m.group(1)), float(m.group(2))))
    return trials


def count_attack_bursts(df: pl.DataFrame) -> int:
    """Count contiguous non-BENIGN runs in time order (by startOffset)."""
    if "state" not in df.columns or len(df) == 0:
        return 0
    ordered = df.sort("startOffset")["state"]
    is_attack = (ordered != "BENIGN").to_list()
    bursts = 0
    prev = False
    for v in is_attack:
        if v and not prev:
            bursts += 1
        prev = v
    return bursts


def find_matching_log(time_of_attack_dir: Path, brand: str, attack_token: str) -> Path | None:
    """Find a Time_of_Attack log file matching brand + attack-type token, case-insensitive."""
    brand_l, token_l = brand.lower(), attack_token.lower()
    for sub in time_of_attack_dir.glob(f"time_of_attack_*{brand}*"):
        for f in sub.glob("*.txt"):
            if token_l in f.name.lower():
                return f
    return None


def corroborate(df: pl.DataFrame, brand: str, attack_token: str, time_of_attack_dir: Path) -> dict:
    """Compare detected burst count against the Time_of_Attack trial count for this file."""
    log_path = find_matching_log(time_of_attack_dir, brand, attack_token)
    if log_path is None:
        return {"time_of_attack_log": None, "n_trials_logged": None, "n_bursts_detected": None}
    trials = parse_time_of_attack_file(log_path)
    bursts = count_attack_bursts(df)
    return {
        "time_of_attack_log": str(log_path.name),
        "n_trials_logged": len(trials),
        "n_bursts_detected": bursts,
    }
