"""Explicit CSV schema for ICS-NAD Labeled_CSV files.

Relying on Polars schema inference is unsafe here: some multi-million-row files have
columns that look like integers in the first N rows but contain floats later on (e.g.
`sPackets` starts as whole numbers then hits `0.0`), which crashes `read_csv` mid-file.
Declaring every column's dtype up front avoids inference entirely.
"""

import polars as pl

STRING_COLUMNS = {
    "sAddress", "rAddress", "sMACs", "rMACs", "sIPs", "rIPs",
    "protocol", "startDate", "endDate", "state",
}

_ALL_COLUMNS = [
    "sAddress", "rAddress", "sMACs", "rMACs", "sIPs", "rIPs", "protocol", "startDate", "endDate",
    "start", "end", "startOffset", "endOffset", "duration", "sPackets", "rPackets",
    "sBytesSum", "rBytesSum", "sBytesMax", "rBytesMax", "sBytesMin", "rBytesMin", "sBytesAvg", "rBytesAvg",
    "sLoad", "rLoad", "sPayloadSum", "rPayloadSum", "sPayloadMax", "rPayloadMax", "sPayloadMin", "rPayloadMin",
    "sPayloadAvg", "rPayloadAvg", "sInterPacketAvg", "rInterPacketAvg", "sttl", "rttl",
    "sAckRate", "rAckRate", "sUrgRate", "rUrgRate", "sFinRate", "rFinRate", "sPshRate", "rPshRate",
    "sSynRate", "rSynRate", "sRstRate", "rRstRate", "sWinTCP", "rWinTCP",
    "sFragmentRate", "rFragmentRate", "sAckDelayMax", "rAckDelayMax", "sAckDelayMin", "rAckDelayMin",
    "sAckDelayAvg", "rAckDelayAvg", "state",
]

SCHEMA: dict[str, pl.DataType] = {
    col: (pl.Utf8 if col in STRING_COLUMNS else pl.Float64) for col in _ALL_COLUMNS
}


def read_csv_typed(path, **kwargs) -> pl.DataFrame:
    """Read an ICS-NAD CSV with the fixed schema, tolerating files missing the `state` column.

    `ignore_errors=True` is required: at least one file (ABB_LONG_ackflood_faster) has a
    single truncated row from an interrupted write during original capture (confirmed via a
    raw comma-count scan: 1 row out of ~5M with 30 fields instead of 61). Dropping such rows
    is safe at this scale (<0.0001% of any affected file) and far more robust than a
    format-specific repair pass. A second file (ABB_LONG_icmpflood_faster) has the opposite
    issue (rows with more fields than the schema) hence `truncate_ragged_lines`. Any resulting
    null `state` values are handled by the caller.
    """
    header = pl.read_csv(path, n_rows=0).columns
    schema = {col: dtype for col, dtype in SCHEMA.items() if col in header}
    kwargs.setdefault("ignore_errors", True)
    kwargs.setdefault("truncate_ragged_lines", True)
    return pl.read_csv(path, schema_overrides=schema, **kwargs)
