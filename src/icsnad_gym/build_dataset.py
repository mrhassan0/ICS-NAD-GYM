"""Phase 1 entry point: harmonize labels across all 61 ICS-NAD Labeled_CSV files,
corroborate against Time_of_Attack where applicable, write Parquet outputs, and
produce a label-quality report.

Usage: python -m icsnad_gym.build_dataset
"""

import json
from pathlib import Path

from icsnad_gym.label_harmonizer import harmonize_file, FILE_STRATEGY, expected_label_from_filename
from icsnad_gym.time_of_attack import corroborate

RAW_DIR = Path("/home/user5/Desktop/MS THESIS/datasets/ICS-NAD/Labeled_CSV")
TOA_DIR = Path("/home/user5/Desktop/MS THESIS/datasets/ICS-NAD/Time_of_Attack")
OUT_DIR = Path("/home/user5/Desktop/MS THESIS/data/parquet")
REPORT_PATH = Path("/home/user5/Desktop/MS THESIS/data/label_quality_report.json")

_FLOOD_TOKENS = {"SYNFLOOD", "TCPFLOOD", "ACKFLOOD", "ICMPFLOOD"}


def brand_of(path: Path) -> str:
    parent = path.parent.name
    if parent.startswith("Siemens"):
        return "Siemens"
    return parent  # "ABB" or "Schneider"


def brand_glob_variants(brand: str) -> list[str]:
    # Time_of_Attack folder names misspell Siemens as "Seimens" for some dates.
    return [brand, "Seimens"] if brand == "Siemens" else [brand]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("**/*.csv"))
    print(f"Found {len(files)} CSV files")

    all_reports = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name} ...", flush=True)
        try:
            df, report = harmonize_file(f)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_reports.append({"file": f.name, "method": "ERROR", "error": str(e)})
            continue
        report["brand"] = brand_of(f)

        expected = report.get("expected_label")
        if expected in _FLOOD_TOKENS:
            best = {"n_trials_logged": None, "n_bursts_detected": None}
            for brand_variant in brand_glob_variants(report["brand"]):
                c = corroborate(df, brand_variant, expected.lower(), TOA_DIR)
                if c["n_trials_logged"] is not None:
                    best = c
                    break
            report.update(best)

        out_path = OUT_DIR / f"{f.stem}.parquet"
        df.write_parquet(out_path)
        report["parquet_path"] = str(out_path)
        all_reports.append(report)

    REPORT_PATH.write_text(json.dumps(all_reports, indent=2, default=str))

    # Console summary
    ok_reports = [r for r in all_reports if r["method"] != "ERROR"]
    total_rows = sum(r["n_rows"] for r in ok_reports)
    total_uncertain = sum(
        r.get("n_attack_rows", 0) if r["method"] == "signature" else r.get("n_uncertain_rows", 0)
        for r in ok_reports
    )
    by_method = {}
    for r in all_reports:
        by_method.setdefault(r["method"], []).append(r["file"])

    print("\n=== SUMMARY ===")
    print(f"Total files: {len(all_reports)}, total rows: {total_rows:,}, uncertain/reconstructed rows: {total_uncertain:,}")
    for method, fnames in by_method.items():
        print(f"  {method}: {len(fnames)} files")
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
