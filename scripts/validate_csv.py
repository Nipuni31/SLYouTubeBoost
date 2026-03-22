"""
CLI: validate a processed CSV (same rules as data_prep_v2 output).

  python scripts/validate_csv.py Data/processed_yt_v2.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.data_validation import ValidationError, validate_processed_v2, validate_raw_youtube_snapshot

import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_csv.py <path.csv> [--raw]")
        sys.exit(1)
    path = sys.argv[1]
    raw = "--raw" in sys.argv
    df = pd.read_csv(path)
    try:
        if raw:
            validate_raw_youtube_snapshot(df)
        else:
            validate_processed_v2(df)
        print(f"OK: {path} ({len(df)} rows)")
    except ValidationError as e:
        print(f"FAILED:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
