"""
Stress-test and validate tabular data before training or serving.

Simple, dependency-free checks (no Great Expectations required).
For GE: install great_expectations and wrap these expectations in a GE suite later.

Usage:
    from validation.data_validation import validate_processed_v2, ValidationError
    validate_processed_v2(df)  # raises ValidationError on failure

CLI:
    python -m validation.data_validation Data/processed_yt_v2.csv
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Raised when one or more data quality checks fail."""

    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("\n".join(messages))


def _check(name: str, ok: bool, detail: str = "") -> str | None:
    if not ok:
        return f"  FAIL: {name}" + (f" ({detail})" if detail else "")
    return None


def _no_inf_nan_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    errs = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if s.isna().any():
            n = int(s.isna().sum())
            errs.append(f"  FAIL: column '{c}' has {n} NaN values")
        if pd.api.types.is_numeric_dtype(s):
            arr = s.to_numpy()
            if np.isinf(arr).any():
                errs.append(f"  FAIL: column '{c}' contains inf")
    return errs


def validate_raw_youtube_snapshot(df: pd.DataFrame) -> None:
    """
    Validate raw youtube_data.csv-style snapshot before merging.
    """
    errors: list[str] = []
    required = ["channel_id", "view_count", "subscriber_count", "video_count"]
    for c in required:
        if c not in df.columns:
            errors.append(_check(f"required column '{c}'", False, "missing") or "")

    errors = [e for e in errors if e]

    if "channel_id" in df.columns:
        dups = df["channel_id"].duplicated().sum()
        if dups > 0:
            errors.append(f"  FAIL: duplicate channel_id rows: {dups}")

    for col in ["view_count", "subscriber_count", "video_count"]:
        if col in df.columns:
            if (df[col] < 0).any():
                errors.append(f"  FAIL: {col} must be >= 0")
            err = _check(col + " finite", not np.isinf(df[col]).any())
            if err:
                errors.append(err)

    if errors:
        raise ValidationError(["validate_raw_youtube_snapshot:"] + errors)


def validate_processed_v2(df: pd.DataFrame, strict: bool = True) -> None:
    """
    Validate processed_yt_v2 dataframe before save or after load.

    Checks:
      - Non-negative counts
      - years_active present and positive
      - Target in {0, 1}
      - No NaN/inf in model numeric columns
      - channel_id unique
    """
    errors: list[str] = []

    required = [
        "channel_id",
        "subscriber_count",
        "video_count",
        "years_active",
        "avg_views_per_video",
        "views_last_30_days",
        "upload_frequency",
        "engagement_ratio",
        "upload_consistency",
        "high_performing",
        "target_mode",
    ]
    for c in required:
        if c not in df.columns:
            errors.append(f"  FAIL: missing required column '{c}'")

    if not errors:
        if (df["subscriber_count"] < 0).any():
            errors.append("  FAIL: subscriber_count must be >= 0")
        if (df["video_count"] < 0).any():
            errors.append("  FAIL: video_count must be >= 0")
        if (df["views_last_30_days"] < 0).any():
            errors.append("  FAIL: views_last_30_days must be >= 0")

        if not df["years_active"].notna().all():
            errors.append("  FAIL: years_active must not be null")
        if df["years_active"].notna().any():
            if (df["years_active"] <= 0).any():
                errors.append("  FAIL: years_active must be > 0")
            if np.isinf(df["years_active"]).any():
                errors.append("  FAIL: years_active must be finite")

        if (df["avg_views_per_video"] < 0).any():
            errors.append("  FAIL: avg_views_per_video must be >= 0")
        if (df["upload_frequency"] < 0).any():
            errors.append("  FAIL: upload_frequency must be >= 0")
        if (df["upload_consistency"] < 0).any():
            errors.append("  FAIL: upload_consistency must be >= 0")

        if not df["high_performing"].isin([0, 1]).all():
            errors.append("  FAIL: high_performing must be 0 or 1")

        dup = df["channel_id"].duplicated().sum()
        if dup > 0:
            errors.append(f"  FAIL: duplicate channel_id: {dup} rows")

        numeric_cols = [
            "subscriber_count",
            "video_count",
            "years_active",
            "avg_views_per_video",
            "views_last_30_days",
            "upload_frequency",
            "engagement_ratio",
            "upload_consistency",
        ]
        errors.extend(_no_inf_nan_cols(df, numeric_cols))

        if "growth_6m" in df.columns:
            gm = df["growth_6m"]
            if gm.notna().any():
                sub = gm[~gm.isna()]
                if np.isinf(sub).any():
                    errors.append("  FAIL: growth_6m contains inf")
                if (sub < 0).any():
                    errors.append("  FAIL: growth_6m must be >= 0 when present")

    if errors:
        raise ValidationError(["validate_processed_v2:"] + errors)


def validate_data(df: pd.DataFrame) -> None:
    """
    Backwards-compatible alias: assert-style checks on core columns.
    Raises AssertionError or ValidationError.
    """
    assert (df["subscriber_count"] >= 0).all(), "subscriber_count must be >= 0"
    assert (df["video_count"] >= 0).all(), "video_count must be >= 0"
    assert df["years_active"].notnull().all(), "years_active must not be null"
    validate_processed_v2(df)


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: python -m validation.data_validation <path.csv>")
        return 1
    path = argv[0]
    df = pd.read_csv(path)
    try:
        validate_processed_v2(df)
        print(f"OK: {path} passed validate_processed_v2 ({len(df)} rows)")
    except ValidationError as e:
        print(f"FAILED: {path}\n{e}")
        return 1
    return 0


if __name__ == "__main__":
    main()
