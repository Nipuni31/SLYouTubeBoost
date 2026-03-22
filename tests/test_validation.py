"""Tests for validation.data_validation (no real CSV required)."""

import pandas as pd
import pytest

from validation.data_validation import (
    ValidationError,
    validate_processed_v2,
    validate_raw_youtube_snapshot,
)


def _minimal_processed_valid() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel_id": ["a", "b"],
            "subscriber_count": [100, 200],
            "video_count": [10, 20],
            "years_active": [1.0, 2.0],
            "avg_views_per_video": [100.0, 200.0],
            "views_last_30_days": [1.0, 2.0],
            "upload_frequency": [0.5, 1.0],
            "engagement_ratio": [0.1, 0.2],
            "upload_consistency": [0.0, 0.1],
            "high_performing": [0, 1],
            "target_mode": ["engagement_ratio", "engagement_ratio"],
        }
    )


def test_validate_processed_v2_passes():
    df = _minimal_processed_valid()
    validate_processed_v2(df)


def test_validate_processed_v2_fails_negative_subs():
    df = _minimal_processed_valid()
    df.loc[0, "subscriber_count"] = -1
    with pytest.raises(ValidationError):
        validate_processed_v2(df)


def test_validate_raw_youtube_snapshot_passes():
    df = pd.DataFrame(
        {
            "channel_id": ["x"],
            "view_count": [1],
            "subscriber_count": [1],
            "video_count": [1],
        }
    )
    validate_raw_youtube_snapshot(df)
