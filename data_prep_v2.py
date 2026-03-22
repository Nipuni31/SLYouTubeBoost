"""
Build processed_yt_v2.csv: time-aware features, no label leakage in trained features, meaningful target.

Optional inputs:
  - Data/channel_enriched.csv (from scripts/fetch_youtube_channels.py)
  - Data/videos_flat.csv (from scripts/fetch_youtube_videos.py)

Run: python data_prep_v2.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from validation.data_validation import ValidationError, validate_processed_v2, validate_raw_youtube_snapshot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
RAW = os.path.join(DATA_DIR, "youtube_data.csv")
CHANNEL_ENRICHED = os.path.join(DATA_DIR, "channel_enriched.csv")
VIDEOS_FLAT = os.path.join(DATA_DIR, "videos_flat.csv")
OUT = os.path.join(DATA_DIR, "processed_yt_v2.csv")


def main():
    base = pd.read_csv(RAW)
    base = base.drop_duplicates(subset=["channel_id"], keep="first").reset_index(drop=True)
    try:
        validate_raw_youtube_snapshot(base)
    except ValidationError as e:
        print("Raw data validation failed:\n", e)
        raise
    base["total_views"] = base["view_count"]

    if os.path.exists(CHANNEL_ENRICHED):
        ch = pd.read_csv(CHANNEL_ENRICHED)
        df = base.merge(ch, on="channel_id", how="left", suffixes=("_snap", "_api"))
        for c in ["subscriber_count", "video_count", "total_views"]:
            snap, api = c + "_snap", c + "_api"
            if api in df.columns:
                df[c] = df[api].combine_first(df[snap]) if snap in df.columns else df[api]
            elif snap in df.columns:
                df[c] = df[snap]
        if "channel_created_date_api" in df.columns:
            df["channel_created_date"] = df["channel_created_date_api"]
        elif "channel_created_date_snap" in df.columns:
            df["channel_created_date"] = df["channel_created_date_snap"]
        drop_cols = [c for c in df.columns if c.endswith("_snap") or c.endswith("_api")]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
    else:
        df = base.copy()
        print("Warning: channel_enriched.csv not found. years_active will use placeholder.")

    today = pd.Timestamp.today(tz=None)

    if "channel_created_date" in df.columns and df["channel_created_date"].notna().any():
        df["channel_created_date"] = pd.to_datetime(df["channel_created_date"], utc=True, errors="coerce")
        created = df["channel_created_date"].dt.tz_localize(None)
        df["years_active"] = ((today - created).dt.days / 365.0).clip(lower=0.01)
    else:
        df["years_active"] = 3.0
        print("Warning: channel_created_date missing — years_active = 3.0. Run fetch_youtube_channels.py.")

    has_videos = os.path.exists(VIDEOS_FLAT)
    if has_videos:
        v = pd.read_csv(VIDEOS_FLAT)
        v["publish_date"] = pd.to_datetime(v["publish_date"], utc=True, errors="coerce")
        cutoff = pd.Timestamp.today(tz="UTC") - pd.Timedelta(days=30)
        v_recent = v[v["publish_date"] >= cutoff]
        views_30 = v_recent.groupby("channel_id")["view_count"].sum().rename("views_last_30_days")
        v["eng_row"] = (v["like_count"].fillna(0) + v["comment_count"].fillna(0)) / v["view_count"].clip(
            lower=1
        )
        eng = v.groupby("channel_id")["eng_row"].mean().rename("engagement_ratio")
        v["month"] = v["publish_date"].dt.to_period("M")
        upm = v.groupby(["channel_id", "month"]).size().reset_index(name="n")
        consistency = upm.groupby("channel_id")["n"].std().fillna(0).rename("upload_consistency")
        agg = pd.concat([views_30, eng, consistency], axis=1)
        df = df.merge(agg, on="channel_id", how="left")
        df["views_last_30_days"] = df["views_last_30_days"].fillna(0)
        df["engagement_ratio"] = df["engagement_ratio"].fillna(df["engagement_ratio"].median())
        df["upload_consistency"] = df["upload_consistency"].fillna(0)
    else:
        df["views_last_30_days"] = df["total_views"] * (30.0 / 365.0)
        df["engagement_ratio"] = df["view_count"] / df["subscriber_count"].clip(lower=1)
        df["upload_consistency"] = 0.0
        print("Warning: videos_flat.csv not found — using approximations.")

    df["avg_views_per_video"] = df["total_views"] / df["video_count"].clip(lower=1)
    df["upload_frequency"] = df["video_count"] / (df["years_active"].clip(lower=0.01) * 12.0)

    if has_videos:
        df["growth_6m"] = (df["views_last_30_days"] * 6.0) / df["total_views"].clip(lower=1)
        q = df["growth_6m"].quantile(0.90)
        df["high_performing"] = (df["growth_6m"] > q).astype(int)
        df["target_mode"] = "growth_6m"
    else:
        q = df["engagement_ratio"].quantile(0.90)
        df["high_performing"] = (df["engagement_ratio"] > q).astype(int)
        df["growth_6m"] = np.nan
        df["target_mode"] = "engagement_ratio"

    keep = [
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
    if has_videos:
        keep.append("growth_6m")

    out_df = df[[c for c in keep if c in df.columns]].copy()
    try:
        validate_processed_v2(out_df)
    except ValidationError as e:
        print("Processed data validation failed — not writing CSV:\n", e)
        raise
    out_df.to_csv(OUT, index=False)
    print(f"Saved {len(out_df)} rows to {OUT}")
    print("Target balance:\n", out_df["high_performing"].value_counts(normalize=True))
    print("target_mode:", out_df["target_mode"].iloc[0])


if __name__ == "__main__":
    main()
