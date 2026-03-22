# Data pipeline v2 (time-aware, reduced leakage)

## Overview

| Stage | What |
|--------|------|
| **1. Raw data** | `Data/youtube_data.csv` (snapshot: `time`, `channel_id`, `view_count`, `subscriber_count`, `video_count`) |
| **2. Channel API (optional)** | `scripts/fetch_youtube_channels.py` → `Data/channel_enriched.csv` (`channel_created_date`, stats, `uploads_playlist_id`) |
| **3. Video API (optional)** | `scripts/fetch_youtube_videos.py` → `Data/videos_flat.csv` (per-video views, likes, comments, publish dates) |
| **4. Preprocess** | `python data_prep_v2.py` → `Data/processed_yt_v2.csv` |
| **5. Train** | `python train_model_v2.py` → `outputs/model_pipeline_v2.pkl` + `outputs/model_config.json` |
| **6. App** | Flask loads v2 model automatically when `model_config.json` and `model_pipeline_v2.pkl` exist |

## API setup

1. [Google Cloud Console](https://console.cloud.google.com/) → enable **YouTube Data API v3**.
2. Create an API key; restrict it to YouTube Data API.
3. Set environment variable: `YOUTUBE_API_KEY=your_key` (PowerShell: `$env:YOUTUBE_API_KEY="..."`).

## Commands

```bash
# 1) Fetch channel metadata (all unique channels from youtube_data.csv; use --limit 200 for tests)
python scripts/fetch_youtube_channels.py --limit 200

# 2) Fetch recent videos per channel (quota-heavy; default 500 channels)
python scripts/fetch_youtube_videos.py --limit-channels 100

# 3) Build processed dataset
python data_prep_v2.py

# 4) Train and write model + feature list for the web app
python train_model_v2.py
```

## Target and leakage rules

- **If `videos_flat.csv` exists:**  
  - `growth_6m = (views_last_30_days * 6) / total_views`  
  - `high_performing` = top 10% by `growth_6m`  
  - **Model inputs exclude** `views_last_30_days` (and `growth_6m` is never a feature).

- **If only snapshot data (no videos):**  
  - `high_performing` = top 10% by channel-level `engagement_ratio` (approximation).  
  - **Model inputs exclude** `engagement_ratio`.

Dropped from the model (vs old pipeline): duplicate `total_views`/`subscribers` as separate features, and **no** `growth_rate` in **X** when it was used to define the old label.

## Quota note

Full runs for ~100k channels cost thousands of API units. Use `--limit` while developing; schedule full fetches over multiple days if needed.

## Data validation (Phase 1)

Built-in checks (no Great Expectations required; you can wrap these in GE later):

- **`validation/data_validation.py`**
  - `validate_raw_youtube_snapshot(df)` — after dedupe: required columns, non-negative counts, `channel_id` unique, no `inf`.
  - `validate_processed_v2(df)` — full processed table: non-negative metrics, `years_active` finite and &gt; 0, `high_performing` ∈ {0,1}, no NaN/inf in numeric feature columns, unique `channel_id`.
  - `validate_data(df)` — asserts your three core rules, then runs `validate_processed_v2`.

Integrated into **`data_prep_v2.py`** (after load + after build) and **`train_model_v2.py`** (on load).

**CLI:**

```bash
python -m validation.data_validation Data/processed_yt_v2.csv
python scripts/validate_csv.py Data/processed_yt_v2.csv
python scripts/validate_csv.py Data/youtube_data.csv --raw
```
