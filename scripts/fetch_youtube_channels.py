"""
Fetch channel-level metadata from YouTube Data API v3 (channels.list).

Requires: pip install google-api-python-client
Set YOUTUBE_API_KEY in environment or pass --api-key.

Outputs: Data/channel_enriched.csv with:
  channel_id, channel_created_date, subscriber_count, video_count, total_views, uploads_playlist_id

Usage:
  python scripts/fetch_youtube_channels.py --limit 100   # test with 100 channels
  python scripts/fetch_youtube_channels.py               # all unique channels from Data/youtube_data.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print("Install: pip install google-api-python-client")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(BASE_DIR, "Data", "youtube_data.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "Data", "channel_enriched.csv")


def chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube channel metadata (batch).")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV with channel_id column")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Max unique channels (for testing)")
    parser.add_argument("--api-key", default=os.environ.get("YOUTUBE_API_KEY", ""))
    parser.add_argument("--sleep", type=float, default=0.05, help="Seconds between API calls")
    args = parser.parse_args()

    if not args.api_key:
        print("Set YOUTUBE_API_KEY or pass --api-key")
        sys.exit(1)

    df_in = pd.read_csv(args.input)
    if "channel_id" not in df_in.columns:
        print("Input CSV must contain column: channel_id")
        sys.exit(1)

    ids = df_in["channel_id"].dropna().astype(str).unique().tolist()
    if args.limit:
        ids = ids[: args.limit]
    print(f"Fetching {len(ids)} channels...")

    youtube = build("youtube", "v3", developerKey=args.api_key)
    rows = []

    for batch in chunked(ids, 50):
        try:
            resp = (
                youtube.channels()
                .list(part="snippet,statistics,contentDetails", id=",".join(batch))
                .execute()
            )
        except HttpError as e:
            print("HttpError:", e)
            sys.exit(1)

        for item in resp.get("items", []):
            cid = item["id"]
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            cd = item.get("contentDetails", {})
            uploads = cd.get("relatedPlaylists", {}).get("uploads", "")

            pub = snippet.get("publishedAt", "")
            vc = int(stats.get("viewCount", 0) or 0)
            sc = int(stats.get("subscriberCount", 0) or 0) if "subscriberCount" in stats else 0
            vidc = int(stats.get("videoCount", 0) or 0)

            rows.append(
                {
                    "channel_id": cid,
                    "channel_created_date": pub,
                    "subscriber_count": sc,
                    "video_count": vidc,
                    "total_views": vc,
                    "uploads_playlist_id": uploads,
                }
            )

        time.sleep(args.sleep)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
