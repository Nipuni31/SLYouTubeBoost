"""
Fetch recent videos per channel (playlistItems + videos.list) for engagement / recency features.

Quota-heavy: use --limit-channels for development. Default fetches up to 50 videos per channel.

Outputs: Data/videos_flat.csv with:
  video_id, channel_id, publish_date, view_count, like_count, comment_count

Requires: YOUTUBE_API_KEY
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
DEFAULT_CHANNELS = os.path.join(BASE_DIR, "Data", "channel_enriched.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "Data", "videos_flat.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels-csv", default=DEFAULT_CHANNELS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-channels", type=int, default=500, help="Max channels to fetch videos for")
    parser.add_argument("--max-videos-per-channel", type=int, default=50)
    parser.add_argument("--api-key", default=os.environ.get("YOUTUBE_API_KEY", ""))
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()

    if not args.api_key:
        print("Set YOUTUBE_API_KEY or pass --api-key")
        sys.exit(1)

    if not os.path.exists(args.channels_csv):
        print(f"Run fetch_youtube_channels.py first. Missing: {args.channels_csv}")
        sys.exit(1)

    ch = pd.read_csv(args.channels_csv)
    if "uploads_playlist_id" not in ch.columns:
        print("channel_enriched.csv must include uploads_playlist_id")
        sys.exit(1)

    ch = ch.dropna(subset=["uploads_playlist_id"])
    ch = ch.head(args.limit_channels)

    youtube = build("youtube", "v3", developerKey=args.api_key)
    rows = []

    for _, row in ch.iterrows():
        pid = row["uploads_playlist_id"]
        cid = row["channel_id"]
        if not pid:
            continue
        page_token = None
        fetched = 0
        while fetched < args.max_videos_per_channel:
            try:
                req = (
                    youtube.playlistItems()
                    .list(
                        part="contentDetails",
                        playlistId=pid,
                        maxResults=min(50, args.max_videos_per_channel - fetched),
                        pageToken=page_token,
                    )
                )
                pl_resp = req.execute()
            except HttpError as e:
                print("playlistItems HttpError:", e, "channel", cid)
                break

            vids = []
            for it in pl_resp.get("items", []):
                vid = it.get("contentDetails", {}).get("videoId")
                if vid:
                    vids.append(vid)
            if not vids:
                break

            # videos.list in batches of 50
            for i in range(0, len(vids), 50):
                batch = vids[i : i + 50]
                try:
                    vr = (
                        youtube.videos()
                        .list(part="snippet,statistics", id=",".join(batch))
                        .execute()
                    )
                except HttpError as e:
                    print("videos.list HttpError:", e)
                    break
                for v in vr.get("items", []):
                    vid = v["id"]
                    sn = v.get("snippet", {})
                    st = v.get("statistics", {})
                    pub = sn.get("publishedAt", "")
                    views = int(st.get("viewCount", 0) or 0)
                    likes = int(st.get("likeCount", 0) or 0) if "likeCount" in st else 0
                    comments = int(st.get("commentCount", 0) or 0) if "commentCount" in st else 0
                    rows.append(
                        {
                            "video_id": vid,
                            "channel_id": cid,
                            "publish_date": pub,
                            "view_count": views,
                            "like_count": likes,
                            "comment_count": comments,
                        }
                    )
                    fetched += 1
                    if fetched >= args.max_videos_per_channel:
                        break

            page_token = pl_resp.get("nextPageToken")
            if not page_token or fetched >= args.max_videos_per_channel:
                break
            time.sleep(args.sleep)

        time.sleep(args.sleep)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} video rows to {args.output}")


if __name__ == "__main__":
    main()
