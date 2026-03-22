#!/usr/bin/env python3
"""
Single entrypoint: optional API fetch → preprocess → train (Phase 2) → MLflow → versioned models/.

MLflow logging is ON by default for training (passes --mlflow to train_model_phase2.py).
Use --skip-mlflow only if you want to disable experiment tracking.

Usage:
  python run_pipeline.py
  python run_pipeline.py --optuna
  python run_pipeline.py --skip-fetch --skip-mlflow
  python run_pipeline.py --fetch-limit 500   # channels for YouTube API (needs YOUTUBE_API_KEY)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "outputs"
DATA_PROCESSED = ROOT / "Data" / "processed_yt_v2.csv"


def run_step(desc: str, cmd: list[str]) -> None:
    print(f"\n>>> {desc}\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end MLOps pipeline")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip YouTube API channel fetch")
    parser.add_argument("--fetch-limit", type=int, default=None, help="Max channels for fetch_youtube_channels.py")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip data_prep_v2.py")
    parser.add_argument("--skip-train", action="store_true", help="Skip train_model_phase2.py")
    parser.add_argument("--skip-mlflow", action="store_true", help="Do not pass --mlflow to training")
    parser.add_argument("--skip-version", action="store_true", help="Skip copying to models/vN/")
    parser.add_argument("--optuna", action="store_true", help="Pass --optuna to training")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_fetch and os.environ.get("YOUTUBE_API_KEY"):
        cmd = [py, str(ROOT / "scripts" / "fetch_youtube_channels.py")]
        if args.fetch_limit:
            cmd.extend(["--limit", str(args.fetch_limit)])
        try:
            run_step("Fetch channel metadata (YouTube API)", cmd)
        except subprocess.CalledProcessError as e:
            print("Fetch failed; continuing without new channel_enriched.csv", e)
    elif not args.skip_fetch:
        print("Skipping fetch: set YOUTUBE_API_KEY to enable, or use --skip-fetch explicitly.")

    if not args.skip_preprocess:
        if not (ROOT / "Data" / "youtube_data.csv").exists():
            print("ERROR: Data/youtube_data.csv not found. Add raw data before running the pipeline.")
            return 1
        run_step("Preprocess (data_prep_v2.py)", [py, str(ROOT / "data_prep_v2.py")])
    else:
        if not DATA_PROCESSED.exists():
            print("ERROR: processed data missing and --skip-preprocess set.")
            return 1

    if not args.skip_train:
        tcmd = [py, str(ROOT / "train_model_phase2.py")]
        if not args.skip_mlflow:
            tcmd.append("--mlflow")
        if args.optuna:
            tcmd.extend(["--optuna", "--trials", "15"])
        run_step("Train & compare models (train_model_phase2.py)", tcmd)
    else:
        if not (OUTPUT_DIR / "model_pipeline_v2.pkl").exists():
            print("ERROR: no trained model in outputs/ and --skip-train set.")
            return 1

    if not args.skip_version:
        from mlops.versioning import save_versioned_artifacts

        save_versioned_artifacts(ROOT, OUTPUT_DIR)

    print("\n>>> Pipeline finished OK.")
    print(f"    Deployed model: {OUTPUT_DIR / 'model_pipeline_v2.pkl'}")
    print("    Flask: python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
