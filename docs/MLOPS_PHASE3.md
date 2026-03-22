# Phase 3 — MLOps

## 1. Experiment tracking (MLflow)

- **Dependency:** `mlflow` (see `requirements.txt`).
- **Training:** `python train_model_phase2.py --mlflow`
- **Default store:** `./mlruns` (or set `MLFLOW_TRACKING_URI`, e.g. `http://127.0.0.1:5000` for a local server).
- **Logged:** experiment params (`best_model`, `target_mode`, `n_features`, Optuna flags), per-model metrics (`ModelName__test_roc_auc`, etc.), artifacts (`model_comparison_phase2.csv`, `model_config.json`, plots, `model_pipeline_v2.pkl`, `mlflow_metrics.json`), and **`mlflow.sklearn.log_model`** under the `model` artifact path.

**UI:** `mlflow ui` (from project root) to browse runs.

## 2. Pipeline automation

Single command:

```bash
python run_pipeline.py
```

Steps (see `run_pipeline.py`):

1. **Fetch** (optional): `scripts/fetch_youtube_channels.py` if `YOUTUBE_API_KEY` is set.
2. **Preprocess:** `data_prep_v2.py` (requires `Data/youtube_data.csv`).
3. **Train:** `train_model_phase2.py --mlflow` (omit MLflow with `--skip-mlflow`).
4. **Version:** copy artifacts to `models/v1`, `models/v2`, …

Flags: `--skip-fetch`, `--skip-preprocess`, `--skip-train`, `--skip-mlflow`, `--skip-version`, `--fetch-limit N`, `--optuna`.

## 3. Model versioning

Directory layout:

```
models/
  README.md
  v1/
    model.pkl
    model_config.json
    metrics.json
    model_comparison.csv
    version_meta.json
    ...
```

Each run of `run_pipeline.py` (without `--skip-version`) increments the version folder. The Flask app still reads **`outputs/model_pipeline_v2.pkl`** by default; deploy a specific `models/vN/` by copying into `outputs/` or setting paths in your deployment config.

## 4. Docker

```bash
docker build -t yt-predictor .
docker run -p 5000:5000 -v ./outputs:/app/outputs yt-predictor
```

Mount `outputs/` with a trained `model_pipeline_v2.pkl` (and `model_config.json`) or run the pipeline inside a job before serving.

## 5. CI/CD (GitHub Actions)

Workflow: `.github/workflows/ci.yml`

- **test-and-lint:** `pip install -r requirements.txt -r requirements-dev.txt`, `ruff check .`, `pytest tests/ -v`
- **docker:** build image (no push; add registry login + `push: true` when ready)

## 6. Local dev checks

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
pytest tests/ -v
```
