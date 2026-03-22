# Versioned models

Each successful `run_pipeline.py` run copies artifacts into `models/v1`, `models/v2`, …

Contents per version:

| File | Description |
|------|-------------|
| `model.pkl` | Copy of `outputs/model_pipeline_v2.pkl` |
| `model_config.json` | Feature list + metadata for the app |
| `metrics.json` | Best model test metrics |
| `model_comparison.csv` | Full Phase 2 comparison table |
| `calibration_curves_phase2.png` | Calibration plot (if present) |
| `shap_summary_phase2.png` | SHAP summary (if present) |
| `version_meta.json` | Version label + source path |

The running app loads from `outputs/` by default. Point `MODEL_PATH` or sync `models/vN/` → `outputs/` if you deploy a specific version.
