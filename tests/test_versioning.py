"""Tests for mlops.versioning."""
import json
from pathlib import Path

import joblib

from mlops.versioning import next_version_dir, save_versioned_artifacts


def test_next_version_increments(tmp_path: Path):
    (tmp_path / "models" / "v1").mkdir(parents=True)
    v = next_version_dir(tmp_path)
    assert v.name == "v2"


def test_save_versioned_artifacts(tmp_path: Path):
    out = tmp_path / "outputs"
    out.mkdir()
    # minimal sklearn-like object
    joblib.dump({"dummy": 1}, out / "model_pipeline_v2.pkl")
    cfg = {"features": ["a"], "algorithm": "Test"}
    with open(out / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    with open(out / "model_comparison_phase2.csv", "w", encoding="utf-8") as f:
        f.write("model,test_roc_auc\nTest,0.9\n")

    vdir = save_versioned_artifacts(tmp_path, out)
    assert (vdir / "model.pkl").exists()
    assert (vdir / "model_config.json").exists()
    assert (vdir / "metrics.json").exists()
