"""
Versioned model registry under models/v1, v2, ...

Each version contains: model.pkl, model_config.json, metrics.json, and optional artifacts.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def next_version_dir(project_root: str | Path) -> Path:
    root = Path(project_root)
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    versions = []
    for p in models_dir.iterdir():
        if p.is_dir() and len(p.name) > 1 and p.name[0] == "v" and p.name[1:].isdigit():
            versions.append(int(p.name[1:]))
    n = max(versions) + 1 if versions else 1
    vdir = models_dir / f"v{n}"
    vdir.mkdir(parents=True, exist_ok=True)
    return vdir


def save_versioned_artifacts(
    project_root: str | Path,
    outputs_dir: str | Path,
    *,
    version_dir: Path | None = None,
) -> Path:
    """
    Copy training outputs into models/vN/ with stable filenames.

    Returns path to the version directory (e.g. models/v3).
    """
    root = Path(project_root)
    out = Path(outputs_dir)
    vdir = version_dir or next_version_dir(root)

    shutil.copy2(out / "model_pipeline_v2.pkl", vdir / "model.pkl")
    shutil.copy2(out / "model_config.json", vdir / "model_config.json")

    comp = out / "model_comparison_phase2.csv"
    if comp.exists():
        shutil.copy2(comp, vdir / "model_comparison.csv")

    metrics_src = out / "mlflow_metrics.json"
    if metrics_src.exists():
        shutil.copy2(metrics_src, vdir / "metrics.json")
    elif comp.exists():
        import pandas as pd

        df = pd.read_csv(comp)
        best = df.loc[df["test_roc_auc"].idxmax()]
        m = {k: float(v) if hasattr(v, "item") else v for k, v in best.to_dict().items()}
        with open(vdir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)

    for name in ("calibration_curves_phase2.png", "shap_summary_phase2.png"):
        p = out / name
        if p.exists():
            shutil.copy2(p, vdir / name)

    meta = {"version": vdir.name, "source_outputs": str(out)}
    with open(vdir / "version_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Versioned artifacts saved to {vdir}")
    return vdir
