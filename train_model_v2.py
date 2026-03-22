"""
Train XGBoost on processed_yt_v2.csv — excludes label-leaking columns per target_mode.

Outputs:
  - outputs/model_pipeline_v2.pkl
  - outputs/model_config.json  (feature list for Flask)
"""
from __future__ import annotations

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import shap

from validation.data_validation import ValidationError, validate_processed_v2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data", "processed_yt_v2.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    mode = str(df["target_mode"].iloc[0])
    # Never use target or direct leakage
    if mode == "growth_6m":
        # Target uses views_last_30_days + total_views — exclude views_last_30_days from X
        cols = [
            "subscriber_count",
            "video_count",
            "years_active",
            "avg_views_per_video",
            "upload_frequency",
            "engagement_ratio",
            "upload_consistency",
        ]
    else:
        # Target uses engagement_ratio — exclude it from X
        cols = [
            "subscriber_count",
            "video_count",
            "years_active",
            "avg_views_per_video",
            "views_last_30_days",
            "upload_frequency",
            "upload_consistency",
        ]
    return [c for c in cols if c in df.columns]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    try:
        validate_processed_v2(df)
    except ValidationError as e:
        print("Training data validation failed:\n", e)
        raise
    mode = str(df["target_mode"].iloc[0])
    feature_cols = get_feature_columns(df)
    y = df["high_performing"]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].median())

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    preprocessor = ColumnTransformer([("num", StandardScaler(), feature_cols)])
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    scale_pos_weight=sum(y_train == 0) / max(sum(y_train == 1), 1),
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    print("Test F1:", round(f1_score(y_test, y_test_pred), 3))
    print("Test AUC:", round(roc_auc_score(y_test, y_test_proba), 3))

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (v2)")
    plt.savefig(os.path.join(OUTPUT_DIR, "cm_v2.png"), dpi=300, bbox_inches="tight")
    plt.close()

    clf = model.named_steps["classifier"]
    imp_df = pd.DataFrame(
        {"feature": feature_cols, "imp": clf.feature_importances_}
    ).sort_values("imp", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, x="imp", y="feature")
    plt.title("Feature Importance (v2)")
    plt.savefig(os.path.join(OUTPUT_DIR, "features_v2.png"), dpi=300, bbox_inches="tight")
    plt.close()

    X_test_trans = model.named_steps["preprocessor"].transform(X_test)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_trans[: min(100, len(X_test_trans))])
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]
    else:
        shap_pos = shap_values
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_pos,
        X_test_trans[: min(100, len(X_test_trans))],
        feature_names=feature_cols,
        max_display=len(feature_cols),
        show=False,
    )
    plt.title("SHAP Summary (v2)")
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_v2.png"), dpi=300, bbox_inches="tight")
    plt.close()

    model_path = os.path.join(OUTPUT_DIR, "model_pipeline_v2.pkl")
    joblib.dump(model, model_path)
    cfg = {
        "features": feature_cols,
        "target_mode": mode,
        "data_file": "processed_yt_v2.csv",
        "model_file": "model_pipeline_v2.pkl",
    }
    with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("Saved:", model_path)
    print("Config:", cfg)


if __name__ == "__main__":
    main()
