"""
PHASE 2 — Model reliability: Stratified CV, hyperparameter tuning, multi-model comparison,
PR-AUC, calibration curves. Saves best pipeline to outputs/ for Flask.

Usage:
  python train_model_phase2.py
  python train_model_phase2.py --optuna --trials 15   # optional Optuna for XGB/LGBM only

Requires: processed_yt_v2.csv (run data_prep_v2.py first).
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

import shap

from train_model_v2 import DATA_PATH, OUTPUT_DIR, get_feature_columns
from validation.data_validation import ValidationError, validate_processed_v2

warnings.filterwarnings("ignore", category=UserWarning)


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler(), feature_cols)])


def scale_pos_weight(y: pd.Series) -> float:
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    return float(n0) / max(int(n1), 1)


def tune_with_gridsearch(
    name: str,
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> tuple[GridSearchCV, dict]:
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs, {
        "best_params": gs.best_params_,
        "best_cv_auc": float(gs.best_score_),
    }


def optuna_tune_xgb(
    feature_cols: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    spw: float,
    n_trials: int,
) -> tuple[Pipeline, dict]:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42,
            "scale_pos_weight": spw,
        }
        clf = XGBClassifier(**params)
        pipe = Pipeline(
            [("preprocessor", build_preprocessor(feature_cols)), ("classifier", clf)]
        )
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["random_state"] = 42
    best["scale_pos_weight"] = spw
    clf = XGBClassifier(**best)
    pipe = Pipeline(
        [("preprocessor", build_preprocessor(feature_cols)), ("classifier", clf)]
    )
    pipe.fit(X_train, y_train)
    return pipe, {"best_params": best, "best_cv_auc": study.best_value}


def optuna_tune_lgbm(
    feature_cols: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    spw: float,
    n_trials: int,
) -> tuple[Pipeline, dict]:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
        }
        clf = LGBMClassifier(**params, scale_pos_weight=spw)
        pipe = Pipeline(
            [("preprocessor", build_preprocessor(feature_cols)), ("classifier", clf)]
        )
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["random_state"] = 42
    best["verbose"] = -1
    clf = LGBMClassifier(**best, scale_pos_weight=spw)
    pipe = Pipeline(
        [("preprocessor", build_preprocessor(feature_cols)), ("classifier", clf)]
    )
    pipe.fit(X_train, y_train)
    return pipe, {"best_params": best, "best_cv_auc": study.best_value}


def evaluate_on_test(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }


def cv_metrics(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> dict:
    auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    pr = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
    )
    return {
        "roc_auc_mean": float(auc.mean()),
        "roc_auc_std": float(auc.std()),
        "f1_mean": float(f1.mean()),
        "f1_std": float(f1.std()),
        "pr_auc_mean": float(pr.mean()),
        "pr_auc_std": float(pr.std()),
    }


def plot_calibration_curves(
    models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_path: str,
) -> None:
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curves (test set)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def shap_for_best(model: Pipeline, feature_cols: list[str], X_test: pd.DataFrame, out_path: str) -> None:
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    X_bg = pre.transform(X_test)
    sample = X_bg[: min(200, len(X_test))]
    if isinstance(clf, LogisticRegression):
        bg = X_bg[: min(2000, len(X_bg))]
        explainer = shap.LinearExplainer(clf, bg)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else np.array(sv[0])
    else:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]
        elif hasattr(sv, "values"):
            sv = sv.values
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, sample, feature_names=feature_cols, show=False, max_display=len(feature_cols))
    plt.title("SHAP summary (Phase 2 best model)")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna", action="store_true", help="Use Optuna for XGBoost & LightGBM (GridSearch for LR/RF)")
    parser.add_argument("--trials", type=int, default=15, help="Optuna trials per model")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    validate_processed_v2(df)
    mode = str(df["target_mode"].iloc[0])
    feature_cols = get_feature_columns(df)
    y = df["high_performing"]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    spw = scale_pos_weight(y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prep = build_preprocessor(feature_cols)

    results: list[dict] = []
    fitted: dict[str, Pipeline] = {}

    # --- Logistic Regression ---
    pipe_lr = Pipeline(
        [
            ("preprocessor", prep),
            (
                "classifier",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    gs_lr, meta_lr = tune_with_gridsearch(
        "logistic_regression",
        pipe_lr,
        {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        X_train,
        y_train,
        cv,
    )
    fitted["LogisticRegression"] = gs_lr.best_estimator_
    cv_m = cv_metrics(gs_lr.best_estimator_, X_train, y_train, cv)
    te = evaluate_on_test(gs_lr.best_estimator_, X_test, y_test)
    results.append(
        {
            "model": "LogisticRegression",
            "best_cv_auc": meta_lr["best_cv_auc"],
            **{f"cv_{k}": v for k, v in cv_m.items()},
            **{f"test_{k}": v for k, v in te.items()},
        }
    )

    # --- Random Forest ---
    pipe_rf = Pipeline(
        [
            ("preprocessor", build_preprocessor(feature_cols)),
            (
                "classifier",
                RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            ),
        ]
    )
    gs_rf, meta_rf = tune_with_gridsearch(
        "random_forest",
        pipe_rf,
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [6, 10, None],
        },
        X_train,
        y_train,
        cv,
    )
    fitted["RandomForest"] = gs_rf.best_estimator_
    cv_m = cv_metrics(gs_rf.best_estimator_, X_train, y_train, cv)
    te = evaluate_on_test(gs_rf.best_estimator_, X_test, y_test)
    results.append(
        {
            "model": "RandomForest",
            "best_cv_auc": meta_rf["best_cv_auc"],
            **{f"cv_{k}": v for k, v in cv_m.items()},
            **{f"test_{k}": v for k, v in te.items()},
        }
    )

    # --- XGBoost ---
    if args.optuna and HAS_OPTUNA:
        pipe_xgb, meta_xgb = optuna_tune_xgb(
            feature_cols, X_train, y_train, cv, spw, args.trials
        )
        fitted["XGBoost"] = pipe_xgb
        cv_m = cv_metrics(pipe_xgb, X_train, y_train, cv)
        te = evaluate_on_test(pipe_xgb, X_test, y_test)
        results.append(
            {
                "model": "XGBoost",
                "best_cv_auc": meta_xgb["best_cv_auc"],
                **{f"cv_{k}": v for k, v in cv_m.items()},
                **{f"test_{k}": v for k, v in te.items()},
            }
        )
    else:
        pipe_xgb = Pipeline(
            [
                ("preprocessor", build_preprocessor(feature_cols)),
                (
                    "classifier",
                    XGBClassifier(
                        random_state=42,
                        scale_pos_weight=spw,
                    ),
                ),
            ]
        )
        gs_xgb, meta_xgb = tune_with_gridsearch(
            "xgboost",
            pipe_xgb,
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [4, 6, 8],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__subsample": [0.8],
            },
            X_train,
            y_train,
            cv,
        )
        fitted["XGBoost"] = gs_xgb.best_estimator_
        cv_m = cv_metrics(gs_xgb.best_estimator_, X_train, y_train, cv)
        te = evaluate_on_test(gs_xgb.best_estimator_, X_test, y_test)
        results.append(
            {
                "model": "XGBoost",
                "best_cv_auc": meta_xgb["best_cv_auc"],
                **{f"cv_{k}": v for k, v in cv_m.items()},
                **{f"test_{k}": v for k, v in te.items()},
            }
        )

    # --- LightGBM ---
    if HAS_LGBM:
        if args.optuna and HAS_OPTUNA:
            pipe_lgb, meta_lgb = optuna_tune_lgbm(
                feature_cols, X_train, y_train, cv, spw, args.trials
            )
            fitted["LightGBM"] = pipe_lgb
            cv_m = cv_metrics(pipe_lgb, X_train, y_train, cv)
            te = evaluate_on_test(pipe_lgb, X_test, y_test)
            results.append(
                {
                    "model": "LightGBM",
                    "best_cv_auc": meta_lgb["best_cv_auc"],
                    **{f"cv_{k}": v for k, v in cv_m.items()},
                    **{f"test_{k}": v for k, v in te.items()},
                }
            )
        else:
            pipe_lgb = Pipeline(
                [
                    ("preprocessor", build_preprocessor(feature_cols)),
                    (
                        "classifier",
                        LGBMClassifier(random_state=42, verbose=-1, scale_pos_weight=spw),
                    ),
                ]
            )
            gs_lgb, meta_lgb = tune_with_gridsearch(
                "lightgbm",
                pipe_lgb,
                {
                    "classifier__n_estimators": [100, 200],
                    "classifier__max_depth": [4, 8],
                    "classifier__learning_rate": [0.05, 0.1],
                },
                X_train,
                y_train,
                cv,
            )
            fitted["LightGBM"] = gs_lgb.best_estimator_
            cv_m = cv_metrics(gs_lgb.best_estimator_, X_train, y_train, cv)
            te = evaluate_on_test(gs_lgb.best_estimator_, X_test, y_test)
            results.append(
                {
                    "model": "LightGBM",
                    "best_cv_auc": meta_lgb["best_cv_auc"],
                    **{f"cv_{k}": v for k, v in cv_m.items()},
                    **{f"test_{k}": v for k, v in te.items()},
                }
            )
    else:
        print("LightGBM not installed; skip. pip install lightgbm")

    comp = pd.DataFrame(results)
    comp_path = os.path.join(OUTPUT_DIR, "model_comparison_phase2.csv")
    comp.to_csv(comp_path, index=False)
    print("\n=== Model comparison (test set + CV on train) ===\n")
    print(comp.to_string(index=False))
    print(f"\nSaved: {comp_path}")

    plot_calibration_curves(fitted, X_test, y_test, os.path.join(OUTPUT_DIR, "calibration_curves_phase2.png"))

    # Best by test ROC-AUC
    best_name = comp.loc[comp["test_roc_auc"].idxmax(), "model"]
    best_model = fitted[best_name]
    print(f"\nBest model (by test ROC-AUC): {best_name}")

    # Refit best on full non-test data for deployment
    best_model.fit(X, y)

    model_path = os.path.join(OUTPUT_DIR, "model_pipeline_v2.pkl")
    joblib.dump(best_model, model_path)

    clf_type = type(best_model.named_steps["classifier"]).__name__
    row = comp[comp["model"] == best_name].iloc[0].to_dict()
    metrics_test = {k: float(v) if hasattr(v, "item") else v for k, v in row.items()}
    cfg = {
        "features": feature_cols,
        "target_mode": mode,
        "data_file": "processed_yt_v2.csv",
        "model_file": "model_pipeline_v2.pkl",
        "phase": 2,
        "algorithm": best_name,
        "classifier_class": clf_type,
        "metrics_test": metrics_test,
    }
    with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # SHAP for best (tree or linear)
    try:
        shap_for_best(
            best_model,
            feature_cols,
            X_test,
            os.path.join(OUTPUT_DIR, "shap_summary_phase2.png"),
        )
    except Exception as e:
        print("SHAP plot skipped:", e)

    print(f"Saved model: {model_path}")
    print(f"Calibration plot: outputs/calibration_curves_phase2.png")


if __name__ == "__main__":
    main()
