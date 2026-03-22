# Phase 2 — Model reliability

## What was added

1. **Stratified K-Fold (k=5)**  
   Used inside **GridSearchCV** / Optuna objectives for hyperparameter selection, and **cross_val_score** reports mean ± std for ROC-AUC, F1, and **PR-AUC** (average precision) on the training set.

2. **Hyperparameter tuning**  
   - **Default:** `GridSearchCV` with `cv=StratifiedKFold(5)`, `scoring='roc_auc'`.  
   - **Optional:** `python train_model_phase2.py --optuna --trials 15` uses **Optuna** for **XGBoost** and **LightGBM** only (LR and RF stay on GridSearch).

3. **Model comparison**  
   Trains and evaluates:
   - Logistic Regression (baseline)  
   - Random Forest  
   - XGBoost  
   - LightGBM (if `pip install lightgbm`)  

   **Metrics (held-out test + CV on train):** ROC-AUC, F1, PR-AUC.

4. **Calibration**  
   `sklearn.calibration.calibration_curve` — combined plot: `outputs/calibration_curves_phase2.png` (all models vs diagonal “perfect calibration”). Use this to judge whether UI probabilities are well-calibrated.

## Commands

```bash
# After data_prep_v2.py produced Data/processed_yt_v2.csv
python train_model_phase2.py

# With Optuna for gradient boosting models
python train_model_phase2.py --optuna --trials 20
```

## Outputs

| File | Description |
|------|-------------|
| `outputs/model_comparison_phase2.csv` | Per-model CV + test metrics |
| `outputs/calibration_curves_phase2.png` | Calibration curves |
| `outputs/shap_summary_phase2.png` | SHAP for the selected best model |
| `outputs/model_pipeline_v2.pkl` | **Best** model (by test ROC-AUC), refit on **full** data |
| `outputs/model_config.json` | Features + `algorithm`, `classifier_class`, `phase`, `metrics_test` |

## Flask / SHAP

The app uses **TreeExplainer** for tree-based models and **LinearExplainer** for **LogisticRegression** so probabilities in the UI stay explainable after Phase 2.

## Optional: probability calibration

If calibration curves show systematic over/under-confidence, wrap the final classifier with `sklearn.calibration.CalibratedClassifierCV` in a follow-up experiment (not applied by default so the pipeline stays simple).
