from flask import Flask, render_template, request, session
import json
import os
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'outputs', 'model_config.json')
MODEL_PATH_V2 = os.path.join(BASE_DIR, 'outputs', 'model_pipeline_v2.pkl')
MODEL_PATH_LEGACY = os.path.join(BASE_DIR, 'outputs', 'model_pipeline.pkl')
STATIC_OUT = os.path.join(BASE_DIR, 'static', 'outputs')
os.makedirs(STATIC_OUT, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-in-production')

# Default legacy features (v1); overridden by model_config.json when present
NUM_FEATURES_LEGACY = [
    'view_count', 'subscriber_count', 'video_count', 'years_active',
    'total_views', 'subscribers', 'growth_rate', 'engagement_proxy', 'activity_score',
]


def get_model_paths_and_features():
    """Prefer v2 pipeline + config if trained; else legacy."""
    if os.path.exists(CONFIG_PATH) and os.path.exists(MODEL_PATH_V2):
        with open(CONFIG_PATH, encoding='utf-8') as f:
            cfg = json.load(f)
        feats = cfg.get('features', NUM_FEATURES_LEGACY)
        return MODEL_PATH_V2, feats, cfg.get('target_mode', 'v2')
    return MODEL_PATH_LEGACY, NUM_FEATURES_LEGACY, 'legacy'


def load_model():
    path, _, _ = get_model_paths_and_features()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. Run train_model_v2.py or train_model.py first."
        )
    return joblib.load(path)

def generate_xai_feedback(feature_shap, probability):
    """Generate model-aware, SHAP-based feedback (positive = strength, negative = weakness)."""
    strengths = []
    weaknesses = []
    sorted_features = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)
    for feature, value in sorted_features[:5]:
        readable = feature.replace("_", " ").title()
        if value > 0:
            strengths.append(f"{readable} positively influenced performance (+{value:.3f}).")
        else:
            weaknesses.append(f"{readable} reduced performance impact ({value:.3f}).")
    if probability >= 0.75:
        summary = "Your channel aligns strongly with patterns of high-performing Sri Lankan channels."
    elif probability >= 0.5:
        summary = "Your channel shows moderate characteristics of high-performing channels."
    else:
        summary = "Your channel lacks several key characteristics of top-performing channels."
    return summary, strengths, weaknesses


@app.route('/', methods=['GET'])
def index():
    _, feats, _ = get_model_paths_and_features()
    # Empty form for "Try another prediction"; previous values for "Back"; else median defaults
    if request.args.get('reset') or request.args.get('new'):
        defaults = {f: '' for f in feats}
        defaults_source = 'empty'
    elif session.get('last_input'):
        defaults = {f: session['last_input'].get(f, '') for f in feats}
        defaults_source = 'previous'
    else:
        defaults = {f: 0 for f in feats}
        csv_v2 = os.path.join(BASE_DIR, 'Data', 'processed_yt_v2.csv')
        csv_v1 = os.path.join(BASE_DIR, 'Data', 'processed_yt.csv')
        csv_path = csv_v2 if os.path.exists(csv_v2) else csv_v1
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                for f in feats:
                    if f in df.columns:
                        med = df[f].median()
                        defaults[f] = float(med) if not pd.isna(med) else 0.0
            except Exception:
                pass
        defaults_source = 'median'
    return render_template('index.html', features=feats, defaults=defaults, defaults_source=defaults_source)


@app.route('/predict', methods=['POST'])
def predict():
    model = load_model()
    _, feats, _ = get_model_paths_and_features()
    # collect inputs
    vals = []
    vals_raw = {}
    for f in feats:
        v = request.form.get(f, '')
        vals_raw[f] = v
        # sanitize common thousand separators and whitespace
        if isinstance(v, str):
            v = v.replace(',', '').strip()
        try:
            v = float(v)
        except Exception:
            v = 0.0
        vals.append(v)

    # store submitted values so "Back" can restore them
    session['last_input'] = vals_raw

    print('Received raw inputs:', vals_raw)

    X = pd.DataFrame([vals], columns=feats)
    # Ensure numeric dtypes
    X = X.astype(float)
    pred = model.predict(X)[0]

    # Robustly extract probability for the positive class label (1)
    proba_arr = model.predict_proba(X)
    clf = model.named_steps['classifier']
    classes = np.array(getattr(clf, 'classes_', []))
    pos_idx = None
    if classes.size > 0:
        try:
            matches = np.where(classes == 1)[0]
            if matches.size > 0:
                pos_idx = int(matches[0])
        except Exception:
            pos_idx = None
        if pos_idx is None:
            # try converting class labels to int strings
            for i, c in enumerate(classes):
                try:
                    if int(c) == 1:
                        pos_idx = i
                        break
                except Exception:
                    continue
        if pos_idx is None:
            # fallback for binary classifiers: pick index of the larger label
            if proba_arr.ndim > 1 and proba_arr.shape[1] > 1:
                try:
                    pos_idx = int(np.argmax(classes))
                except Exception:
                    pos_idx = 1
            else:
                pos_idx = 0
    else:
        pos_idx = 1 if (proba_arr.ndim > 1 and proba_arr.shape[1] > 1) else 0

    # get probability value
    if proba_arr.ndim == 1:
        proba = float(proba_arr[0])
    else:
        proba = float(proba_arr[0, pos_idx])

    print(f'Classes: {classes}, using positive index: {pos_idx}')
    print(f'Prediction: {pred}, Probability: {proba}')
    
    
    # compute SHAP for this single sample (tree vs linear classifier)
    X_trans = model.named_steps['preprocessor'].transform(X)
    clf = model.named_steps['classifier']
    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, X_trans)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_pos = shap_values[1] if len(shap_values) > 1 else np.array(shap_values[0])
        else:
            shap_pos = shap_values
    else:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_pos = shap_values[1]
        elif hasattr(shap_values, 'values'):
            shap_pos = shap_values.values if len(shap_values.values.shape) == 2 else shap_values.values[:, :, 1]
        else:
            shap_pos = shap_values
    if len(shap_pos.shape) == 1:
        shap_pos = shap_pos.reshape(-1, 1)

    # SHAP for this sample: feature -> contribution to positive class
    shap_sample_row = shap_pos[0]
    feature_shap = dict(zip(feats, shap_sample_row))
    summary, xai_strengths, xai_weaknesses = generate_xai_feedback(feature_shap, proba)

    # build simple bar plot of absolute shap for this sample
    sample_shap = np.abs(shap_sample_row)
    s = pd.Series(sample_shap, index=feats).sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    s.plot(kind='barh', color='C0')
    plt.title('Absolute SHAP values (this sample)')
    plt.xlabel('Absolute SHAP')
    plt.tight_layout()
    img_path = os.path.join('static', 'outputs', 'shap_sample.png')
    plt.savefig(os.path.join(BASE_DIR, img_path), dpi=200)
    plt.close()

    return render_template('result.html', prediction=int(pred), probability=float(proba), img_path='/' + img_path, summary=summary, xai_strengths=xai_strengths, xai_weaknesses=xai_weaknesses)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
