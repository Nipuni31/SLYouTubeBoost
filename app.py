from flask import Flask, render_template, request
import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'outputs', 'model_pipeline.pkl')
STATIC_OUT = os.path.join(BASE_DIR, 'static', 'outputs')
os.makedirs(STATIC_OUT, exist_ok=True)

app = Flask(__name__)

# features used by the model
NUM_FEATURES = ['view_count', 'subscriber_count', 'video_count', 'years_active',
                'total_views', 'subscribers', 'growth_rate', 'engagement_proxy', 'activity_score']


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run the training script first.")
    return joblib.load(MODEL_PATH)


@app.route('/', methods=['GET'])
def index():
    # show a simple form to input feature values
    defaults = {f: 0 for f in NUM_FEATURES}
    # try to load medians from training data to provide sensible defaults
    csv_path = os.path.join(BASE_DIR, 'Data', 'processed_yt.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for f in NUM_FEATURES:
                if f in df.columns:
                    med = df[f].median()
                    # keep numeric medians as simple Python types
                    defaults[f] = float(med) if not pd.isna(med) else 0.0
        except Exception:
            pass

    return render_template('index.html', features=NUM_FEATURES, defaults=defaults)


@app.route('/predict', methods=['POST'])
def predict():
    model = load_model()
    # collect inputs
    vals = []
    vals_raw = {}
    for f in NUM_FEATURES:
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

    # debug log of received raw inputs
    print('Received raw inputs:', vals_raw)

    X = pd.DataFrame([vals], columns=NUM_FEATURES)
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

    # compute SHAP for this single sample
    X_trans = model.named_steps['preprocessor'].transform(X)
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    shap_values = explainer.shap_values(X_trans)
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]
    elif hasattr(shap_values, 'values'):
        shap_pos = shap_values.values if len(shap_values.values.shape) == 2 else shap_values.values[:, :, 1]
    else:
        shap_pos = shap_values
    if len(shap_pos.shape) == 1:
        shap_pos = shap_pos.reshape(-1, 1)

    # build simple bar plot of absolute shap for this sample
    sample_shap = np.abs(shap_pos[0])
    s = pd.Series(sample_shap, index=NUM_FEATURES).sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    s.plot(kind='barh', color='C0')
    plt.title('Absolute SHAP values (this sample)')
    plt.xlabel('Absolute SHAP')
    plt.tight_layout()
    img_path = os.path.join('static', 'outputs', 'shap_sample.png')
    plt.savefig(os.path.join(BASE_DIR, img_path), dpi=200)
    plt.close()

    return render_template('result.html', prediction=int(pred), probability=float(proba), img_path='/' + img_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
