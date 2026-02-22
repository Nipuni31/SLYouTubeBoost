import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'outputs', 'model_pipeline.pkl')

print('Loading model from', MODEL_PATH)
model = joblib.load(MODEL_PATH)

NUM_FEATURES = ['view_count', 'subscriber_count', 'video_count', 'years_active',
                'total_views', 'subscribers', 'growth_rate', 'engagement_proxy', 'activity_score']

# High-performing sample from your input
vals = {
    'view_count': 200000,
    'subscriber_count': 150000,
    'video_count': 500,
    'years_active': 6,
    'total_views': 50000000,
    'subscribers': 150000,
    'growth_rate': 0.25,
    'engagement_proxy': 0.08,
    'activity_score': 0.95,
}

X = pd.DataFrame([vals], columns=NUM_FEATURES)
print('\nInput dataframe:')
print(X.to_dict(orient='records'))

pre = model.named_steps['preprocessor']
clf = model.named_steps['classifier']

X_trans = pre.transform(X)
print('\nTransformed input (first row):')
print(X_trans.shape)
print(X_trans[0])

print('\nClassifier classes_: ', getattr(clf, 'classes_', None))

pred = model.predict(X)[0]
proba = model.predict_proba(X)
print('\nModel predict:', pred)
print('Model predict_proba:', proba)

# If XGBoost, show raw margin if possible
try:
    if hasattr(clf, 'predict_proba'):
        pass
    if hasattr(clf, 'predict'):
        print('Decision function / raw prediction (if available):')
        if hasattr(clf, 'predict_proba'):
            print('proba shape', proba.shape)
except Exception as e:
    print('Extra debug error:', e)

print('\nDone')

# Inspect training distribution for context
csv_path = os.path.join(BASE_DIR, 'Data', 'processed_yt.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print('\nLoaded training data:', df.shape)
    feats = NUM_FEATURES
    print('\nFeature means and std by class (sample):')
    for f in feats:
        if f in df.columns:
            m0 = df[df['high_performing']==0][f].mean()
            s0 = df[df['high_performing']==0][f].std()
            m1 = df[df['high_performing']==1][f].mean()
            s1 = df[df['high_performing']==1][f].std()
            med0 = df[df['high_performing']==0][f].median()
            med1 = df[df['high_performing']==1][f].median()
            print(f"{f}: class0 mean={m0:.4g}, std={s0:.4g}, med={med0:.4g} | class1 mean={m1:.4g}, std={s1:.4g}, med={med1:.4g}")
else:
    print('\nTraining CSV not found at', csv_path)

# Build samples at class1 medians and means to test model response
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    med_sample = {f: float(df[df['high_performing']==1][f].median()) if f in df.columns else 0.0 for f in NUM_FEATURES}
    mean_sample = {f: float(df[df['high_performing']==1][f].mean()) if f in df.columns else 0.0 for f in NUM_FEATURES}

    for name, sample in [('class1_median', med_sample), ('class1_mean', mean_sample)]:
        Xs = pd.DataFrame([sample], columns=NUM_FEATURES)
        p = model.predict(Xs)[0]
        proba = model.predict_proba(Xs)
        print(f"\n{name} predict: {p}, proba: {proba}")
