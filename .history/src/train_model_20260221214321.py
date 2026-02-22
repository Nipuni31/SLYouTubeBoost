import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ===== LOAD PROCESSED DATA =====
data_dir = r'D:\L4S1\ML\Assignment\impl\Data'
df = pd.read_csv(os.path.join(data_dir, 'processed_yt.csv'))
print('Data loaded:', df.shape)
print('Columns:', df.columns.tolist())
print('Target balance:\n', df['high_performing'].value_counts(normalize=True))

# ===== DEFINE FEATURES (UPDATE THESE AFTER RUNNING data_prep.py PRINT) =====
num_features = ['subscribers', 'total_views', 'video_count', 'engagement_rate', 
                'avg_video_duration', 'upload_frequency', 'years_active']  # Add your cols
cat_features = ['category']  # If no category, make empty: []

X = df[num_features + cat_features]
y = df['high_performing']

# ===== SPLITS: 70/15/15 STRATIFIED =====
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

# ===== PREPROCESSOR =====
numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# ===== XGBOOST MODEL =====
params = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'scale_pos_weight': sum(y_train==0)/sum(y_train==1),
    'random_state': 42,
    'eval_metric': 'logloss'
}
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(**params))
])

# ===== TRAIN =====
model.fit(X_train, y_train, 
          classifier__eval_set=[(preprocessor.transform(X_val), y_val)],
          classifier__early_stopping_rounds=20, verbose=False)

# ===== PREDICT & METRICS =====
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

# Scores
scores = {
    'Train F1': f1_score(y_train, y_train_pred),
    'Val F1': f1_score(y_val, y_val_pred),
    'Test F1': f1_score(y_test, y_test_pred),
    'Test AUC': roc_auc_score(y_test, y_test_proba),
    'Test Acc': (y_test == y_test_pred).mean()
}
print('Scores:', {k: round(v,3) for k,v in scores.items()})

# ===== PLOTS =====
outputs_dir = r'D:\L4S1\ML\Assignment\impl\outputs'
os.makedirs(outputs_dir, exist_ok=True)

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Test Confusion Matrix')
plt.ylabel('True'), plt.xlabel('Predicted')
plt.savefig(os.path.join(outputs_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Feature Importance
feature_names = (num_features + 
                 list(model.named_steps['preprocessor'].named_transformers_['cat']
                      .named_steps['onehot'].get_feature_names_out(cat_features)))
importances = model.named_steps['classifier'].feature_importances_
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}
                     ).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(data=imp_df, x='importance', y='feature')
plt.title('Top 10 Feature Importances')
plt.savefig(os.path.join(outputs_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save model & metrics
joblib.dump(model, os.path.join(outputs_dir, 'model.pkl'))
pd.DataFrame([scores]).to_csv(os.path.join(outputs_dir, 'metrics.csv'), index=False)
print('Saved model, plots, metrics to outputs/')

print('TRAINING COMPLETE!')
