import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
data_dir = r'D:\L4S1\ML\Assignment\impl\Data'
outputs_dir = r'D:\L4S1\ML\Assignment\impl\outputs'
os.makedirs(outputs_dir, exist_ok=True)

df = pd.read_csv(os.path.join(data_dir, 'processed_yt.csv'))
print('Data:', df.shape, '\nTarget:', df['high_performing'].value_counts(normalize=True))

# YOUR EXACT FEATURES (no category → num only)
num_features = ['view_count', 'subscriber_count', 'video_count', 'years_active', 
                'total_views', 'subscribers', 'growth_rate', 'engagement_proxy', 'activity_score']
cat_features = []  # No categories

X = df[num_features]
y = df['high_performing']

# Splits
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

# Preprocessor (num only)
preprocessor = ColumnTransformer([('num', StandardScaler(), num_features)])

# Model
model = Pipeline([('preprocessor', preprocessor), ('classifier', XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8,
    scale_pos_weight=sum(y_train==0)/sum(y_train==1), random_state=42))])

model.fit(X_train, y_train)

# Metrics
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:,1]
print('Test F1:', round(f1_score(y_test, y_test_pred), 3))
print('Test AUC:', round(roc_auc_score(y_test, y_test_proba), 3))

# Plots
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6,4)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues');
plt.title('Confusion Matrix'); plt.savefig(os.path.join(outputs_dir, 'cm.png'), dpi=300, bbox_inches='tight'); plt.close()

feature_names = num_features
importances = model.named_steps['classifier'].feature_importances_
imp_df = pd.DataFrame({'feature':feature_names, 'imp':importances}).sort_values('imp', ascending=False)
plt.figure(figsize=(10,6)); sns.barplot(data=imp_df, x='imp', y='feature');
plt.title('Feature Importance'); plt.savefig(os.path.join(outputs_dir, 'features.png'), dpi=300, bbox_inches='tight'); plt.close()

# ===== FIXED SHAP BLOCK =====
import shap

print('Computing SHAP...')

X_test_trans = model.named_steps['preprocessor'].transform(X_test)
feature_names = num_features

explainer = shap.TreeExplainer(model.named_steps['classifier'])
shap_values = explainer.shap_values(X_test_trans[:100])

# FIXED: Use shap_values[1] (.values if Explanation obj) for positive class
shap_pos = shap_values[1] if isinstance(shap_values[1], np.ndarray) else shap_values[1].values

# Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_pos, X_test_trans[:100], 
                  feature_names=feature_names, 
                  max_display=9, show=False)
plt.title('SHAP Summary Plot')
plt.savefig(os.path.join(outputs_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

# Bar plot
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_pos, X_test_trans[:100], 
                  feature_names=feature_names, 
                  plot_type='bar', show=False)
plt.savefig(os.path.join(outputs_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
plt.close()

print('SHAP plots SAVED!')
print('Done! Check outputs/')
