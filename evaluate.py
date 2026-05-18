# evaluate.py - FIXED VERSION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib                                          # FIX 3: import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# --- Reload everything this file needs ---
# FIX 1 & 2: recreate y_test and y_pred_xgb by loading saved model + data
df = pd.read_csv('cleaned_data.csv')

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(    # FIX 1: y_test now defined
    X, y, test_size=0.2, random_state=42, stratify=y
)

xgb = joblib.load('churn_model.pkl')                    # FIX 4: xgb now defined
y_pred_xgb = xgb.predict(X_test)                       # FIX 2: y_pred_xgb now defined

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stay', 'Churn'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - XGBoost')
plt.savefig('confusion_matrix.png')
plt.show()

# --- Feature Importance ---
feature_names = joblib.load('feature_names.pkl')
importances = xgb.feature_importances_

feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nTop 10 important features:")
print(feat_df)