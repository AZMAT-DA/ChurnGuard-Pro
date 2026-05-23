# multi_train.py — Train models for all 3 industries

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (classification_report,
                                     roc_auc_score, accuracy_score,
                                     precision_score, recall_score, f1_score)
from xgboost          import XGBClassifier
from imblearn.over_sampling import SMOTE
from industry_config  import INDUSTRIES

# Create models folder if it does not exist
os.makedirs('models', exist_ok=True)

# Store results for all industries
all_results = {}


def load_industry_data(config: dict) -> pd.DataFrame:
    """Load dataset based on industry config."""
    if config['file_type'] == 'csv':
        df = pd.read_csv(config['file'])
    else:
        df = pd.read_excel(config['file'], sheet_name=config['sheet'])
    return df


def preprocess_industry(df: pd.DataFrame, config: dict):
    """
    Generic preprocessing for any industry dataset.
    Returns X (features) and y (target).
    """
    # Drop useless columns
    df = df.drop(columns=config['drop_cols'], errors='ignore')

    # Map target column to 0/1 if needed
    target = config['target']
    if config['target_map']:
        df[target] = df[target].map(config['target_map'])

    # Drop rows where target is missing
    df = df.dropna(subset=[target])

    # Separate target
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Drop rows with too many missing values
    X = X.dropna(thresh=len(X.columns) * 0.5)
    y = y[X.index]

    # Fill remaining missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna(X[col].mode()[0]
                                   if not X[col].mode().empty else 'Unknown')
        else:
            X[col] = X[col].fillna(X[col].median())

    # One-hot encode categorical columns
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Convert all to float
    bool_cols = X.select_dtypes(include='bool').columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(int)
    X = X.astype(float)

    return X, y


def train_industry_model(industry_key: str, config: dict):
    """Train and save a model for one industry."""

    print(f"\n{'='*55}")
    print(f"TRAINING: {config['icon']} {config['name']} model")
    print(f"{'='*55}")

    # Load and preprocess
    df  = load_industry_data(config)
    X, y = preprocess_industry(df, config)

    print(f"Customers  : {len(X):,}")
    print(f"Features   : {X.shape[1]}")
    print(f"Churned    : {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"Stayed     : {(y==0).sum():,} ({(1-y.mean())*100:.1f}%)")

    # Scale numerical features
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # SMOTE balancing
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"\nAfter SMOTE: {pd.Series(y_bal).value_counts().to_dict()}")

    # Train XGBoost
    model = XGBClassifier(
        n_estimators  = 300,
        max_depth     = 5,
        learning_rate = 0.05,
        subsample     = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        gamma         = 0.1,
        eval_metric   = 'logloss',
        random_state  = 42
    )
    model.fit(X_bal, y_bal)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\nResults for {config['name']}:")
    print(classification_report(y_test, y_pred,
                                target_names=['Stay', 'Churn']))
    print(f"AUC Score : {auc:.4f}")

    # Cross-validation — tests model 5 times on different splits
    print(f"\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(
        XGBClassifier(
            n_estimators=300, max_depth=5,
            learning_rate=0.05, eval_metric='logloss',
            random_state=42
        ),
        X_scaled, y,
        cv=5,
        scoring='roc_auc'
    )
    print(f"CV AUC scores : {[round(s, 4) for s in cv_scores]}")
    print(f"CV AUC mean   : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Save model, scaler, feature names
    joblib.dump(model,              config['model_file'])
    joblib.dump(scaler,             config['scaler_file'])
    joblib.dump(X.columns.tolist(), config['features_file'])

    print(f"\nSaved: {config['model_file']}")

    # Store results
    all_results[config['name']] = {
        'Accuracy' : round(accuracy_score(y_test, y_pred)  * 100, 1),
        'Precision': round(precision_score(y_test, y_pred) * 100, 1),
        'Recall'   : round(recall_score(y_test, y_pred)    * 100, 1),
        'F1-Score' : round(f1_score(y_test, y_pred)        * 100, 1),
        'AUC'      : round(auc * 100, 1),
        'CV_AUC'   : round(cv_scores.mean() * 100, 1),
        'Customers': len(X),
        'Features' : X.shape[1]
    }


# ---- Train all 3 industries ----
for key, config in INDUSTRIES.items():
    try:
        train_industry_model(key, config)
    except Exception as e:
        print(f"ERROR training {key}: {e}")

# ---- Final comparison table ----
print(f"\n{'='*55}")
print("FINAL RESULTS — ALL INDUSTRIES")
print(f"{'='*55}")

results_df = pd.DataFrame(all_results).T
print(results_df.to_string())

results_df.to_csv('models/industry_results.csv')
print("\nSaved: models/industry_results.csv")
print("\nAll industry models trained successfully!")