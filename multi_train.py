# multi_train.py — Train models for all 3 industries

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (classification_report,
                                        roc_auc_score, accuracy_score,
                                        precision_score, recall_score,
                                        f1_score)
from xgboost                    import XGBClassifier
from imblearn.over_sampling     import SMOTE
from industry_config            import INDUSTRIES

# Create models folder
os.makedirs('models', exist_ok=True)

all_results = {}


def load_industry_data(config: dict) -> pd.DataFrame:
    """Load dataset based on industry config."""
    if config['file_type'] == 'csv':
        df = pd.read_csv(config['file'])
    else:
        df = pd.read_excel(config['file'],
                           sheet_name=config['sheet'])
    return df


def preprocess_industry(df: pd.DataFrame, config: dict):
    """
    Generic preprocessing for any industry dataset.
    Fully fixed for Python 3.14 pandas string dtype issue.
    """

    # Drop useless ID columns
    df = df.drop(columns=config['drop_cols'], errors='ignore')

    # Map target column to 0/1 if needed
    target = config['target']
    if config['target_map']:
        df[target] = df[target].map(config['target_map'])

    # Drop rows where target is missing
    df = df.dropna(subset=[target])

    # Separate target from features
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Reset index for clean alignment
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print(f"  Raw shape: {X.shape}")
    print(f"  Columns: {X.columns.tolist()}")

    # ---- Step 1: Try to convert all columns to numeric first ----
    # Some columns look like numbers but are stored as strings
    for col in X.columns:
        try:
            converted = pd.to_numeric(X[col], errors='coerce')
            # Only convert if more than 80% of values are numeric
            if converted.notna().sum() > len(X) * 0.8:
                X[col] = converted
        except Exception:
            pass

    # ---- Step 2: Fill missing values safely ----
    for col in X.columns:
        try:
            # Use pandas API to check type — works on all pandas versions
            if pd.api.types.is_numeric_dtype(X[col]):
                # Numeric column — use median
                med = X[col].dropna().median()
                X[col] = X[col].fillna(med if not np.isnan(med) else 0)
            elif pd.api.types.is_bool_dtype(X[col]):
                X[col] = X[col].fillna(False)
            else:
                # Text or categorical column — use most common value
                mode_series = X[col].dropna().mode()
                fill_val    = mode_series.iloc[0] if len(mode_series) > 0 \
                              else 'Unknown'
                X[col]      = X[col].fillna(fill_val)
        except Exception as ex:
            print(f"  Warning: could not fill {col} normally: {ex}")
            # Last resort — fill with 0
            X[col] = X[col].fillna(0)

    # ---- Step 3: One-hot encode all remaining text columns ----
    cat_cols = X.select_dtypes(
        include=['object', 'category', 'string']
    ).columns.tolist()

    if cat_cols:
        print(f"  One-hot encoding {len(cat_cols)} "
              f"text columns: {cat_cols}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # ---- Step 4: Convert boolean columns to integer ----
    bool_cols = X.select_dtypes(include='bool').columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(int)

    # ---- Step 5: Convert everything to float64 ----
    # This is what the ML models and SMOTE require
    for col in X.columns:
        try:
            X[col] = X[col].astype(float)
        except Exception:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    print(f"  Final shape: {X.shape}")
    print(f"  Data types: {X.dtypes.value_counts().to_dict()}")
    return X, y


def train_industry_model(industry_key: str, config: dict):
    """Train and save a complete model for one industry."""

    print(f"\n{'='*55}")
    print(f"TRAINING: {config['icon']} {config['name']} model")
    print(f"{'='*55}")

    # Check file exists before doing anything
    if not os.path.exists(config['file']):
        print(f"ERROR: File not found: {config['file']}")
        print(f"Please put the dataset in the datasets/ folder")
        return

    # Load data
    print(f"Loading: {config['file']}")
    df      = load_industry_data(config)
    print(f"Loaded shape: {df.shape}")

    # Preprocess
    X, y = preprocess_industry(df, config)

    print(f"\nCustomers  : {len(X):,}")
    print(f"Features   : {X.shape[1]}")
    print(f"Churned    : {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"Stayed     : {(y==0).sum():,} "
          f"({(1-y.mean())*100:.1f}%)")

    # Scale numerical features
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    # Train/test split — 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"\nTraining rows : {len(X_train):,}")
    print(f"Testing rows  : {len(X_test):,}")

    # SMOTE — balance churners and non-churners
    print("\nApplying SMOTE balancing...")
    smote        = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: "
          f"{pd.Series(y_bal).value_counts().to_dict()}")

    # Train XGBoost model
    print("\nTraining XGBoost model...")
    model = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        gamma            = 0.1,
        eval_metric      = 'logloss',
        random_state     = 42
    )
    model.fit(X_bal, y_bal)
    print("Training complete!")

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\nResults for {config['name']}:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Stay', 'Churn']
    ))
    print(f"AUC Score : {auc:.4f}")

    # 5-fold cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv_model = XGBClassifier(
        n_estimators  = 100,
        max_depth     = 5,
        learning_rate = 0.05,
        eval_metric   = 'logloss',
        random_state  = 42
    )
    cv_scores = cross_val_score(
        cv_model, X_scaled, y,
        cv=5,
        scoring='roc_auc'
    )
    print(f"CV AUC scores : {[round(s, 4) for s in cv_scores]}")
    print(f"CV Mean AUC   : {cv_scores.mean():.4f} "
          f"(+/- {cv_scores.std():.4f})")

    # Save model, scaler, feature names
    joblib.dump(model,              config['model_file'])
    joblib.dump(scaler,             config['scaler_file'])
    joblib.dump(X.columns.tolist(), config['features_file'])

    print(f"\nFiles saved:")
    print(f"  {config['model_file']}")
    print(f"  {config['scaler_file']}")
    print(f"  {config['features_file']}")

    # Store results for final table
    all_results[config['name']] = {
        'Customers': len(X),
        'Features' : X.shape[1],
        'Accuracy' : round(accuracy_score(y_test, y_pred)  * 100, 1),
        'Precision': round(precision_score(y_test, y_pred) * 100, 1),
        'Recall'   : round(recall_score(y_test, y_pred)    * 100, 1),
        'F1-Score' : round(f1_score(y_test, y_pred)        * 100, 1),
        'AUC'      : round(auc * 100, 1),
        'CV_AUC'   : round(cv_scores.mean() * 100, 1),
    }


# ================================================
# TRAIN ALL 3 INDUSTRIES
# ================================================
print("ChurnGuard Pro — Multi-Industry Model Training")
print("=" * 55)

for key, config in INDUSTRIES.items():
    try:
        train_industry_model(key, config)
    except Exception as e:
        import traceback
        print(f"\nERROR training {key}: {e}")
        print(traceback.format_exc())

# ================================================
# FINAL COMPARISON TABLE
# ================================================
print(f"\n{'='*55}")
print("FINAL RESULTS — ALL INDUSTRIES")
print(f"{'='*55}")

if all_results:
    results_df = pd.DataFrame(all_results).T
    print(results_df.to_string())
    results_df.to_csv('models/industry_results.csv')
    print("\nSaved: models/industry_results.csv")
    print("\nAll models trained successfully!")
    print("\nFiles in models/ folder:")
    for f in sorted(os.listdir('models')):
        print(f"  {f}")
else:
    print("No models trained. Check errors above.")