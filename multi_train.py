# multi_train.py — Train models for all 3 industries
# Level 1 Part 2: GridSearchCV + Part 3: Cross-Validation

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection    import (train_test_split, cross_val_score,
                                        GridSearchCV)
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (classification_report,
                                        roc_auc_score, accuracy_score,
                                        precision_score, recall_score,
                                        f1_score)
from xgboost                    import XGBClassifier
from imblearn.over_sampling     import SMOTE
from industry_config            import INDUSTRIES

os.makedirs('models', exist_ok=True)
all_results = {}


def load_industry_data(config: dict) -> pd.DataFrame:
    if config['file_type'] == 'csv':
        df = pd.read_csv(config['file'])
    else:
        df = pd.read_excel(config['file'], sheet_name=config['sheet'])
    return df


def preprocess_industry(df: pd.DataFrame, config: dict):
    df = df.drop(columns=config['drop_cols'], errors='ignore')
    target = config['target']
    if config['target_map']:
        df[target] = df[target].map(config['target_map'])
    df = df.dropna(subset=[target])
    y  = df[target].astype(int)
    X  = df.drop(columns=[target])
    X  = X.reset_index(drop=True)
    y  = y.reset_index(drop=True)

    # Convert columns to numeric where possible
    for col in X.columns:
        try:
            converted = pd.to_numeric(X[col], errors='coerce')
            if converted.notna().sum() > len(X) * 0.8:
                X[col] = converted
        except Exception:
            pass

    # Fill missing values safely
    for col in X.columns:
        try:
            if pd.api.types.is_numeric_dtype(X[col]):
                med = X[col].dropna().median()
                X[col] = X[col].fillna(
                    med if not np.isnan(med) else 0
                )
            else:
                mode_v = X[col].dropna().mode()
                X[col] = X[col].fillna(
                    mode_v.iloc[0] if len(mode_v) > 0 else 'Unknown'
                )
        except Exception:
            X[col] = X[col].fillna(0)

    # One-hot encode text columns
    cat_cols = X.select_dtypes(
        include=['object', 'category', 'string']
    ).columns.tolist()
    if cat_cols:
        print(f"  Encoding {len(cat_cols)} text columns")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Convert bool to int
    bool_cols = X.select_dtypes(include='bool').columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(int)

    # Convert all to float
    for col in X.columns:
        try:
            X[col] = X[col].astype(float)
        except Exception:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    print(f"  Final shape: {X.shape}")
    return X, y


def train_industry_model(industry_key: str, config: dict):
    print(f"\n{'='*55}")
    print(f"TRAINING: {config['icon']} {config['name']}")
    print(f"{'='*55}")

    if not os.path.exists(config['file']):
        print(f"ERROR: File not found: {config['file']}")
        return

    df   = load_industry_data(config)
    X, y = preprocess_industry(df, config)

    print(f"Customers : {len(X):,}")
    print(f"Features  : {X.shape[1]}")
    print(f"Churn rate: {y.mean()*100:.1f}%")

    # Scale
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE
    smote        = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {pd.Series(y_bal).value_counts().to_dict()}")

    # ================================================
    # LEVEL 1 PART 2 — GridSearchCV
    # Automatically finds best XGBoost parameters
    # ================================================
    print("\nRunning GridSearchCV to find best parameters...")
    print("Testing combinations of: n_estimators, max_depth, learning_rate")

    # Parameter grid — we test these combinations
    param_grid = {
        'n_estimators'  : [100, 300],
        'max_depth'     : [4, 5, 6],
        'learning_rate' : [0.05, 0.1],
        'subsample'     : [0.8],
        'colsample_bytree': [0.8],
    }

    base_model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        min_child_weight=3,
        gamma=0.1
    )

    grid_search = GridSearchCV(
        estimator  = base_model,
        param_grid = param_grid,
        cv         = 3,          # 3-fold CV inside grid search
        scoring    = 'roc_auc',  # optimize for AUC
        n_jobs     = -1,         # use all CPU cores
        verbose    = 1
    )
    grid_search.fit(X_bal, y_bal)

    print(f"\nBest parameters found by GridSearchCV:")
    for param, val in grid_search.best_params_.items():
        print(f"  {param}: {val}")
    print(f"Best CV AUC (during search): "
          f"{grid_search.best_score_:.4f}")

    # Train final model with best parameters
    best_model = grid_search.best_estimator_
    print("\nTraining final model with best parameters...")

    # ================================================
    # LEVEL 1 PART 3 — Cross-Validation
    # Tests model 5 times for reliable score
    # ================================================
    print("\nRunning 5-fold cross-validation for reliable evaluation...")
    cv_scores = cross_val_score(
        best_model,
        X_scaled, y,
        cv=5,
        scoring='roc_auc'
    )
    print(f"CV AUC scores : {[round(s,4) for s in cv_scores]}")
    print(f"CV Mean AUC   : {cv_scores.mean():.4f} "
          f"(+/- {cv_scores.std():.4f})")
    print(f"\nWhat cross-validation tells us:")
    print(f"  Mean AUC {cv_scores.mean()*100:.1f}% is the TRUE "
          f"reliable performance")
    print(f"  Low variance ({cv_scores.std()*100:.2f}%) means "
          f"model is stable and consistent")

    # Evaluate on held-out test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\nFinal Results for {config['name']}:")
    print(classification_report(
        y_test, y_pred, target_names=['Stay', 'Churn']
    ))
    print(f"Test AUC   : {auc:.4f}")
    print(f"CV AUC     : {cv_scores.mean():.4f}")

    # Save best model
    joblib.dump(best_model,         config['model_file'])
    joblib.dump(scaler,             config['scaler_file'])
    joblib.dump(X.columns.tolist(), config['features_file'])

    # Save best params for reference
    joblib.dump(
        grid_search.best_params_,
        config['model_file'].replace('.pkl', '_params.pkl')
    )

    print(f"\nSaved: {config['model_file']}")

    all_results[config['name']] = {
        'Customers'     : len(X),
        'Features'      : X.shape[1],
        'Accuracy'      : round(accuracy_score(y_test,y_pred)*100, 1),
        'Precision'     : round(precision_score(y_test,y_pred)*100, 1),
        'Recall'        : round(recall_score(y_test,y_pred)*100, 1),
        'F1-Score'      : round(f1_score(y_test,y_pred)*100, 1),
        'AUC'           : round(auc*100, 1),
        'CV_AUC'        : round(cv_scores.mean()*100, 1),
        'CV_Std'        : round(cv_scores.std()*100, 2),
        'Best_Estimators'  : grid_search.best_params_['n_estimators'],
        'Best_Depth'       : grid_search.best_params_['max_depth'],
        'Best_LR'          : grid_search.best_params_['learning_rate'],
    }


# Train all 3 industries
print("ChurnGuard Pro — Level 1 Part 2+3 Training")
print("GridSearchCV + Cross-Validation")
print("=" * 55)

for key, config in INDUSTRIES.items():
    try:
        train_industry_model(key, config)
    except Exception as e:
        import traceback
        print(f"\nERROR training {key}: {e}")
        print(traceback.format_exc())

# Final comparison
print(f"\n{'='*55}")
print("FINAL RESULTS WITH GRIDSEARCHCV")
print(f"{'='*55}")

if all_results:
    results_df = pd.DataFrame(all_results).T
    print(results_df[[
        'Customers','Accuracy','Precision',
        'Recall','F1-Score','AUC','CV_AUC','CV_Std'
    ]].to_string())

    results_df.to_csv('models/industry_results.csv')
    print("\nSaved: models/industry_results.csv")

    # Show best params per industry
    print(f"\n{'='*55}")
    print("BEST PARAMETERS FOUND PER INDUSTRY")
    print(f"{'='*55}")
    for industry, row in results_df.iterrows():
        print(f"\n{industry}:")
        print(f"  n_estimators : {int(row['Best_Estimators'])}")
        print(f"  max_depth    : {int(row['Best_Depth'])}")
        print(f"  learning_rate: {row['Best_LR']}")
        print(f"  CV AUC       : {row['CV_AUC']}% "
              f"(+/- {row['CV_Std']}%)")

    print("\nAll models trained with optimal parameters!")
else:
    print("No models trained. Check errors above.")