# pages/10_Retrain.py — Model Retraining Pipeline

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (roc_auc_score, accuracy_score,
                                        precision_score, recall_score,
                                        f1_score, classification_report)
from xgboost                    import XGBClassifier
from imblearn.over_sampling     import SMOTE
from industry_config            import INDUSTRIES

st.set_page_config(page_title="Retrain Models", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

if st.session_state.role != 'admin':
    st.error("Access denied. Admin only feature.")
    st.stop()

st.title("Model Retraining Pipeline")
st.write(
    "Upload new customer data and retrain the model for any industry. "
    "The system automatically preprocesses, trains, evaluates, "
    "and saves the new model."
)
st.divider()

st.warning(
    "**Admin only feature.** Retraining replaces the current model. "
    "Make sure your new data is clean and representative before retraining."
)

# ================================================
# STEP 1 — SELECT INDUSTRY
# ================================================
st.subheader("Step 1 — Select Industry to Retrain")

industry = st.selectbox(
    "Industry",
    options=['telecom', 'banking', 'ecommerce'],
    format_func=lambda x: {
        'telecom'  : '📡 Telecom',
        'banking'  : '🏦 Banking',
        'ecommerce': '🛒 E-Commerce'
    }[x]
)

config = INDUSTRIES[industry]

# Show current model info
st.info(
    f"Current model: `{config['model_file']}` — "
    f"will be replaced after retraining."
)

# ================================================
# STEP 2 — UPLOAD NEW DATA
# ================================================
st.subheader("Step 2 — Upload New Training Data")

st.write(
    "Upload a CSV file with customer data for this industry. "
    "The file must contain the target column (churn indicator)."
)

# Show expected format
format_info = {
    'telecom'  : {
        'target'    : 'Churn (Yes/No)',
        'key_cols'  : 'tenure, MonthlyCharges, Contract, InternetService',
        'example'   : 'WA_Fn-UseC_-Telco-Customer-Churn.csv format'
    },
    'banking'  : {
        'target'    : 'Exited (0/1)',
        'key_cols'  : 'CreditScore, Age, Balance, NumOfProducts',
        'example'   : 'Churn_Modelling.csv format'
    },
    'ecommerce': {
        'target'    : 'Churn (0/1)',
        'key_cols'  : 'Tenure, SatisfactionScore, Complain, CashbackAmount',
        'example'   : 'E_Commerce_Dataset.xlsx E Comm sheet format'
    }
}

info = format_info[industry]
st.markdown(f"""
- **Target column:** `{info['target']}`
- **Key feature columns:** `{info['key_cols']}`
- **Expected format:** {info['example']}
""")

uploaded = st.file_uploader(
    "Upload your new training CSV",
    type=["csv"],
    help="CSV file with customer data and churn labels"
)

if uploaded:
    df_new = pd.read_csv(uploaded)
    st.success(f"Uploaded {len(df_new):,} rows, "
               f"{len(df_new.columns)} columns")
    st.dataframe(df_new.head(5), use_container_width=True)

    # Check target column exists
    target_col = config['target']
    if target_col not in df_new.columns:
        st.error(
            f"Target column '{target_col}' not found in uploaded file. "
            f"Available columns: {df_new.columns.tolist()}"
        )
        st.stop()
    else:
        st.success(f"Target column '{target_col}' found.")

    st.divider()

    # ================================================
    # STEP 3 — RETRAIN SETTINGS
    # ================================================
    st.subheader("Step 3 — Training Settings")

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider(
            "Test set size (%)", 10, 30, 20
        )
        use_smote = st.checkbox(
            "Use SMOTE balancing", value=True
        )
    with col2:
        n_estimators  = st.selectbox(
            "Number of trees", [100, 200, 300], index=1
        )
        use_gridsearch = st.checkbox(
            "Use GridSearchCV (slower but better)", value=False
        )

    st.divider()

    # ================================================
    # STEP 4 — RETRAIN BUTTON
    # ================================================
    st.subheader("Step 4 — Start Retraining")

    if st.button("Retrain Model Now",
                 use_container_width=True,
                 type="primary"):

        progress = st.progress(0)
        status   = st.empty()

        try:
            # ---- Preprocessing ----
            status.info("Step 1/6 — Preprocessing data...")
            progress.progress(10)

            df_train = df_new.drop(
                columns=config['drop_cols'], errors='ignore'
            )

            # Map target
            if config['target_map']:
                df_train[target_col] = df_train[target_col].map(
                    config['target_map']
                )

            df_train = df_train.dropna(subset=[target_col])
            y = df_train[target_col].astype(int)
            X = df_train.drop(columns=[target_col])
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

            # Convert to numeric
            for col in X.columns:
                try:
                    conv = pd.to_numeric(X[col], errors='coerce')
                    if conv.notna().sum() > len(X) * 0.8:
                        X[col] = conv
                except Exception:
                    pass

            # Fill missing
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].median())
                else:
                    mode = X[col].dropna().mode()
                    X[col] = X[col].fillna(
                        mode.iloc[0] if len(mode) > 0 else 'Unknown'
                    )

            # Encode categoricals
            cat_cols = X.select_dtypes(
                include=['object','category','string']
            ).columns.tolist()
            if cat_cols:
                X = pd.get_dummies(
                    X, columns=cat_cols, drop_first=True
                )

            bool_cols = X.select_dtypes(include='bool').columns
            if len(bool_cols) > 0:
                X[bool_cols] = X[bool_cols].astype(int)
            X = X.astype(float)

            status.info(
                f"Step 1/6 — Data preprocessed: "
                f"{X.shape[0]:,} rows, {X.shape[1]} features"
            )
            progress.progress(20)

            # ---- Scale ----
            status.info("Step 2/6 — Scaling features...")
            scaler   = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), columns=X.columns
            )
            progress.progress(30)

            # ---- Split ----
            status.info("Step 3/6 — Splitting train/test...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y,
                test_size=test_size/100,
                random_state=42,
                stratify=y
            )
            progress.progress(40)

            # ---- SMOTE ----
            if use_smote:
                status.info("Step 4/6 — Balancing with SMOTE...")
                smote        = SMOTE(random_state=42)
                X_bal, y_bal = smote.fit_resample(X_train, y_train)
            else:
                X_bal, y_bal = X_train, y_train
            progress.progress(50)

            # ---- Train ----
            status.info("Step 5/6 — Training XGBoost model...")

            if use_gridsearch:
                from sklearn.model_selection import GridSearchCV
                param_grid = {
                    'n_estimators' : [100, 200],
                    'max_depth'    : [4, 5],
                    'learning_rate': [0.05, 0.1],
                }
                base = XGBClassifier(
                    eval_metric='logloss', random_state=42
                )
                gs   = GridSearchCV(
                    base, param_grid, cv=3,
                    scoring='roc_auc', n_jobs=-1
                )
                gs.fit(X_bal, y_bal)
                model = gs.best_estimator_
                st.info(f"Best params: {gs.best_params_}")
            else:
                model = XGBClassifier(
                    n_estimators  = n_estimators,
                    max_depth     = 5,
                    learning_rate = 0.05,
                    subsample     = 0.8,
                    eval_metric   = 'logloss',
                    random_state  = 42
                )
                model.fit(X_bal, y_bal)

            progress.progress(70)

            # ---- Evaluate ----
            status.info("Step 6/6 — Evaluating model...")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc    = roc_auc_score(y_test, y_prob)

            cv_scores = cross_val_score(
                model, X_scaled, y, cv=5, scoring='roc_auc'
            )
            progress.progress(85)

            # ---- Save ----
            status.info("Saving new model...")
            joblib.dump(model,              config['model_file'])
            joblib.dump(scaler,             config['scaler_file'])
            joblib.dump(X.columns.tolist(), config['features_file'])
            progress.progress(100)

            # ---- Results ----
            status.success("Model retrained and saved successfully!")

            st.divider()
            st.subheader("New Model Performance")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Accuracy",
                f"{accuracy_score(y_test,y_pred)*100:.1f}%"
            )
            c2.metric(
                "AUC Score",
                f"{auc*100:.1f}%"
            )
            c3.metric(
                "CV AUC (5-fold)",
                f"{cv_scores.mean()*100:.1f}%"
            )
            c4.metric(
                "Recall",
                f"{recall_score(y_test,y_pred)*100:.1f}%"
            )

            st.subheader("Classification Report")
            report = classification_report(
                y_test, y_pred,
                target_names=['Stay','Churn'],
                output_dict=True
            )
            st.dataframe(
                pd.DataFrame(report).T,
                use_container_width=True
            )

            st.subheader("Cross-Validation Scores")
            cv_df = pd.DataFrame({
                'Fold'     : [f"Fold {i+1}" for i in range(5)],
                'AUC Score': [f"{s*100:.2f}%" for s in cv_scores]
            })
            st.dataframe(
                cv_df, use_container_width=True, hide_index=True
            )
            st.metric(
                "Mean CV AUC",
                f"{cv_scores.mean()*100:.2f}% "
                f"(+/- {cv_scores.std()*100:.2f}%)"
            )

            st.success(
                f"The {industry.capitalize()} model has been "
                "retrained with your new data. "
                "All predictions will now use this new model."
            )

        except Exception as e:
            import traceback
            st.error(f"Retraining failed: {e}")
            st.code(traceback.format_exc())
            progress.empty()