# pages/4_Model_Report.py — Multi-industry Model Report with GridSearchCV

import streamlit as st
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Report", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

st.title("Model Performance Report")
st.write("Complete comparison of all trained models across 3 industries.")
st.divider()

# ---- Load multi-industry results ----
try:
    df           = pd.read_csv('models/industry_results.csv',
                               index_col=0)
    multi_loaded = True
except Exception:
    multi_loaded = False

# ---- Load telecom 4-model comparison ----
try:
    telecom_df     = pd.read_csv('model_comparison.csv')
    telecom_loaded = True
except Exception:
    telecom_loaded = False

# ================================================
# INDUSTRY SUMMARY METRICS
# ================================================
if multi_loaded:
    st.subheader("Best Model per Industry")

    cols = st.columns(3)
    icons = ['📡', '🏦', '🛒']

    for i, (industry, row) in enumerate(df.iterrows()):
        with cols[i]:
            st.markdown(f"**{icons[i]} {industry}**")
            st.metric("AUC Score",    f"{row['AUC']}%")
            st.metric("CV AUC",       f"{row['CV_AUC']}%")
            st.metric("Accuracy",     f"{row['Accuracy']}%")
            st.metric("Recall",       f"{row['Recall']}%")

    st.divider()

    # ---- Full comparison table ----
    st.subheader("All Industries — Full Metrics")

    display_cols = [
        c for c in [
            'Customers', 'Features', 'Accuracy',
            'Precision', 'Recall', 'F1-Score', 'AUC', 'CV_AUC'
        ] if c in df.columns
    ]

    st.dataframe(
        df[display_cols].style.highlight_max(
            subset=[c for c in display_cols
                    if c not in ['Customers', 'Features']],
            color='#C0DD97'
        ),
        use_container_width=True
    )
    st.caption(
        "Green highlight = best score in that column. "
        "CV_AUC is cross-validated — more reliable than single AUC."
    )

    st.divider()

    # ---- AUC and CV charts ----
    st.subheader("Visual Comparison")

    col1, col2 = st.columns(2)
    colors     = ['#3B82F6', '#10B981', '#F59E0B']

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        bars    = ax.bar(df.index, df['AUC'],
                         color=colors, alpha=0.85, width=0.5)
        ax.set_title('AUC Score by Industry', fontweight='bold')
        ax.set_ylabel('AUC (%)')
        ax.set_ylim(70, 100)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                    f'{h}%', ha='center',
                    fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        cv_vals = df['CV_AUC'].values if 'CV_AUC' in df.columns \
                  else df['AUC'].values
        bars    = ax.bar(df.index, cv_vals,
                         color=colors, alpha=0.85, width=0.5)
        ax.set_title('Cross-Validation AUC (5-fold)',
                     fontweight='bold')
        ax.set_ylabel('CV AUC (%)')
        ax.set_ylim(70, 100)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                    f'{h}%', ha='center',
                    fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ---- GridSearchCV best parameters ----
    st.subheader("Best Parameters Found by GridSearchCV")
    st.write(
        "GridSearchCV tested 12 parameter combinations × 3-fold CV "
        "= 36 training runs per industry to find these optimal values:"
    )

    param_data = []
    for key, name in [('telecom','📡 Telecom'),
                      ('banking','🏦 Banking'),
                      ('ecommerce','🛒 E-Commerce')]:
        param_file = f'models/{key}_model_params.pkl'
        try:
            params = joblib.load(param_file)
            param_data.append({
                'Industry'      : name,
                'n_estimators'  : params.get('n_estimators', '-'),
                'max_depth'     : params.get('max_depth', '-'),
                'learning_rate' : params.get('learning_rate', '-'),
                'subsample'     : params.get('subsample', 0.8),
            })
        except Exception:
            param_data.append({
                'Industry'      : name,
                'n_estimators'  : 'Run multi_train.py',
                'max_depth'     : '-',
                'learning_rate' : '-',
                'subsample'     : '-',
            })

    st.dataframe(
        pd.DataFrame(param_data),
        use_container_width=True
    )
    st.caption(
        "These parameters were automatically discovered — "
        "not manually set. This is Level 1 Part 2 of ChurnGuard Pro."
    )

    st.divider()

# ================================================
# TELECOM 4-MODEL COMPARISON (Phase 1)
# ================================================
st.subheader("Telecom — 4 Model Comparison (Phase 1)")
st.write("Logistic Regression, Random Forest, XGBoost, Neural Network")

if telecom_loaded:
    st.dataframe(
        telecom_df.style.highlight_max(
            subset=[c for c in
                    ['Accuracy','Precision','Recall','F1-Score','AUC']
                    if c in telecom_df.columns],
            color='#C0DD97'
        ),
        use_container_width=True
    )
else:
    st.info("Run train_model.py to see 4-model comparison.")

st.divider()

# ================================================
# TRAINING CHARTS
# ================================================
st.subheader("Generated Charts from Training")

col1, col2 = st.columns(2)
with col1:
    try:
        st.image('model_comparison_chart.png',
                 caption='Model comparison chart',
                 use_container_width=True)
    except Exception:
        st.info("Run visualize_results.py to generate charts.")

with col2:
    try:
        st.image('feature_importance.png',
                 caption='Feature importance — Telecom XGBoost',
                 use_container_width=True)
    except Exception:
        st.info("Run visualize_results.py to generate charts.")

try:
    st.image('neural_network_training.png',
             caption='Neural Network training loss curve',
             use_container_width=True)
except Exception:
    pass

st.divider()
st.caption(
    "ChurnGuard Pro | Model: XGBoost with GridSearchCV | "
    "Industries: Telecom, Banking, E-Commerce | "
    "Evaluation: 5-fold Cross-Validation"
)