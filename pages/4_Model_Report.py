# pages/4_Model_Report.py — Multi-industry Model Report

import streamlit as st
import pandas as pd
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
    df = pd.read_csv('models/industry_results.csv', index_col=0)
    multi_loaded = True
except Exception:
    multi_loaded = False

# ---- Load original telecom comparison ----
try:
    telecom_df = pd.read_csv('model_comparison.csv')
    telecom_loaded = True
except Exception:
    telecom_loaded = False

# ---- Best model summary ----
if multi_loaded:
    st.subheader("Best Model per Industry")
    c1, c2, c3 = st.columns(3)

    for i, (industry, row) in enumerate(df.iterrows()):
        col = [c1, c2, c3][i]
        col.metric(f"{industry} Best AUC", f"{row['AUC']}%")
        col.metric(f"{industry} Accuracy",  f"{row['Accuracy']}%")
        col.metric(f"{industry} CV AUC",    f"{row['CV_AUC']}%")

    st.divider()

    st.subheader("All Industries Comparison")
    st.dataframe(
        df.style.highlight_max(
            subset=['Accuracy','Precision',
                    'Recall','F1-Score','AUC','CV_AUC'],
            color='#C0DD97'
        ),
        use_container_width=True
    )

    st.divider()

    # AUC comparison chart
    st.subheader("AUC Score by Industry")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        colors  = ['#3B82F6', '#10B981', '#F59E0B']
        bars    = ax.bar(df.index, df['AUC'],
                         color=colors, alpha=0.85)
        ax.set_title('AUC Score by Industry',
                     fontweight='bold')
        ax.set_ylabel('AUC (%)')
        ax.set_ylim(70, 100)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                    f'{h}%', ha='center',
                    fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        bars    = ax.bar(df.index, df['CV_AUC'],
                         color=colors, alpha=0.85)
        ax.set_title('Cross-Validation AUC (5-fold)',
                     fontweight='bold')
        ax.set_ylabel('CV AUC (%)')
        ax.set_ylim(70, 100)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                    f'{h}%', ha='center',
                    fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.divider()

# ---- Original 4 model comparison (Telecom) ----
st.subheader("Telecom — 4 Model Comparison (Phase 1)")

if telecom_loaded:
    st.dataframe(
        telecom_df.style.highlight_max(
            subset=['Accuracy','Precision',
                    'Recall','F1-Score','AUC'],
            color='#C0DD97'
        ),
        use_container_width=True
    )
else:
    st.info("Run train_model.py to see 4-model comparison.")

st.divider()

# ---- Training charts ----
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
                 caption='Feature importance',
                 use_container_width=True)
    except Exception:
        st.info("Run visualize_results.py to generate charts.")

try:
    st.image('neural_network_training.png',
             caption='Neural Network training loss curve',
             use_container_width=True)
except Exception:
    pass