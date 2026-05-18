# pages/4_Model_Report.py — Model Performance Report

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="Model Report", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

st.title("Model Performance Report")
st.write("Complete comparison of all 4 models trained in Phase 1.")
st.divider()

# ---- Load comparison data ----
try:
    df = pd.read_csv('model_comparison.csv')
except:
    st.error("model_comparison.csv not found. Run train_model.py first.")
    st.stop()

# ---- Summary cards ----
best = df.loc[df['AUC'].idxmax()]
st.subheader("Best Model Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Model",    best['Model'])
c2.metric("Best AUC",      f"{best['AUC']}%")
c3.metric("Best Recall",   f"{best['Recall']}%")
c4.metric("Best F1-Score", f"{best['F1-Score']}%")

st.divider()

# ---- Full comparison table ----
st.subheader("All Models Comparison")
st.dataframe(
    df.style.highlight_max(
        subset=['Accuracy','Precision','Recall','F1-Score','AUC'],
        color='#C0DD97'
    ),
    use_container_width=True
)

st.divider()

# ---- Charts ----
st.subheader("Visual Comparison")

col1, col2 = st.columns(2)

with col1:
    # AUC bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    colors  = ['#3B82F6','#8B5CF6','#10B981','#F59E0B'][:len(df)]
    bars    = ax.bar(df['Model'], df['AUC'], color=colors, alpha=0.85)
    ax.set_title('AUC Score by Model', fontweight='bold')
    ax.set_ylabel('AUC (%)')
    ax.set_ylim(70, 90)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                f'{h}%', ha='center', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8, rotation=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    # Recall bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    bars    = ax.bar(df['Model'], df['Recall'], color=colors, alpha=0.85)
    ax.set_title('Recall Score by Model', fontweight='bold')
    ax.set_ylabel('Recall (%)')
    ax.set_ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f'{h}%', ha='center', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8, rotation=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ---- Model charts if available ----
st.subheader("Generated Charts from Training")
col1, col2 = st.columns(2)

with col1:
    try:
        st.image('model_comparison_chart.png',
                 caption='Model comparison chart', use_column_width=True)
    except:
        st.info("Run visualize_results.py to generate this chart.")

with col2:
    try:
        st.image('feature_importance.png',
                 caption='Feature importance', use_column_width=True)
    except:
        st.info("Run visualize_results.py to generate this chart.")

try:
    st.image('neural_network_training.png',
             caption='Neural Network training loss curve',
             use_column_width=True)
except:
    pass