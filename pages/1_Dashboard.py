# pages/1_Dashboard.py — Analytics Dashboard

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

st.title("Analytics Dashboard")
st.write("Overview of your customer churn data and model performance.")
st.divider()

# ---- Load data ----
try:
    df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

# ---- Key Metrics ----
st.subheader("Key Metrics")

total       = len(df_raw)
churned     = df_raw['Churn'].value_counts().get('Yes', 0)
stayed      = df_raw['Churn'].value_counts().get('No', 0)
churn_rate  = round(churned / total * 100, 1)
avg_monthly = round(df_raw['MonthlyCharges'].mean(), 2)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers",  f"{total:,}")
col2.metric("Churned",          f"{churned:,}", delta=f"-{churn_rate}%", delta_color="inverse")
col3.metric("Retained",         f"{stayed:,}")
col4.metric("Churn Rate",       f"{churn_rate}%")
col5.metric("Avg Monthly Bill", f"${avg_monthly}")

st.divider()

# ---- Charts ----
st.subheader("Churn Analysis Charts")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Churn by contract type")
    contract_churn = df_raw.groupby('Contract')['Churn'].apply(
        lambda x: round((x == 'Yes').sum() / len(x) * 100, 1)
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    colors = ['#E24B4A', '#EF9F27', '#639922']
    ax.bar(contract_churn.index, contract_churn.values, color=colors, alpha=0.85)
    ax.set_ylabel('Churn rate (%)')
    ax.set_ylim(0, 55)
    for i, v in enumerate(contract_churn.values):
        ax.text(i, v + 0.5, f'{v}%', ha='center', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.write("Churn by internet service")
    internet_churn = df_raw.groupby('InternetService')['Churn'].apply(
        lambda x: round((x == 'Yes').sum() / len(x) * 100, 1)
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(internet_churn.index, internet_churn.values,
           color=['#E24B4A', '#639922', '#EF9F27'], alpha=0.85)
    ax.set_ylabel('Churn rate (%)')
    ax.set_ylim(0, 55)
    for i, v in enumerate(internet_churn.values):
        ax.text(i, v + 0.5, f'{v}%', ha='center', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col3:
    st.write("Tenure distribution by churn")
    fig, ax = plt.subplots(figsize=(4, 3))
    churned_t = df_raw[df_raw['Churn'] == 'Yes']['tenure']
    stayed_t  = df_raw[df_raw['Churn'] == 'No']['tenure']
    ax.hist(stayed_t,  bins=20, alpha=0.6, color='#639922', label='Stayed')
    ax.hist(churned_t, bins=20, alpha=0.6, color='#E24B4A', label='Churned')
    ax.set_xlabel('Tenure (months)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ---- Model Comparison ----
st.subheader("Model Performance (Phase 1 Results)")
try:
    comparison_df = pd.read_csv('model_comparison.csv')
    st.dataframe(
        comparison_df.style.highlight_max(
            subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
            color='#C0DD97'
        ),
        use_container_width=True
    )
except Exception:
    st.info("model_comparison.csv not found.")

st.divider()

# ---- Raw Data Preview ----
st.subheader("Dataset Preview")
st.write(f"Showing first 10 rows of {total:,} total customers")
st.dataframe(df_raw.head(10), use_container_width=True)