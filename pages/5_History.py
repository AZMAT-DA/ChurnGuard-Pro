# pages/5_History.py — Prediction history tracker

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="History", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

st.title("Prediction History")
st.write("All predictions made through the Predict Single page are logged here.")
st.divider()

history_file = 'prediction_history.csv'

if not os.path.exists(history_file):
    st.info("No predictions made yet. Go to Predict Single and make "
            "your first prediction — it will appear here automatically.")
    st.stop()

# Load history
df = pd.read_csv(history_file)

if df.empty:
    st.info("History file is empty. Make a prediction first.")
    st.stop()

# ---- Summary metrics ----
st.subheader("Summary")

total      = len(df)
churned    = len(df[df['Prediction'] == 'CHURN'])
stayed     = len(df[df['Prediction'] == 'STAY'])
high_risk  = len(df[df['Risk_Level'] == 'High Risk']) if 'Risk_Level' in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total predictions",  total)
c2.metric("Predicted churn",    churned)
c3.metric("Predicted stay",     stayed)
c4.metric("High risk flagged",  high_risk)

st.divider()

# ---- Filter options ----
st.subheader("Filter History")

col1, col2 = st.columns(2)
with col1:
    pred_filter = st.selectbox(
        "Filter by prediction",
        ["All", "CHURN", "STAY"]
    )
with col2:
    if 'Risk_Level' in df.columns:
        risk_filter = st.selectbox(
            "Filter by risk level",
            ["All", "High Risk", "Medium Risk", "Low Risk"]
        )
    else:
        risk_filter = "All"

# Apply filters
filtered = df.copy()
if pred_filter != "All":
    filtered = filtered[filtered['Prediction'] == pred_filter]
if risk_filter != "All" and 'Risk_Level' in filtered.columns:
    filtered = filtered[filtered['Risk_Level'] == risk_filter]

st.write(f"Showing {len(filtered)} of {total} predictions")

# ---- History table ----
st.subheader("Prediction Log")
st.dataframe(filtered, use_container_width=True)

st.divider()

# ---- Download history ----
col1, col2 = st.columns(2)

with col1:
    csv_data = filtered.to_csv(index=False)
    st.download_button(
        label="Download filtered history as CSV",
        data=csv_data,
        file_name="prediction_history_export.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    if st.button("Clear all history", use_container_width=True):
        os.remove(history_file)
        st.success("History cleared.")
        st.rerun()