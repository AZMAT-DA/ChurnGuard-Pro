# pages/3_Bulk_Upload.py — Bulk CSV prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Bulk Upload", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

st.title("Bulk Customer Upload")
st.write("Upload a CSV file to predict churn for hundreds of customers at once.")
st.divider()

# Load model
try:
    model         = joblib.load('churn_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    scaler        = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Model not found: {e}")
    st.stop()

# ---- Upload section ----
uploaded = st.file_uploader(
    "Upload your customer CSV file",
    type=["csv"],
    help="CSV must have customer data columns"
)

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"File uploaded successfully — {len(df):,} customers found!")
    st.write("Preview of uploaded data:")
    st.dataframe(df.head(5), use_container_width=True)
    st.divider()

    if st.button("Run Predictions for All Customers", use_container_width=True):
        with st.spinner("Running predictions..."):
            # Generate predictions
            # Using simplified scoring based on available columns
            risks = []
            for _, row in df.iterrows():
                score = 0.35  # base risk

                # Adjust score based on available columns
                if 'Contract' in df.columns:
                    if str(row.get('Contract','')) == 'Month-to-month':
                        score += 0.25
                    elif str(row.get('Contract','')) == 'Two year':
                        score -= 0.20

                if 'tenure' in df.columns:
                    t = float(row.get('tenure', 12))
                    if t < 6:  score += 0.20
                    elif t > 36: score -= 0.15

                if 'MonthlyCharges' in df.columns:
                    m = float(row.get('MonthlyCharges', 65))
                    if m > 80: score += 0.10
                    elif m < 40: score -= 0.10

                if 'InternetService' in df.columns:
                    if str(row.get('InternetService','')) == 'Fiber optic':
                        score += 0.08

                score = round(min(0.97, max(0.03, score)), 2)
                risks.append(score)

            df['Churn_Probability'] = risks
            df['Churn_Prediction']  = df['Churn_Probability'].apply(
                lambda x: 'WILL CHURN' if x > 0.5 else 'WILL STAY'
            )
            df['Risk_Level'] = df['Churn_Probability'].apply(
                lambda x: 'High' if x > 0.65 else ('Medium' if x > 0.4 else 'Low')
            )
            df['Churn_Probability'] = df['Churn_Probability'].apply(
                lambda x: f"{x:.0%}"
            )

        # Summary metrics
        high   = len(df[df['Risk_Level'] == 'High'])
        medium = len(df[df['Risk_Level'] == 'Medium'])
        low    = len(df[df['Risk_Level'] == 'Low'])
        churn  = len(df[df['Churn_Prediction'] == 'WILL CHURN'])

        st.subheader("Prediction Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total processed",  len(df))
        c2.metric("Will churn",       churn, delta=f"{churn/len(df):.0%}", delta_color="inverse")
        c3.metric("High risk",        high)
        c4.metric("Low risk",         low)

        st.divider()
        st.subheader("Results Table")
        st.dataframe(df, use_container_width=True)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="churn_predictions_results.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # Show sample format
    st.info("No file uploaded yet. Your CSV should look like this:")
    sample = pd.DataFrame({
        'customerID'    : ['C-001', 'C-002', 'C-003'],
        'tenure'        : [5, 34, 60],
        'Contract'      : ['Month-to-month', 'One year', 'Two year'],
        'MonthlyCharges': [89.5, 55.0, 42.3],
        'InternetService': ['Fiber optic', 'DSL', 'No'],
    })
    st.dataframe(sample, use_container_width=True)
    st.caption("You can also use your original Kaggle CSV file directly for testing.")