# pages/2_Predict_Single.py — Single customer prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predict Single", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

st.title("Predict Single Customer")
st.write("Enter customer details below and click Predict to get the churn risk.")
st.divider()

# Load model
try:
    model         = joblib.load('churn_model.pkl')
    scaler        = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    model_loaded  = True
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ---- Input form ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    gender       = st.selectbox("Gender",      ["Male", "Female"])
    senior       = st.selectbox("Senior citizen", ["No", "Yes"])
    partner      = st.selectbox("Has partner?",   ["Yes", "No"])
    dependents   = st.selectbox("Has dependents?", ["Yes", "No"])
    tenure       = st.slider("Tenure (months)", 0, 72, 12)
    phone        = st.selectbox("Phone service?", ["Yes", "No"])
    multi_lines  = st.selectbox("Multiple lines?",
                                ["No", "Yes", "No phone service"])

with col2:
    st.subheader("Service & Billing")
    internet     = st.selectbox("Internet service",
                                ["DSL", "Fiber optic", "No"])
    contract     = st.selectbox("Contract type",
                                ["Month-to-month", "One year", "Two year"])
    paperless    = st.selectbox("Paperless billing?", ["Yes", "No"])
    payment      = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly      = st.slider("Monthly charges ($)", 18, 120, 65)
    total        = monthly * tenure

st.divider()

# ---- Build input row matching training columns ----
def build_input():
    row = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

    # Scale numerical features
    num_scaled = scaler.transform([[tenure, monthly, total]])[0]
    for col, val in zip(['tenure', 'MonthlyCharges', 'TotalCharges'], num_scaled):
        if col in row.columns:
            row[col] = val

    # Binary features
    binary_map = {
        'gender'         : 1 if gender == "Male" else 0,
        'SeniorCitizen'  : 1 if senior == "Yes" else 0,
        'Partner'        : 1 if partner == "Yes" else 0,
        'Dependents'     : 1 if dependents == "Yes" else 0,
        'PhoneService'   : 1 if phone == "Yes" else 0,
        'PaperlessBilling': 1 if paperless == "Yes" else 0,
        'MultipleLines'  : {'No phone service': 0, 'No': 1, 'Yes': 2}[multi_lines],
    }
    for col, val in binary_map.items():
        if col in row.columns:
            row[col] = val

    # One-hot features
    onehot = {
        'InternetService_Fiber optic'           : 1 if internet == "Fiber optic" else 0,
        'InternetService_No'                    : 1 if internet == "No" else 0,
        'Contract_One year'                     : 1 if contract == "One year" else 0,
        'Contract_Two year'                     : 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)' : 1 if payment == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check'        : 1 if payment == "Electronic check" else 0,
        'PaymentMethod_Mailed check'            : 1 if payment == "Mailed check" else 0,
    }
    for col, val in onehot.items():
        if col in row.columns:
            row[col] = val

    return row

# ---- Predict button ----
if st.button("Predict Churn Risk", use_container_width=True):
    input_row = build_input()
    prob      = model.predict_proba(input_row)[0][1]
    pred      = model.predict(input_row)[0]

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Probability", f"{prob:.0%}")
    col2.metric("Tenure",            f"{tenure} months")
    col3.metric("Monthly Bill",      f"${monthly}")

    st.progress(float(prob))

    if pred == 1:
        st.error(f"This customer is likely to CHURN  (Risk: {prob:.0%})")
        st.write("**Recommended actions:**")
        st.write("- Call customer and offer a loyalty discount")
        st.write("- Offer upgrade to a One Year or Two Year contract")
        st.write("- Assign a dedicated support agent")
    else:
        st.success(f"This customer is likely to STAY  (Churn risk: {prob:.0%})")
        st.write("**Recommended actions:**")
        st.write("- No immediate action needed")
        st.write("- Continue regular service monitoring")