# pages/2_Predict_Single.py — Single prediction with SHAP + PDF + database

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

st.set_page_config(page_title="Predict Single", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

# Load model files
try:
    model         = joblib.load('churn_model.pkl')
    scaler        = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Load helper modules
try:
    from recommendations import get_recommendations, get_risk_label
    from pdf_report       import generate_churn_report
    from database         import save_prediction
    modules_loaded = True
except Exception as e:
    st.warning(f"Helper modules not loaded: {e}")
    modules_loaded = False

st.title("Predict Single Customer")
st.write("Enter customer details and click Predict to get churn risk, "
         "SHAP explanation, smart recommendations, and a PDF report.")
st.divider()

# ---- Input Form ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    gender      = st.selectbox("Gender",           ["Male", "Female"])
    senior      = st.selectbox("Senior citizen",   ["No", "Yes"])
    partner     = st.selectbox("Has partner?",     ["Yes", "No"])
    dependents  = st.selectbox("Has dependents?",  ["Yes", "No"])
    tenure      = st.slider("Tenure (months)", 0, 72, 12)
    phone       = st.selectbox("Phone service?",   ["Yes", "No"])
    multi_lines = st.selectbox("Multiple lines?",
                               ["No", "Yes", "No phone service"])

with col2:
    st.subheader("Service & Billing")
    internet  = st.selectbox("Internet service",
                             ["DSL", "Fiber optic", "No"])
    tech_supp = st.selectbox("Tech support",
                             ["No", "Yes", "No internet service"])
    contract  = st.selectbox("Contract type",
                             ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing?", ["Yes", "No"])
    payment   = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.slider("Monthly charges ($)", 18, 120, 65)
    total   = monthly * tenure

st.divider()

# ---- Build input row ----
def build_input():
    row = pd.DataFrame(
        [np.zeros(len(feature_names))],
        columns=feature_names
    )
    num_scaled = scaler.transform([[tenure, monthly, total]])[0]
    for col, val in zip(['tenure', 'MonthlyCharges', 'TotalCharges'], num_scaled):
        if col in row.columns:
            row[col] = val

    binary_map = {
        'gender'          : 1 if gender == "Male" else 0,
        'SeniorCitizen'   : 1 if senior == "Yes" else 0,
        'Partner'         : 1 if partner == "Yes" else 0,
        'Dependents'      : 1 if dependents == "Yes" else 0,
        'PhoneService'    : 1 if phone == "Yes" else 0,
        'PaperlessBilling': 1 if paperless == "Yes" else 0,
        'MultipleLines'   : {'No phone service': 0, 'No': 1, 'Yes': 2}[multi_lines],
    }
    for col, val in binary_map.items():
        if col in row.columns:
            row[col] = val

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
    pred      = int(model.predict(input_row)[0])

    # Build customer data dict
    customer_data = {
        'tenure'          : f"{tenure} months",
        'monthly_charges' : f"${monthly}",
        'contract'        : contract,
        'internet'        : internet,
        'payment'         : payment,
        'senior'          : senior,
        'partner'         : partner,
        'dependents'      : dependents,
        'phone'           : phone,
        'paperless'       : paperless,
        'tech_support'    : tech_supp,
    }

    # Get risk level
    if modules_loaded:
        risk = get_risk_label(prob)
    else:
        risk = ("High Risk" if prob > 0.65 else
                "Medium Risk" if prob > 0.4 else "Low Risk")

    # Save to database
    if modules_loaded:
        try:
            save_prediction(
                user_id       = st.session_state.get('user_id', 0),
                username      = st.session_state.get('username', 'unknown'),
                customer_data = customer_data,
                prob          = prob,
                pred          = pred,
                risk_level    = risk
            )
        except Exception as e:
            st.warning(f"Could not save to history: {e}")

    st.divider()

    # ---- Results metrics ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn Probability", f"{prob:.0%}")
    c2.metric("Tenure",            f"{tenure} months")
    c3.metric("Monthly Bill",      f"${monthly}")
    c4.metric("Contract",          contract)

    st.progress(float(prob))

    if pred == 1:
        st.error(f"This customer is likely to CHURN — Risk: {prob:.0%}")
    else:
        st.success(f"This customer is likely to STAY — Churn risk: {prob:.0%}")

    st.divider()

    # ================================================
    # SHAP EXPLANATION
    # ================================================
    st.subheader("Why did the model predict this?")

    shap_loaded = False
    shap_recs   = []

    try:
        from shap_explainer import (get_shap_explanation,
                                    get_shap_recommendations,
                                    plot_shap_bar)

        with st.spinner("Calculating SHAP explanation..."):
            shap_exp   = get_shap_explanation(input_row)
            shap_recs  = get_shap_recommendations(shap_exp, pred)
            shap_chart = plot_shap_bar(shap_exp)
            shap_loaded = True

        # Show SHAP bar chart
        st.image(shap_chart, use_container_width=True)
        st.caption(
            "Red bars = factors pushing this customer toward churn. "
            "Green bars = factors keeping them loyal. "
            "Longer bar = stronger impact on the prediction."
        )

        st.divider()

        # Top risk factors side by side
        st.subheader("Top contributing factors")

        risk_inc = shap_exp['risk_increasers']
        risk_dec = shap_exp['risk_decreasers']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Factors increasing churn risk:**")
            if len(risk_inc) > 0:
                for feat, val in risk_inc.head(3).items():
                    clean = (feat.replace('_', ' ')
                                 .replace('InternetService', 'Internet:')
                                 .replace('Contract', 'Contract:')
                                 .replace('PaymentMethod', 'Payment:'))[:35]
                    st.error(f"↑ {clean}  (+{val:.3f})")
            else:
                st.info("No strong risk-increasing factors found.")

        with col2:
            st.markdown("**Factors reducing churn risk:**")
            if len(risk_dec) > 0:
                for feat, val in risk_dec.head(3).items():
                    clean = (feat.replace('_', ' ')
                                 .replace('InternetService', 'Internet:')
                                 .replace('Contract', 'Contract:')
                                 .replace('PaymentMethod', 'Payment:'))[:35]
                    st.success(f"↓ {clean}  ({val:.3f})")
            else:
                st.info("No strong risk-reducing factors found.")

        st.divider()

    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")
        shap_loaded = False

    # ================================================
    # RECOMMENDATIONS
    # ================================================
    st.subheader("Smart Retention Recommendations")

    # Use SHAP recommendations if available
    # Otherwise fall back to rule-based recommendations
    if shap_loaded and shap_recs:
        final_recs = shap_recs
        st.write(
            "These recommendations are generated directly from "
            "what the ML model learned — not manual rules."
        )
    elif modules_loaded:
        final_recs = get_recommendations(customer_data, prob, pred)
        st.write("Recommendations based on customer profile analysis.")
    else:
        final_recs = ["Please check that all helper files exist."]

    if pred == 1:
        st.warning(
            f"Risk Level: {risk} — "
            f"{len(final_recs)} action(s) recommended"
        )
    else:
        st.info(f"Risk Level: {risk} — Routine monitoring advised")

    for i, rec in enumerate(final_recs, 1):
        if pred == 1:
            st.error(f"Action {i}: {rec}")
        else:
            st.success(f"Note {i}: {rec}")

    st.divider()

    # ================================================
    # PDF DOWNLOAD
    # ================================================
    st.subheader("Download Report")
    st.write(
        "Download a professional PDF report of this prediction "
        "including customer details, risk assessment, "
        "and all recommendations."
    )

    if modules_loaded:
        try:
            pdf_bytes = generate_churn_report(
                customer_data, prob, pred, final_recs
            )
            filename = (
                f"churn_report_"
                f"{datetime.date.today().strftime('%Y%m%d')}_"
                f"{tenure}mo_{contract.replace(' ', '-')}.pdf"
            )
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True
            )
            st.caption(
                "The PDF includes customer details, churn probability, "
                "risk assessment, and all recommendations."
            )
        except Exception as e:
            st.error(f"PDF generation error: {e}")
    else:
        st.info(
            "PDF generation unavailable. "
            "Check that pdf_report.py exists."
        )