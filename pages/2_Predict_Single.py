# pages/2_Predict_Single.py — Multi-industry prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os

st.set_page_config(page_title="Predict Single", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

# Load helpers
try:
    from recommendations import get_recommendations, get_risk_label
    from pdf_report       import generate_churn_report
    from database         import save_prediction
    from industry_config  import INDUSTRIES
    modules_loaded = True
except Exception as e:
    st.warning(f"Helper modules not loaded: {e}")
    modules_loaded = False

st.title("Predict Single Customer")
st.write("Select your industry, enter customer details, and get instant churn prediction.")
st.divider()

# ================================================
# INDUSTRY SELECTOR
# ================================================
industry_options = {
    'telecom'  : '📡 Telecom',
    'banking'  : '🏦 Banking',
    'ecommerce': '🛒 E-Commerce'
}

selected = st.selectbox(
    "Select Industry",
    options=list(industry_options.keys()),
    format_func=lambda x: industry_options[x],
    index=0
)

st.session_state.selected_industry = selected

# Load model for selected industry
config = INDUSTRIES[selected]

try:
    model         = joblib.load(config['model_file'])
    scaler        = joblib.load(config['scaler_file'])
    feature_names = joblib.load(config['features_file'])
    st.success(f"Model loaded: {industry_options[selected]}")
except Exception as e:
    st.error(f"Model not found for {selected}. Run multi_train.py first. Error: {e}")
    st.stop()

st.divider()

# ================================================
# INPUT FORMS — DIFFERENT PER INDUSTRY
# ================================================

customer_data = {}

# ---- TELECOM FORM ----
if selected == 'telecom':
    st.subheader("📡 Telecom Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        gender      = st.selectbox("Gender", ["Male", "Female"])
        senior      = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner     = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents  = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure      = st.slider("Tenure (months)", 0, 72, 12)
        phone       = st.selectbox("Phone Service?", ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines?",
                                   ["No", "Yes", "No phone service"])

    with col2:
        internet  = st.selectbox("Internet Service",
                                 ["DSL", "Fiber optic", "No"])
        tech_supp = st.selectbox("Tech Support",
                                 ["No", "Yes", "No internet service"])
        contract  = st.selectbox("Contract Type",
                                 ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing?", ["Yes", "No"])
        payment   = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.slider("Monthly Charges ($)", 18, 120, 65)
        total   = monthly * tenure

    customer_data = {
        'tenure': f"{tenure} months", 'monthly_charges': f"${monthly}",
        'contract': contract, 'internet': internet,
        'payment': payment, 'senior': senior,
        'partner': partner, 'tech_support': tech_supp,
        'paperless': paperless,
    }

    def build_telecom():
        row = pd.DataFrame(
            [np.zeros(len(feature_names))], columns=feature_names
        )
        num_scaled = scaler.transform([[tenure, monthly, total]])[0]
        for col, val in zip(
            ['tenure', 'MonthlyCharges', 'TotalCharges'], num_scaled
        ):
            if col in row.columns:
                row[col] = val

        binary = {
            'gender'          : 1 if gender == "Male" else 0,
            'SeniorCitizen'   : 1 if senior == "Yes" else 0,
            'Partner'         : 1 if partner == "Yes" else 0,
            'Dependents'      : 1 if dependents == "Yes" else 0,
            'PhoneService'    : 1 if phone == "Yes" else 0,
            'PaperlessBilling': 1 if paperless == "Yes" else 0,
            'MultipleLines'   : {
                'No phone service': 0, 'No': 1, 'Yes': 2
            }[multi_lines],
        }
        for col, val in binary.items():
            if col in row.columns:
                row[col] = val

        onehot = {
            'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
            'InternetService_No'         : 1 if internet == "No" else 0,
            'Contract_One year'          : 1 if contract == "One year" else 0,
            'Contract_Two year'          : 1 if contract == "Two year" else 0,
            'PaymentMethod_Credit card (automatic)':
                1 if payment == "Credit card (automatic)" else 0,
            'PaymentMethod_Electronic check':
                1 if payment == "Electronic check" else 0,
            'PaymentMethod_Mailed check':
                1 if payment == "Mailed check" else 0,
        }
        for col, val in onehot.items():
            if col in row.columns:
                row[col] = val
        return row

    build_input = build_telecom

# ---- BANKING FORM ----
elif selected == 'banking':
    st.subheader("🏦 Bank Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        credit_score  = st.slider("Credit Score", 300, 900, 650)
        geography     = st.selectbox("Country",
                                     ["France", "Germany", "Spain"])
        gender        = st.selectbox("Gender", ["Male", "Female"])
        age           = st.slider("Age", 18, 92, 35)
        tenure        = st.slider("Tenure (years)", 0, 10, 3)

    with col2:
        balance       = st.number_input("Account Balance ($)",
                                        min_value=0.0,
                                        max_value=300000.0,
                                        value=50000.0,
                                        step=1000.0)
        num_products  = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card   = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active     = st.selectbox("Active Member?", ["Yes", "No"])
        salary        = st.number_input("Estimated Salary ($)",
                                        min_value=0.0,
                                        max_value=300000.0,
                                        value=60000.0,
                                        step=1000.0)

    customer_data = {
        'tenure'        : f"{tenure} years",
        'monthly_charges': f"${salary/12:.0f}",
        'contract'      : f"{num_products} products",
        'internet'      : geography,
        'payment'       : f"Age {age}",
        'senior'        : "Yes" if age >= 60 else "No",
        'partner'       : f"Balance ${balance:,.0f}",
    }

    def build_banking():
        row = pd.DataFrame(
            [np.zeros(len(feature_names))], columns=feature_names
        )
        raw_vals = {
            'CreditScore'    : credit_score,
            'Age'            : age,
            'Tenure'         : tenure,
            'Balance'        : balance,
            'NumOfProducts'  : num_products,
            'HasCrCard'      : 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember' : 1 if is_active == "Yes" else 0,
            'EstimatedSalary': salary,
        }
        raw_df     = pd.DataFrame([raw_vals])
        geo_map    = {
            'France' : [0, 0],
            'Germany': [1, 0],
            'Spain'  : [0, 1]
        }
        geo_vals   = geo_map.get(geography, [0, 0])
        raw_df['Geography_Germany'] = geo_vals[0]
        raw_df['Geography_Spain']   = geo_vals[1]
        raw_df['Gender_Male'] = 1 if gender == "Male" else 0

        # Scale
        all_cols   = raw_df.columns.tolist()
        scaled     = scaler.transform(raw_df[
            [c for c in all_cols if c in feature_names]
        ])
        scaled_df  = pd.DataFrame(
            scaled,
            columns=[c for c in all_cols if c in feature_names]
        )
        for col in feature_names:
            if col in scaled_df.columns:
                row[col] = scaled_df[col].values[0]
        return row

    build_input = build_banking

# ---- ECOMMERCE FORM ----
elif selected == 'ecommerce':
    st.subheader("🛒 E-Commerce Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        tenure_ec    = st.slider("Tenure (months)", 0, 36, 6)
        login_device = st.selectbox("Preferred Login Device",
                                    ["Mobile Phone", "Computer", "Phone"])
        city_tier    = st.selectbox("City Tier", [1, 2, 3])
        warehouse    = st.slider("Warehouse to Home (km)", 5, 100, 20)
        payment_mode = st.selectbox("Preferred Payment",
                                    ["Debit Card", "UPI", "Credit Card",
                                     "Cash on Delivery", "E wallet", "CC"])
        gender_ec    = st.selectbox("Gender", ["Male", "Female"])
        hours_app    = st.slider("Hours Spent on App", 0, 6, 3)

    with col2:
        num_devices  = st.slider("Number of Devices Registered", 1, 6, 3)
        order_cat    = st.selectbox("Preferred Order Category",
                                    ["Laptop & Accessory", "Mobile",
                                     "Mobile Phone", "Others",
                                     "Fashion", "Grocery"])
        satisfaction = st.selectbox("Satisfaction Score (1-5)",
                                    [1, 2, 3, 4, 5], index=2)
        marital      = st.selectbox("Marital Status",
                                    ["Single", "Married", "Divorced"])
        num_address  = st.slider("Number of Addresses", 1, 10, 3)
        complain     = st.selectbox("Has Complaint?", ["No", "Yes"])
        cashback     = st.slider("Cashback Amount ($)", 0, 300, 150)

    customer_data = {
        'tenure'         : f"{tenure_ec} months",
        'monthly_charges': f"Satisfaction {satisfaction}/5",
        'contract'       : login_device,
        'internet'       : order_cat,
        'payment'        : payment_mode,
        'senior'         : "No",
        'partner'        : marital,
    }

    def build_ecommerce():
        row = pd.DataFrame(
            [np.zeros(len(feature_names))], columns=feature_names
        )
        raw_vals = {
            'Tenure'                      : tenure_ec,
            'CityTier'                    : city_tier,
            'WarehouseToHome'             : warehouse,
            'HourSpendOnApp'              : hours_app,
            'NumberOfDeviceRegistered'    : num_devices,
            'SatisfactionScore'           : satisfaction,
            'NumberOfAddress'             : num_address,
            'Complain'                    : 1 if complain == "Yes" else 0,
            'OrderAmountHikeFromlastYear' : 15,
            'CouponUsed'                  : 2,
            'OrderCount'                  : 3,
            'DaySinceLastOrder'           : 5,
            'CashbackAmount'              : cashback,
        }
        raw_df = pd.DataFrame([raw_vals])

        # One-hot for categorical
        device_cols = {
            'PreferredLoginDevice_Mobile Phone':
                1 if login_device == "Mobile Phone" else 0,
            'PreferredLoginDevice_Phone':
                1 if login_device == "Phone" else 0,
        }
        pay_cols = {
            'PreferredPaymentMode_Credit Card':
                1 if payment_mode == "Credit Card" else 0,
            'PreferredPaymentMode_Debit Card':
                1 if payment_mode == "Debit Card" else 0,
            'PreferredPaymentMode_E wallet':
                1 if payment_mode == "E wallet" else 0,
            'PreferredPaymentMode_UPI':
                1 if payment_mode == "UPI" else 0,
            'PreferredPaymentMode_Cash on Delivery':
                1 if payment_mode == "Cash on Delivery" else 0,
        }
        gender_cols = {
            'Gender_Male': 1 if gender_ec == "Male" else 0,
        }
        order_cols = {
            'PreferedOrderCat_Grocery':
                1 if order_cat == "Grocery" else 0,
            'PreferedOrderCat_Laptop & Accessory':
                1 if order_cat == "Laptop & Accessory" else 0,
            'PreferedOrderCat_Mobile':
                1 if order_cat == "Mobile" else 0,
            'PreferedOrderCat_Mobile Phone':
                1 if order_cat == "Mobile Phone" else 0,
            'PreferedOrderCat_Others':
                1 if order_cat == "Others" else 0,
        }
        marital_cols = {
            'MaritalStatus_Married':
                1 if marital == "Married" else 0,
            'MaritalStatus_Single':
                1 if marital == "Single" else 0,
        }

        all_extra = {
            **device_cols, **pay_cols, **gender_cols,
            **order_cols,  **marital_cols
        }
        for k, v in all_extra.items():
            raw_df[k] = v

        # Scale numeric columns
        num_cols  = list(raw_vals.keys())
        scale_df  = raw_df[num_cols].copy()
        # Build full feature row
        for col in feature_names:
            if col in raw_df.columns:
                row[col] = raw_df[col].values[0]
        # Apply scaler to numeric only
        try:
            num_feat = [c for c in num_cols if c in feature_names]
            if num_feat:
                temp = pd.DataFrame(
                    [np.zeros(len(feature_names))],
                    columns=feature_names
                )
                for c in num_feat:
                    temp[c] = raw_df[c].values[0]
                scaled = scaler.transform(temp)
                row    = pd.DataFrame(scaled, columns=feature_names)
                for col in feature_names:
                    if col in all_extra:
                        row[col] = all_extra[col]
        except Exception:
            pass
        return row

    build_input = build_ecommerce

# ================================================
# PREDICT BUTTON
# ================================================
st.divider()

if st.button("Predict Churn Risk", use_container_width=True):
    try:
        input_row = build_input()
        prob      = model.predict_proba(input_row)[0][1]
        pred      = int(model.predict(input_row)[0])

        risk = get_risk_label(prob) if modules_loaded else (
            "High Risk" if prob > 0.65 else
            "Medium Risk" if prob > 0.4 else "Low Risk"
        )

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

        # Results
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Churn Probability", f"{prob:.0%}")
        c2.metric("Industry",
                  industry_options[selected])
        c3.metric("Risk Level", risk)
        c4.metric("Prediction",
                  "WILL CHURN" if pred == 1 else "WILL STAY")

        st.progress(float(prob))

        if pred == 1:
            st.error(
                f"This customer is likely to CHURN — Risk: {prob:.0%}"
            )
        else:
            st.success(
                f"This customer is likely to STAY — Churn risk: {prob:.0%}"
            )

        st.divider()

        # SHAP + Recommendations
        try:
            from shap_explainer import (get_shap_explanation,
                                        get_shap_recommendations,
                                        plot_shap_bar)
            with st.spinner("Calculating SHAP explanation..."):
                shap_exp   = get_shap_explanation(input_row)
                shap_recs  = get_shap_recommendations(shap_exp, pred)
                shap_chart = plot_shap_bar(shap_exp)

            st.subheader("Why did the model predict this?")
            st.image(shap_chart, use_container_width=True)
            st.caption(
                "Red = increases churn risk | "
                "Green = decreases churn risk"
            )
            final_recs = shap_recs
        except Exception:
            if modules_loaded:
                final_recs = get_recommendations(
                    customer_data, prob, pred
                )
            else:
                final_recs = ["Check that all helper files exist."]

        st.divider()
        st.subheader("Smart Recommendations")

        if pred == 1:
            st.warning(f"{risk} — {len(final_recs)} actions recommended")
            for i, rec in enumerate(final_recs, 1):
                st.error(f"Action {i}: {rec}")
        else:
            st.info(f"{risk} — Routine monitoring advised")
            for i, rec in enumerate(final_recs, 1):
                st.success(f"Note {i}: {rec}")

        st.divider()

        # PDF Download
        if modules_loaded:
            try:
                pdf_bytes = generate_churn_report(
                    customer_data, prob, pred, final_recs
                )
                filename = (
                    f"churnguard_{selected}_"
                    f"{datetime.date.today().strftime('%Y%m%d')}.pdf"
                )
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF error: {e}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.code(traceback.format_exc())