# pages/2_Predict_Single.py — Multi-industry prediction FULLY FIXED

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

config = INDUSTRIES[selected]

try:
    model         = joblib.load(config['model_file'])
    scaler        = joblib.load(config['scaler_file'])
    feature_names = joblib.load(config['features_file'])
    st.success(f"Model loaded: {industry_options[selected]}")
except Exception as e:
    st.error(f"Model not found. Run multi_train.py first. Error: {e}")
    st.stop()

st.divider()

customer_data = {}

# ================================================
# TELECOM FORM
# ================================================
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
        'tenure'         : f"{tenure} months",
        'monthly_charges': f"${monthly}",
        'contract'       : contract,
        'internet'       : internet,
        'payment'        : payment,
        'senior'         : senior,
        'partner'        : partner,
        'tech_support'   : tech_supp,
        'paperless'      : paperless,
    }

    def build_input():
        raw = {
            'gender'          : 1.0 if gender == "Male" else 0.0,
            'SeniorCitizen'   : 1.0 if senior == "Yes" else 0.0,
            'Partner'         : 1.0 if partner == "Yes" else 0.0,
            'Dependents'      : 1.0 if dependents == "Yes" else 0.0,
            'tenure'          : float(tenure),
            'PhoneService'    : 1.0 if phone == "Yes" else 0.0,
            'MultipleLines'   : float({
                'No phone service': 0,
                'No': 1, 'Yes': 2
            }[multi_lines]),
            'PaperlessBilling': 1.0 if paperless == "Yes" else 0.0,
            'MonthlyCharges'  : float(monthly),
            'TotalCharges'    : float(total),
            'InternetService_Fiber optic':
                1.0 if internet == "Fiber optic" else 0.0,
            'InternetService_No':
                1.0 if internet == "No" else 0.0,
            'OnlineSecurity_No internet service':
                1.0 if internet == "No" else 0.0,
            'OnlineSecurity_Yes': 0.0,
            'OnlineBackup_No internet service':
                1.0 if internet == "No" else 0.0,
            'OnlineBackup_Yes': 0.0,
            'DeviceProtection_No internet service':
                1.0 if internet == "No" else 0.0,
            'DeviceProtection_Yes': 0.0,
            'TechSupport_No internet service':
                1.0 if tech_supp == "No internet service" else 0.0,
            'TechSupport_Yes':
                1.0 if tech_supp == "Yes" else 0.0,
            'StreamingTV_No internet service':
                1.0 if internet == "No" else 0.0,
            'StreamingTV_Yes': 0.0,
            'StreamingMovies_No internet service':
                1.0 if internet == "No" else 0.0,
            'StreamingMovies_Yes': 0.0,
            'Contract_One year':
                1.0 if contract == "One year" else 0.0,
            'Contract_Two year':
                1.0 if contract == "Two year" else 0.0,
            'PaymentMethod_Credit card (automatic)':
                1.0 if payment == "Credit card (automatic)" else 0.0,
            'PaymentMethod_Electronic check':
                1.0 if payment == "Electronic check" else 0.0,
            'PaymentMethod_Mailed check':
                1.0 if payment == "Mailed check" else 0.0,
            'total_services': 0.0,
        }
        row_vals  = [raw.get(col, 0.0) for col in feature_names]
        row_array = np.array(row_vals, dtype=float).reshape(1, -1)
        scaled    = scaler.transform(row_array)
        return pd.DataFrame(scaled, columns=feature_names)

# ================================================
# BANKING FORM
# ================================================
elif selected == 'banking':
    st.subheader("🏦 Bank Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        geography    = st.selectbox("Country",
                                    ["France", "Germany", "Spain"])
        gender_b     = st.selectbox("Gender", ["Male", "Female"])
        age          = st.slider("Age", 18, 92, 35)
        tenure_b     = st.slider("Tenure (years)", 0, 10, 3)

    with col2:
        balance      = st.number_input("Account Balance ($)",
                                       0.0, 300000.0, 50000.0, 1000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card  = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active    = st.selectbox("Active Member?", ["Yes", "No"])
        salary       = st.number_input("Estimated Salary ($)",
                                       0.0, 300000.0, 60000.0, 1000.0)

    customer_data = {
        'tenure'         : f"{tenure_b} years",
        'monthly_charges': f"${salary/12:.0f}/month",
        'contract'       : f"{num_products} product(s)",
        'internet'       : geography,
        'payment'        : gender_b,
        'senior'         : "Yes" if age >= 60 else "No",
        'partner'        : f"Balance ${balance:,.0f}",
        'tech_support'   : "Active" if is_active == "Yes" else "Inactive",
        'paperless'      : "Yes" if has_cr_card == "Yes" else "No",
    }

    def build_input():
        raw = {
            'CreditScore'      : float(credit_score),
            'Age'              : float(age),
            'Tenure'           : float(tenure_b),
            'Balance'          : float(balance),
            'NumOfProducts'    : float(num_products),
            'HasCrCard'        : 1.0 if has_cr_card == "Yes" else 0.0,
            'IsActiveMember'   : 1.0 if is_active == "Yes" else 0.0,
            'EstimatedSalary'  : float(salary),
            'Geography_Germany': 1.0 if geography == "Germany" else 0.0,
            'Geography_Spain'  : 1.0 if geography == "Spain" else 0.0,
            'Gender_Male'      : 1.0 if gender_b == "Male" else 0.0,
        }
        row_vals  = [raw.get(col, 0.0) for col in feature_names]
        row_array = np.array(row_vals, dtype=float).reshape(1, -1)
        scaled    = scaler.transform(row_array)
        return pd.DataFrame(scaled, columns=feature_names)

# ================================================
# ECOMMERCE FORM
# ================================================
elif selected == 'ecommerce':
    st.subheader("🛒 E-Commerce Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        tenure_e     = st.slider("Tenure (months)", 0, 36, 6)
        login_device = st.selectbox("Preferred Login Device",
                                    ["Mobile Phone", "Computer", "Phone"])
        city_tier    = st.selectbox("City Tier", [1, 2, 3])
        warehouse    = st.slider("Warehouse to Home (km)", 5, 100, 20)
        payment_mode = st.selectbox("Preferred Payment",
                                    ["Debit Card", "UPI", "Credit Card",
                                     "Cash on Delivery", "E wallet", "CC"])
        gender_e     = st.selectbox("Gender", ["Male", "Female"])
        hours_app    = st.slider("Hours on App per Day", 0, 6, 3)

    with col2:
        num_devices  = st.slider("Devices Registered", 1, 6, 3)
        order_cat    = st.selectbox("Preferred Order Category",
                                    ["Laptop & Accessory", "Mobile",
                                     "Mobile Phone", "Others",
                                     "Fashion", "Grocery"])
        satisfaction = st.selectbox("Satisfaction Score (1=low 5=high)",
                                    [1, 2, 3, 4, 5], index=2)
        marital      = st.selectbox("Marital Status",
                                    ["Single", "Married", "Divorced"])
        num_address  = st.slider("Number of Addresses", 1, 10, 3)
        complain     = st.selectbox("Has Complaint?", ["No", "Yes"])
        cashback     = st.slider("Cashback Amount ($)", 0, 300, 150)

    customer_data = {
        'tenure'         : f"{tenure_e} months",
        'monthly_charges': f"Cashback ${cashback}",
        'contract'       : login_device,
        'internet'       : order_cat,
        'payment'        : payment_mode,
        'senior'         : "No",
        'partner'        : marital,
        'tech_support'   : "Yes" if complain == "No" else "No",
        'paperless'      : "Yes",
    }

    def build_input():
        raw = {
            'Tenure'                      : float(tenure_e),
            'CityTier'                    : float(city_tier),
            'WarehouseToHome'             : float(warehouse),
            'HourSpendOnApp'              : float(hours_app),
            'NumberOfDeviceRegistered'    : float(num_devices),
            'SatisfactionScore'           : float(satisfaction),
            'NumberOfAddress'             : float(num_address),
            'Complain'                    : 1.0 if complain == "Yes" else 0.0,
            'OrderAmountHikeFromlastYear' : 15.0,
            'CouponUsed'                  : 2.0,
            'OrderCount'                  : 3.0,
            'DaySinceLastOrder'           : 5.0,
            'CashbackAmount'              : float(cashback),
            'PreferredLoginDevice_Mobile Phone':
                1.0 if login_device == "Mobile Phone" else 0.0,
            'PreferredLoginDevice_Phone'  :
                1.0 if login_device == "Phone" else 0.0,
            'PreferredPaymentMode_Credit Card':
                1.0 if payment_mode == "Credit Card" else 0.0,
            'PreferredPaymentMode_Debit Card':
                1.0 if payment_mode == "Debit Card" else 0.0,
            'PreferredPaymentMode_E wallet':
                1.0 if payment_mode == "E wallet" else 0.0,
            'PreferredPaymentMode_UPI'    :
                1.0 if payment_mode == "UPI" else 0.0,
            'PreferredPaymentMode_Cash on Delivery':
                1.0 if payment_mode == "Cash on Delivery" else 0.0,
            'Gender_Male'                 :
                1.0 if gender_e == "Male" else 0.0,
            'PreferedOrderCat_Grocery'    :
                1.0 if order_cat == "Grocery" else 0.0,
            'PreferedOrderCat_Laptop & Accessory':
                1.0 if order_cat == "Laptop & Accessory" else 0.0,
            'PreferedOrderCat_Mobile'     :
                1.0 if order_cat == "Mobile" else 0.0,
            'PreferedOrderCat_Mobile Phone':
                1.0 if order_cat == "Mobile Phone" else 0.0,
            'PreferedOrderCat_Others'     :
                1.0 if order_cat == "Others" else 0.0,
            'MaritalStatus_Married'       :
                1.0 if marital == "Married" else 0.0,
            'MaritalStatus_Single'        :
                1.0 if marital == "Single" else 0.0,
        }
        row_vals  = [raw.get(col, 0.0) for col in feature_names]
        row_array = np.array(row_vals, dtype=float).reshape(1, -1)
        scaled    = scaler.transform(row_array)
        return pd.DataFrame(scaled, columns=feature_names)

# ================================================
# PREDICT BUTTON
# ================================================
st.divider()

if st.button("Predict Churn Risk", use_container_width=True):
    try:
        input_row = build_input()
        prob      = float(model.predict_proba(input_row)[0][1])
        pred      = int(model.predict(input_row)[0])

        risk = get_risk_label(prob) if modules_loaded else (
            "High Risk"   if prob > 0.65 else
            "Medium Risk" if prob > 0.40 else "Low Risk"
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
                st.warning(f"Could not save: {e}")

        st.divider()

        # Results
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Churn Probability", f"{prob:.0%}")
        c2.metric("Industry",          industry_options[selected])
        c3.metric("Risk Level",        risk)
        c4.metric("Prediction",
                  "WILL CHURN" if pred == 1 else "WILL STAY")

        st.progress(float(prob))

        if pred == 1:
            st.error(
                f"This customer is likely to CHURN — Risk: {prob:.0%}"
            )
        else:
            st.success(
                f"This customer is likely to STAY — Risk: {prob:.0%}"
            )

        st.divider()

        # ================================================
        # SHAP — TELECOM ONLY
        # ================================================
        final_recs = []

        if selected == 'telecom':
            try:
                import shap
                import io
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from matplotlib.patches import Patch

                shap_explainer_obj = shap.TreeExplainer(model)
                sv                 = shap_explainer_obj.shap_values(
                    input_row
                )
                sv_series = pd.Series(sv[0], index=feature_names)
                sorted_sv = sv_series.reindex(
                    sv_series.abs().sort_values(ascending=False).index
                )
                risk_inc  = sorted_sv[sorted_sv > 0].head(5)
                risk_dec  = sorted_sv[sorted_sv < 0].head(5)

                st.subheader("Why did the model predict this?")
                top8        = sorted_sv.head(8)
                clean_names = [
                    n.replace('_', ' ')[:30] for n in top8.index
                ]
                vals   = top8.values
                colors = [
                    '#E24B4A' if v > 0 else '#1A7A4A' for v in vals
                ]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(range(len(clean_names)), vals,
                        color=colors, alpha=0.85, height=0.6)
                ax.set_yticks(range(len(clean_names)))
                ax.set_yticklabels(clean_names, fontsize=10)
                ax.axvline(x=0, color='gray', linewidth=0.8)
                ax.set_xlabel('SHAP value', fontsize=10)
                ax.set_title('Why did the model predict this?',
                             fontsize=12, fontweight='bold')
                legend = [
                    Patch(color='#E24B4A', alpha=0.85,
                          label='Increases churn risk'),
                    Patch(color='#1A7A4A', alpha=0.85,
                          label='Decreases churn risk')
                ]
                ax.legend(handles=legend,
                          loc='lower right', fontsize=9)
                ax.invert_yaxis()
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=130,
                            bbox_inches='tight')
                buf.seek(0)
                plt.close()
                st.image(buf.getvalue(), use_container_width=True)
                st.caption(
                    "Red = increases churn | Green = decreases churn"
                )

                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Factors increasing risk:**")
                    for feat, val in risk_inc.head(3).items():
                        st.error(
                            f"↑ {feat.replace('_',' ')[:30]}"
                            f"  (+{val:.3f})"
                        )
                with col2:
                    st.markdown("**Factors reducing risk:**")
                    for feat, val in risk_dec.head(3).items():
                        st.success(
                            f"↓ {feat.replace('_',' ')[:30]}"
                            f"  ({val:.3f})"
                        )

                # SHAP-based recommendations
                for feat, val in risk_inc.items():
                    pct = round(abs(val) * 100, 1)
                    if 'Contract' in feat:
                        final_recs.append(
                            f"Contract type is a top churn driver "
                            f"(+{pct}% risk). "
                            "Offer upgrade with 20% discount."
                        )
                    elif 'tenure' in feat.lower():
                        final_recs.append(
                            f"Short tenure increases risk (+{pct}%). "
                            "Assign onboarding support agent."
                        )
                    elif 'MonthlyCharges' in feat:
                        final_recs.append(
                            f"High charges driving churn (+{pct}%). "
                            "Offer 10-15% bill reduction bundle."
                        )
                    elif 'Fiber' in feat:
                        final_recs.append(
                            f"Fiber optic is a risk factor (+{pct}%). "
                            "Check service quality proactively."
                        )
                    elif 'Electronic check' in feat:
                        final_recs.append(
                            f"Electronic check increases risk (+{pct}%). "
                            "Suggest switching to auto-payment."
                        )

            except Exception as shap_e:
                st.info(f"SHAP chart unavailable: {shap_e}")
                if modules_loaded:
                    final_recs = get_recommendations(
                        customer_data, prob, pred
                    )

        # ================================================
        # BANKING RECOMMENDATIONS
        # ================================================
        elif selected == 'banking':
            if pred == 1:
                if num_products >= 3:
                    final_recs.append(
                        "Customer has 3+ products which increases churn. "
                        "Review product fit and simplify offering."
                    )
                if age > 45:
                    final_recs.append(
                        "Older customer needs dedicated relationship manager. "
                        "Schedule personal review call."
                    )
                if is_active == "No":
                    final_recs.append(
                        "Inactive member — send re-engagement campaign "
                        "with exclusive offer."
                    )
                if geography == "Germany":
                    final_recs.append(
                        "Germany has highest churn rate. "
                        "Offer region-specific loyalty program."
                    )
                if balance == 0:
                    final_recs.append(
                        "Zero balance account — customer may already "
                        "be disengaging. Offer incentive to deposit."
                    )
                if not final_recs:
                    final_recs.append(
                        "Schedule personal customer review and "
                        "offer loyalty package."
                    )
            else:
                final_recs.append(
                    "Customer shows low churn risk. "
                    "Maintain current service quality."
                )
                final_recs.append(
                    "Consider upsell — satisfied customer may "
                    "welcome premium product offer."
                )

        # ================================================
        # ECOMMERCE RECOMMENDATIONS
        # ================================================
        elif selected == 'ecommerce':
            if pred == 1:
                if complain == "Yes":
                    final_recs.append(
                        "Customer has filed complaint — resolve immediately "
                        "and offer compensation coupon."
                    )
                if satisfaction <= 2:
                    final_recs.append(
                        f"Low satisfaction score ({satisfaction}/5). "
                        "Send personal apology and discount voucher."
                    )
                if tenure_e < 6:
                    final_recs.append(
                        "New customer (under 6 months). "
                        "Assign dedicated support and welcome rewards."
                    )
                if hours_app < 2:
                    final_recs.append(
                        "Low app engagement. Send push notification "
                        "with personalized product recommendations."
                    )
                if not final_recs:
                    final_recs.append(
                        "Send personalized retention offer with "
                        "cashback increase and free shipping."
                    )
            else:
                final_recs.append(
                    "Customer is satisfied. Maintain service level "
                    "and consider loyalty rewards."
                )
                if satisfaction >= 4:
                    final_recs.append(
                        "High satisfaction customer — ask for a review "
                        "and refer-a-friend program."
                    )

        # ================================================
        # SHOW RECOMMENDATIONS
        # ================================================
        st.divider()
        st.subheader("Smart Recommendations")

        if not final_recs and modules_loaded:
            final_recs = get_recommendations(
                customer_data, prob, pred
            )

        if pred == 1:
            st.warning(
                f"{risk} — {len(final_recs)} action(s) recommended"
            )
            for i, rec in enumerate(final_recs, 1):
                st.error(f"Action {i}: {rec}")
        else:
            st.info(f"{risk} — Routine monitoring advised")
            for i, rec in enumerate(final_recs, 1):
                st.success(f"Note {i}: {rec}")

        st.divider()

        # ================================================
        # PDF DOWNLOAD
        # ================================================
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