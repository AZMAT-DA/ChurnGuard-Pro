# app.py — ChurnGuard Pro — Login and Landing Page

import streamlit as st
from database import initialize_database, login_user, register_user

initialize_database()

st.set_page_config(
    page_title="ChurnGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    .hero-title {
        font-size: 3rem; font-weight: 700;
        color: #1E3A5F; margin-bottom: 0.5rem; line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.2rem; color: #4A5568;
        margin-bottom: 2rem; line-height: 1.6;
    }
    .hero-badge {
        display: inline-block; background: #E8F4FD;
        color: #1E3A5F; padding: 4px 12px;
        border-radius: 20px; font-size: 0.85rem;
        font-weight: 600; margin-bottom: 1rem;
        border: 1px solid #BDD7EE;
    }
    .feature-card {
        background: white; border: 1px solid #E2E8F0;
        border-radius: 12px; padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .feature-title {
        font-size: 1.1rem; font-weight: 600;
        color: #1E3A5F; margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-size: 0.9rem; color: #4A5568; line-height: 1.5;
    }
    .stat-box {
        text-align: center; padding: 1.5rem;
        background: #F7FAFC; border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
    .stat-number { font-size: 2rem; font-weight: 700; color: #1E3A5F; }
    .stat-label  { font-size: 0.85rem; color: #718096; margin-top: 4px; }
    .welcome-banner {
        background: linear-gradient(135deg, #1E3A5F 0%, #2D5986 100%);
        color: white; padding: 2rem; border-radius: 16px;
        margin-bottom: 2rem;
    }
    .welcome-name { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; }
    .welcome-sub  { font-size: 1rem; opacity: 0.85; }
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #1E3A5F, #4299E1, transparent);
        border-radius: 2px; margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

for key, val in {
    'logged_in' : False,
    'user_id'   : None,
    'username'  : '',
    'full_name' : '',
    'role'      : ''
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ================================================
# LOGGED IN — LANDING PAGE
# ================================================
if st.session_state.logged_in:

    st.markdown(f"""
    <div class="welcome-banner">
        <div class="welcome-name">
            Welcome back, {st.session_state.full_name}!
        </div>
        <div class="welcome-sub">
            ChurnGuard Pro — AI-Powered Customer Retention Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown("""
        <div class="hero-badge">
            Final Year Project — Multi-Industry ML Platform
        </div>
        <div class="hero-title">ChurnGuard Pro</div>
        <div class="hero-subtitle">
            An intelligent AI platform that predicts customer churn
            across Telecom, Banking, and E-Commerce. Built with
            XGBoost, Neural Networks, SHAP explainability,
            GridSearchCV tuning, and cross-validated evaluation.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="stat-box"><div class="stat-number">3</div>'
            '<div class="stat-label">Industries supported</div></div>',
            unsafe_allow_html=True
        )
        st.markdown("")
        st.markdown(
            '<div class="stat-box"><div class="stat-number">84%</div>'
            '<div class="stat-label">Best AUC score</div></div>',
            unsafe_allow_html=True
        )
        st.markdown("")
        st.markdown(
            '<div class="stat-box"><div class="stat-number">4</div>'
            '<div class="stat-label">ML models trained</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Navigate to a feature")

    # ---- Row 1 ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 Analytics Dashboard</div>
            <div class="feature-desc">
                Real-time churn statistics and charts for all 3
                industries. Telecom, Banking, and E-Commerce tabs
                with industry-specific visualizations.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/1_Dashboard.py", label="Open Dashboard →")

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🔍 Predict Single Customer</div>
            <div class="feature-desc">
                Enter customer details and get instant churn prediction
                with SHAP explanation, smart recommendations,
                and a downloadable PDF report.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/2_Predict_Single.py", label="Make Prediction →")

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📁 Bulk Upload</div>
            <div class="feature-desc">
                Upload a CSV file with hundreds of customers
                and get churn predictions for all of them
                at once with downloadable results.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/3_Bulk_Upload.py", label="Upload CSV →")

    # ---- Row 2 ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🤖 Model Report</div>
            <div class="feature-desc">
                Compare all 4 models across 3 industries.
                See GridSearchCV best parameters and
                5-fold cross-validation scores.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/4_Model_Report.py", label="View Models →")

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📋 My History</div>
            <div class="feature-desc">
                View all your predictions, filter by risk level,
                download history as CSV, and track your
                prediction activity over time.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/5_History.py", label="View History →")

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">💰 Revenue Impact</div>
            <div class="feature-desc">
                Calculate exactly how much churning customers
                cost your business per month, per year, and
                over their lifetime. Get instant ROI calculation.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/8_Revenue.py", label="Calculate Revenue →")

    # ---- Row 3 ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎯 Customer Segments</div>
            <div class="feature-desc">
                Group customers into segments like High Value
                at Risk, Loyal Champions, and New Customers
                using K-Means clustering.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/9_Segments.py", label="View Segments →")

    with col2:
        if st.session_state.role == 'admin':
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">⚙️ Admin Panel</div>
                <div class="feature-desc">
                    Manage all registered users, view system-wide
                    predictions, download reports, and monitor
                    platform activity.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.page_link("pages/6_Admin.py", label="Open Admin Panel →")
        else:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🏭 Industry Selector</div>
                <div class="feature-desc">
                    View all 3 industry models, check their status,
                    and see performance comparison across
                    Telecom, Banking, and E-Commerce.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.page_link("pages/7_Industry_Select.py", label="Select Industry →")

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">ℹ️ About ChurnGuard Pro</div>
            <div class="feature-desc">
                Built with Python, XGBoost, PyTorch, SHAP,
                GridSearchCV, SQLite, and ReportLab.
                Supports Telecom, Banking, E-Commerce.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- About section ----
    st.subheader("About ChurnGuard Pro")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **What is ChurnGuard Pro?**

        ChurnGuard Pro is an AI-powered customer retention platform
        that predicts which customers are likely to leave across
        three industries — Telecom, Banking, and E-Commerce.
        Instead of losing customers and wondering why, companies
        can identify at-risk customers early and take action
        before they leave.

        **How it works:**
        The system learns patterns from real customer data using
        XGBoost trained with GridSearchCV parameter optimization
        and evaluated with 5-fold cross-validation. SHAP values
        explain every prediction in plain English. The Revenue
        Impact Calculator converts predictions into business value.
        """)

    with col2:
        st.markdown("""
        **Technologies used:**
        - Python 3 — programming language
        - Streamlit — web application framework
        - XGBoost — best performing ML model
        - PyTorch — Neural Network deep learning
        - SHAP — model explainability
        - GridSearchCV — automatic parameter tuning
        - Cross-validation — enterprise-grade evaluation
        - K-Means — customer segmentation clustering
        - SQLite — user database with encrypted passwords
        - ReportLab — professional PDF generation
        - Deployed on Streamlit Cloud

        **Industries:** Telecom, Banking, E-Commerce
        """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = ''
            st.session_state.full_name = ''
            st.session_state.role = ''
            st.rerun()

# ================================================
# NOT LOGGED IN — LOGIN / REGISTER
# ================================================
else:
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("""
        <br><br>
        <div class="hero-badge">
            AI-Powered Multi-Industry Retention Platform
        </div>
        <div class="hero-title">ChurnGuard Pro</div>
        <div class="hero-subtitle">
            Predict customer churn across Telecom, Banking,
            and E-Commerce using XGBoost, Neural Networks,
            SHAP explainability, and GridSearchCV tuning.
            Built for enterprise use.
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            '<div class="stat-box"><div class="stat-number">84%</div>'
            '<div class="stat-label">AUC Score</div></div>',
            unsafe_allow_html=True
        )
        c2.markdown(
            '<div class="stat-box"><div class="stat-number">3</div>'
            '<div class="stat-label">Industries</div></div>',
            unsafe_allow_html=True
        )
        c3.markdown(
            '<div class="stat-box"><div class="stat-number">4</div>'
            '<div class="stat-label">ML Models</div></div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### Sign in to ChurnGuard Pro")
        st.markdown("---")

        tab1, tab2 = st.tabs(["Sign In", "Create Account"])

        with tab1:
            login_username = st.text_input(
                "Username",
                placeholder="Enter your username",
                key="li_user"
            )
            login_pass = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="li_pass"
            )
            if st.button("Sign In", use_container_width=True, key="login_btn"):
                if not login_username or not login_pass:
                    st.error("Please enter both username and password.")
                else:
                    result = login_user(login_username, login_pass)
                    if result['success']:
                        st.session_state.logged_in = True
                        st.session_state.user_id   = result['user_id']
                        st.session_state.username  = result['username']
                        st.session_state.full_name = result['full_name']
                        st.session_state.role      = result['role']
                        st.rerun()
                    else:
                        st.error(result['error'])
            st.caption(
                "Default admin: username = admin | password = 1234"
            )

        with tab2:
            reg_name  = st.text_input(
                "Full name",
                placeholder="e.g. Ahmed Khan",
                key="reg_name"
            )
            reg_email = st.text_input(
                "Email",
                placeholder="e.g. ahmed@email.com",
                key="reg_email"
            )
            reg_user  = st.text_input(
                "Username",
                placeholder="Choose a unique username",
                key="reg_user"
            )
            reg_pass  = st.text_input(
                "Password",
                type="password",
                placeholder="Minimum 6 characters",
                key="reg_pass"
            )
            reg_pass2 = st.text_input(
                "Confirm password",
                type="password",
                placeholder="Repeat your password",
                key="reg_pass2"
            )

            if st.button("Create Account", use_container_width=True, key="register_btn"):
                if not all([reg_name, reg_email, reg_user, reg_pass, reg_pass2]):
                    st.error("Please fill in all fields.")
                elif len(reg_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif reg_pass != reg_pass2:
                    st.error("Passwords do not match.")
                elif '@' not in reg_email:
                    st.error("Please enter a valid email address.")
                else:
                    result = register_user(reg_name, reg_email, reg_user, reg_pass)
                    if result['success']:
                        st.success("Account created! Please sign in using the Sign In tab.")
                    else:
                        st.error(result['error'])