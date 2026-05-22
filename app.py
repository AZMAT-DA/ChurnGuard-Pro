# app.py — Professional landing page with login and registration

import streamlit as st
from database import initialize_database, login_user, register_user

# Initialize database
initialize_database()

st.set_page_config(
    page_title="Churn Predictor Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}

    /* Hero section */
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #4A5568;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: #E8F4FD;
        color: #1E3A5F;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border: 1px solid #BDD7EE;
    }

    /* Feature cards */
    .feature-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: #4A5568;
        line-height: 1.5;
    }

    /* Stats bar */
    .stat-box {
        text-align: center;
        padding: 1.5rem;
        background: #F7FAFC;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #718096;
        margin-top: 4px;
    }

    /* Welcome banner */
    .welcome-banner {
        background: linear-gradient(135deg, #1E3A5F 0%, #2D5986 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    .welcome-name {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .welcome-sub {
        font-size: 1rem;
        opacity: 0.85;
    }

    /* Nav cards */
    .nav-card {
        background: white;
        border: 1.5px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .nav-card:hover {
        border-color: #1E3A5F;
        background: #F0F4F8;
    }
    .nav-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1E3A5F;
    }
    .nav-desc {
        font-size: 0.82rem;
        color: #718096;
        margin-top: 2px;
    }

    /* Divider style */
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #1E3A5F, #4299E1, transparent);
        border-radius: 2px;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

    # Welcome banner
    st.markdown(f"""
    <div class="welcome-banner">
        <div class="welcome-name">Welcome back, {st.session_state.full_name}!</div>
        <div class="welcome-sub">
            Churn Predictor Pro — AI-Powered Customer Retention System
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- About / intro section ----
    st.markdown("""
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown("""
        <div class="hero-badge">Final Year Project — Machine Learning</div>
        <div class="hero-title">Customer Churn<br>Prediction System</div>
        <div class="hero-subtitle">
            An intelligent system that uses Machine Learning to predict
            which telecom customers are likely to leave — so your
            retention team can act before it is too late.
            Built with XGBoost, Neural Networks, and SHAP explainability.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Project stats
        st.markdown('<div class="stat-box"><div class="stat-number">7,043</div><div class="stat-label">Customers in dataset</div></div>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="stat-box"><div class="stat-number">84%</div><div class="stat-label">Best model AUC score</div></div>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="stat-box"><div class="stat-number">4</div><div class="stat-label">ML models trained</div></div>', unsafe_allow_html=True)

    st.markdown("""<div class="section-divider"></div>""",
                unsafe_allow_html=True)

    # ---- Feature navigation cards ----
    st.subheader("Navigate to a feature")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 Analytics Dashboard</div>
            <div class="feature-desc">
                View real-time churn statistics, charts by contract type,
                internet service, tenure distribution, and model performance.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(
            "pages/1_Dashboard.py",
            label="Open Dashboard →"
        )

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
        st.page_link(
            "pages/2_Predict_Single.py",
            label="Make Prediction →"
        )

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
        st.page_link(
            "pages/3_Bulk_Upload.py",
            label="Upload CSV →"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🤖 Model Report</div>
            <div class="feature-desc">
                Compare all 4 trained models — Logistic Regression,
                Random Forest, XGBoost, and Neural Network —
                with accuracy, recall, F1, and AUC scores.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(
            "pages/4_Model_Report.py",
            label="View Models →"
        )

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📋 My History</div>
            <div class="feature-desc">
                View all predictions you have made, filter by risk
                level, download your history as CSV,
                and track your prediction activity over time.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(
            "pages/5_History.py",
            label="View History →"
        )

    with col3:
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
            st.page_link(
                "pages/6_Admin.py",
                label="Open Admin Panel →"
            )
        else:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">ℹ️ About this project</div>
                <div class="feature-desc">
                    Built with Python, Streamlit, XGBoost, PyTorch,
                    and SHAP. Dataset: Telco Customer Churn from Kaggle.
                    Deployed on Streamlit Cloud.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""<div class="section-divider"></div>""",
                unsafe_allow_html=True)

    # ---- About section ----
    st.subheader("About this project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **What is customer churn?**

        Churn means a customer stops using a service and leaves.
        For a telecom company like Jazz or Zong, losing a customer
        costs 5 to 7 times more than keeping one.
        This system predicts who is about to leave so the company
        can offer them a deal before they go.

        **How it works:**
        The system was trained on 7,043 real telecom customer records.
        It learned patterns — like month-to-month contract customers
        with high monthly charges are most likely to churn.
        When you enter a customer's details, it applies those learned
        patterns to predict their churn risk.
        """)

    with col2:
        st.markdown("""
        **Technologies used:**
        - Python 3 — programming language
        - Streamlit — web application framework
        - XGBoost — best performing ML model (84% AUC)
        - PyTorch — Neural Network deep learning
        - SHAP — model explainability
        - SQLite — user database
        - ReportLab — PDF report generation
        - scikit-learn — preprocessing and evaluation
        - Deployed on Streamlit Cloud

        **Dataset:** Telco Customer Churn (Kaggle)
        — 7,043 customers, 21 features
        """)

    st.markdown("""<div class="section-divider"></div>""",
                unsafe_allow_html=True)

    # Logout
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Logout", use_container_width=True):
            for key in ['logged_in', 'user_id',
                        'username', 'full_name', 'role']:
                st.session_state[key] = (
                    False if key == 'logged_in' else ''
                )
            st.rerun()

# ================================================
# NOT LOGGED IN — LOGIN / REGISTER PAGE
# ================================================
else:

    # Hero section
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("""
        <br><br>
        <div class="hero-badge">AI-Powered Machine Learning System</div>
        <div class="hero-title">Predict Customer<br>Churn Before<br>It Happens</div>
        <div class="hero-subtitle">
            An intelligent system that uses XGBoost, Neural Networks,
            and SHAP explainability to identify at-risk customers
            and recommend the right retention actions.
        </div>
        """, unsafe_allow_html=True)

        # Quick stats
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="stat-box"><div class="stat-number">84%</div><div class="stat-label">AUC Score</div></div>', unsafe_allow_html=True)
        c2.markdown('<div class="stat-box"><div class="stat-number">4</div><div class="stat-label">ML Models</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="stat-box"><div class="stat-number">7K+</div><div class="stat-label">Customers</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### Sign in to your account")
        st.markdown("---")

        tab1, tab2 = st.tabs(["Sign In", "Create Account"])

        # ---- LOGIN TAB ----
        with tab1:
            login_user_input = st.text_input(
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

            if st.button("Sign In", use_container_width=True,
                         key="login_btn"):
                if not login_user_input or not login_pass:
                    st.error("Please enter both username and password.")
                else:
                    result = login_user(login_user_input, login_pass)
                    if result['success']:
                        st.session_state.logged_in = True
                        st.session_state.user_id   = result['user_id']
                        st.session_state.username  = result['username']
                        st.session_state.full_name = result['full_name']
                        st.session_state.role      = result['role']
                        st.rerun()
                    else:
                        st.error(result['error'])

            st.caption("Default admin: username = admin | password = 1234")

        # ---- REGISTER TAB ----
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

            if st.button("Create Account",
                         use_container_width=True,
                         key="register_btn"):
                if not all([reg_name, reg_email,
                            reg_user, reg_pass, reg_pass2]):
                    st.error("Please fill in all fields.")
                elif len(reg_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif reg_pass != reg_pass2:
                    st.error("Passwords do not match.")
                elif '@' not in reg_email:
                    st.error("Please enter a valid email address.")
                else:
                    result = register_user(
                        reg_name, reg_email, reg_user, reg_pass
                    )
                    if result['success']:
                        st.success(
                            "Account created! "
                            "Please sign in using the Sign In tab."
                        )
                    else:
                        st.error(result['error'])