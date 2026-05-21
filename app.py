# app.py — Login and Registration page

import streamlit as st
from database import initialize_database, login_user, register_user

# Initialize database when app starts
initialize_database()

st.set_page_config(
    page_title="Churn Predictor Pro",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
for key, val in {
    'logged_in' : False,
    'user_id'   : None,
    'username'  : '',
    'full_name' : '',
    'role'      : '',
    'page'      : 'login'
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---- Already logged in ----
if st.session_state.logged_in:
    st.success(f"Welcome back, {st.session_state.full_name}!")
    st.info("Use the sidebar on the left to navigate to any page.")

    st.page_link("pages/1_Dashboard.py",      label="Go to Dashboard")
    st.page_link("pages/2_Predict_Single.py", label="Predict Single Customer")
    st.page_link("pages/3_Bulk_Upload.py",    label="Bulk Upload")
    st.page_link("pages/4_Model_Report.py",   label="Model Performance Report")
    st.page_link("pages/5_History.py",        label="My Prediction History")

    if st.session_state.role == 'admin':
        st.page_link("pages/6_Admin.py", label="Admin Panel")

    st.divider()
    if st.button("Logout", use_container_width=False):
        for key in ['logged_in','user_id','username','full_name','role']:
            st.session_state[key] = False if key == 'logged_in' else ''
        st.rerun()

# ---- Not logged in ----
else:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])

    with col2:
        st.title("Churn Predictor Pro")
        st.write("Sign in or create a new account to get started.")
        st.divider()

        # Toggle between login and register
        tab1, tab2 = st.tabs(["Sign In", "Create Account"])

        # ---- LOGIN TAB ----
        with tab1:
            st.subheader("Sign in to your account")
            login_username = st.text_input(
                "Username", placeholder="Enter your username", key="li_user"
            )
            login_password = st.text_input(
                "Password", type="password",
                placeholder="Enter your password", key="li_pass"
            )

            if st.button("Sign In", use_container_width=True, key="login_btn"):
                if not login_username or not login_password:
                    st.error("Please enter both username and password.")
                else:
                    result = login_user(login_username, login_password)
                    if result['success']:
                        st.session_state.logged_in = True
                        st.session_state.user_id   = result['user_id']
                        st.session_state.username  = result['username']
                        st.session_state.full_name = result['full_name']
                        st.session_state.role      = result['role']
                        st.rerun()
                    else:
                        st.error(result['error'])

            st.caption("Default admin account: admin / 1234")

        # ---- REGISTER TAB ----
        with tab2:
            st.subheader("Create a new account")
            reg_name  = st.text_input(
                "Full name", placeholder="e.g. Ahmed Khan", key="reg_name"
            )
            reg_email = st.text_input(
                "Email address", placeholder="e.g. ahmed@email.com",
                key="reg_email"
            )
            reg_user  = st.text_input(
                "Username", placeholder="Choose a unique username",
                key="reg_user"
            )
            reg_pass  = st.text_input(
                "Password", type="password",
                placeholder="Minimum 6 characters", key="reg_pass"
            )
            reg_pass2 = st.text_input(
                "Confirm password", type="password",
                placeholder="Repeat your password", key="reg_pass2"
            )

            if st.button("Create Account", use_container_width=True,
                         key="register_btn"):
                # Validate inputs
                if not all([reg_name, reg_email, reg_user,
                            reg_pass, reg_pass2]):
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
                            "Account created successfully! "
                            "Please sign in using the Sign In tab."
                        )
                    else:
                        st.error(result['error'])