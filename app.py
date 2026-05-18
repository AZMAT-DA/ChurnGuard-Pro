# app.py — Login page (home page of the multi-page app)

import streamlit as st

st.set_page_config(
    page_title="Churn Predictor Pro",
    page_icon="📊",
    layout="wide"
)

# Hide default Streamlit menu for cleaner look
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'username' not in st.session_state:
    st.session_state.username = ''

# ---- If already logged in, show welcome and page links ----
if st.session_state.logged_in:
    st.success(f"Welcome back, {st.session_state.username}!")
    st.info("Use the sidebar on the left to navigate to any page.")

    st.page_link("pages/1_Dashboard.py",      label="Go to Dashboard")
    st.page_link("pages/2_Predict_Single.py", label="Predict Single Customer")
    st.page_link("pages/3_Bulk_Upload.py",    label="Bulk Upload")
    st.page_link("pages/4_Model_Report.py",   label="Model Performance Report")

    st.divider()
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username  = ''
        st.rerun()

# ---- Login form ----
else:
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.title("Churn Predictor Pro")
        st.write("Sign in to access your dashboard")
        st.divider()

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password",
                                 placeholder="Enter password")

        if st.button("Sign In", use_container_width=True):
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.rerun()
            else:
                st.error("Wrong username or password. Try: admin / 1234")

        st.divider()
        st.caption("Default login — Username: admin | Password: 1234")