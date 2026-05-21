# pages/6_Admin.py — Admin panel

import streamlit as st
import pandas as pd
from database import get_all_users, get_all_predictions, get_user_count, delete_user

st.set_page_config(page_title="Admin Panel", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

if st.session_state.role != 'admin':
    st.error("Access denied. Admin only page.")
    st.stop()

st.title("Admin Panel")
st.write("Manage all users and view all predictions across the system.")
st.divider()

# ---- Summary metrics ----
st.subheader("System Overview")

all_users = get_all_users()
all_preds = get_all_predictions()

total_users = len(all_users)
total_preds = len(all_preds)
churn_preds = len([p for p in all_preds if p['prediction'] == 'CHURN'])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total users",       total_users)
c2.metric("Total predictions", total_preds)
c3.metric("Churn predicted",   churn_preds)
c4.metric("Stay predicted",    total_preds - churn_preds)

st.divider()

# ---- All users table ----
st.subheader("Registered Users")

if all_users:
    users_df = pd.DataFrame(all_users)
    st.dataframe(users_df, use_container_width=True)

    st.subheader("Delete a User")
    st.warning("Deleting a user also deletes all their predictions.")

    user_options = {
        f"{u['username']} ({u['full_name']})": u['id']
        for u in all_users
        if u['username'] != 'admin'
    }

    if user_options:
        selected = st.selectbox(
            "Select user to delete", list(user_options.keys())
        )
        if st.button("Delete selected user", type="primary"):
            delete_user(user_options[selected])
            st.success(f"User {selected} deleted.")
            st.rerun()
    else:
        st.info("No non-admin users to delete.")
else:
    st.info("No users registered yet.")

st.divider()

# ---- All predictions ----
st.subheader("All Predictions — All Users")

if all_preds:
    preds_df = pd.DataFrame(all_preds)
    st.dataframe(preds_df, use_container_width=True)

    csv = preds_df.to_csv(index=False)
    st.download_button(
        "Download all predictions as CSV",
        data=csv,
        file_name="all_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.info("No predictions made yet.")