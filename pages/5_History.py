# pages/5_History.py — User's personal prediction history

import streamlit as st
import pandas as pd
from database import get_user_predictions

st.set_page_config(page_title="My History", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

st.title("My Prediction History")
st.write(f"All predictions made by {st.session_state.full_name}")
st.divider()

predictions = get_user_predictions(st.session_state.user_id)

if not predictions:
    st.info(
        "You have not made any predictions yet. "
        "Go to Predict Single and make your first prediction!"
    )
    st.stop()

df = pd.DataFrame(predictions)

# ---- Summary ----
st.subheader("Your Summary")

total   = len(df)
churned = len(df[df['prediction'] == 'CHURN'])
stayed  = len(df[df['prediction'] == 'STAY'])
high    = len(df[df['risk_level'] == 'High Risk']) if 'risk_level' in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total predictions", total)
c2.metric("Churn predicted",   churned)
c3.metric("Stay predicted",    stayed)
c4.metric("High risk",         high)

st.divider()

# ---- Filter ----
st.subheader("Filter")

col1, col2 = st.columns(2)
with col1:
    pred_filter = st.selectbox(
        "Filter by prediction", ["All", "CHURN", "STAY"]
    )
with col2:
    risk_filter = st.selectbox(
        "Filter by risk", ["All", "High Risk", "Medium Risk", "Low Risk"]
    )

filtered = df.copy()
if pred_filter != "All":
    filtered = filtered[filtered['prediction'] == pred_filter]
if risk_filter != "All" and 'risk_level' in filtered.columns:
    filtered = filtered[filtered['risk_level'] == risk_filter]

# Show clean columns only
show_cols = ['date', 'time', 'tenure', 'monthly_charges',
             'contract', 'internet', 'churn_probability',
             'prediction', 'risk_level']
show_cols = [c for c in show_cols if c in filtered.columns]

st.write(f"Showing {len(filtered)} of {total} predictions")
st.dataframe(filtered[show_cols], use_container_width=True)

st.divider()

# ---- Download ----
csv = filtered[show_cols].to_csv(index=False)
st.download_button(
    "Download my history as CSV",
    data=csv,
    file_name="my_prediction_history.csv",
    mime="text/csv",
    use_container_width=True
)