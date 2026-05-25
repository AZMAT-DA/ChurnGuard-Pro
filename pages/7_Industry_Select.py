# pages/7_Industry_Select.py — Industry selector and comparison

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Industry Select", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

from industry_config import INDUSTRIES

st.title("Industry Selection")
st.write(
    "ChurnGuard Pro supports 3 industries. "
    "Select your industry to load the right model."
)
st.divider()

# ---- Industry cards ----
st.subheader("Select your industry")

col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

for i, (key, config) in enumerate(INDUSTRIES.items()):
    model_exists = os.path.exists(config['model_file'])
    status = "✅ Model ready" if model_exists else "⚠️ Not trained yet"

    with cols[i]:
        st.markdown(f"""
        <div style='background:white; border:1px solid #E2E8F0;
                    border-radius:12px; padding:1.5rem;
                    margin-bottom:1rem;'>
            <h3 style='color:#1E3A5F; margin:0 0 8px'>
                {config['icon']} {config['name']}
            </h3>
            <p style='color:#4A5568; font-size:0.9rem; margin:0 0 12px'>
                {config['description']}
            </p>
            <p style='color:#2D8A4E; font-size:0.85rem; margin:0'>
                {status}
            </p>
        </div>
        """, unsafe_allow_html=True)

        if model_exists:
            if st.button(
                f"Use {config['name']} model",
                key=f"btn_{key}",
                use_container_width=True
            ):
                st.session_state.selected_industry = key
                st.session_state.industry_name     = config['name']
                st.session_state.industry_icon     = config['icon']
                st.success(
                    f"{config['icon']} {config['name']} model selected! "
                    f"Go to Predict Single to make predictions."
                )
        else:
            st.info("Run multi_train.py to train this model")

st.divider()

# ---- Show current selection ----
if 'selected_industry' in st.session_state:
    current = st.session_state.selected_industry
    config  = INDUSTRIES[current]
    st.success(
        f"Currently active: {config['icon']} "
        f"{config['name']} model"
    )

st.divider()

# ---- Results comparison table ----
st.subheader("Model performance across industries")

try:
    results_df = pd.read_csv('models/industry_results.csv', index_col=0)
    st.dataframe(
        results_df.style.highlight_max(
            subset=['Accuracy', 'Precision',
                    'Recall', 'F1-Score', 'AUC'],
            color='#C0DD97'
        ),
        use_container_width=True
    )
except Exception:
    st.info(
        "Run multi_train.py first to see performance comparison."
    )