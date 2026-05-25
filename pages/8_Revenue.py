# pages/8_Revenue.py — Revenue Impact Calculator

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
from industry_config import INDUSTRIES

st.set_page_config(page_title="Revenue Impact", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

st.title("Revenue Impact Calculator")
st.write(
    "Calculate exactly how much money churning customers "
    "represent — and how much your retention team can save."
)
st.divider()

# ================================================
# INDUSTRY + REVENUE INPUTS
# ================================================
st.subheader("Step 1 — Configure your business")

col1, col2 = st.columns(2)

with col1:
    industry = st.selectbox(
        "Select Industry",
        options=['telecom', 'banking', 'ecommerce'],
        format_func=lambda x: {
            'telecom'  : '📡 Telecom',
            'banking'  : '🏦 Banking',
            'ecommerce': '🛒 E-Commerce'
        }[x]
    )

    avg_monthly = st.number_input(
        "Average monthly revenue per customer ($)",
        min_value=1.0,
        max_value=10000.0,
        value={
            'telecom'  : 65.0,
            'banking'  : 150.0,
            'ecommerce': 45.0
        }[industry],
        step=5.0,
        help="How much does one average customer pay per month?"
    )

with col2:
    avg_lifetime = st.number_input(
        "Average customer lifetime (months)",
        min_value=1,
        max_value=120,
        value={
            'telecom'  : 24,
            'banking'  : 36,
            'ecommerce': 18
        }[industry],
        step=1,
        help="How many months does a typical customer stay?"
    )

    retention_cost = st.number_input(
        "Cost of retention action per customer ($)",
        min_value=0.0,
        max_value=1000.0,
        value=25.0,
        step=5.0,
        help="How much does it cost to call/offer a customer?"
    )

retention_rate = st.slider(
    "Estimated retention success rate (%)",
    min_value=5,
    max_value=80,
    value=30,
    step=5,
    help="What percentage of at-risk customers can you save?"
)

st.divider()

# ================================================
# DATA SOURCE — USE FILE OR MANUAL INPUT
# ================================================
st.subheader("Step 2 — Enter customer data")

data_source = st.radio(
    "How do you want to calculate?",
    ["Upload a CSV file", "Enter numbers manually"],
    horizontal=True
)

total_customers = 0
predicted_churners = 0
df_results = None

if data_source == "Upload a CSV file":
    uploaded = st.file_uploader(
        "Upload your customer CSV with predictions",
        type=["csv"],
        help="Upload the results CSV from Bulk Upload page"
    )

    if uploaded:
        df_results = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_results):,} customers")
        st.dataframe(df_results.head(3), use_container_width=True)

        total_customers = len(df_results)

        # Try to find churn prediction column
        churn_col = None
        for col in df_results.columns:
            if 'churn' in col.lower() and 'prob' not in col.lower():
                churn_col = col
                break

        if churn_col:
            churn_mask = df_results[churn_col].astype(str).str.contains(
                'CHURN|1|Yes|churn', case=False, na=False
            )
            predicted_churners = int(churn_mask.sum())
            st.info(
                f"Found {predicted_churners:,} customers "
                f"predicted to churn from column '{churn_col}'"
            )
        else:
            predicted_churners = st.number_input(
                "Number of customers predicted to churn",
                min_value=0,
                max_value=total_customers,
                value=int(total_customers * 0.25)
            )

else:
    col1, col2 = st.columns(2)
    with col1:
        total_customers = st.number_input(
            "Total customers", min_value=1,
            max_value=1000000, value=1000, step=100
        )
    with col2:
        predicted_churners = st.number_input(
            "Customers predicted to churn",
            min_value=0,
            max_value=total_customers,
            value=int(total_customers * 0.25),
            step=10
        )

st.divider()

# ================================================
# CALCULATE BUTTON
# ================================================
if st.button(
    "Calculate Revenue Impact",
    use_container_width=True,
    type="primary"
):
    if total_customers == 0:
        st.error("Please enter customer data first.")
        st.stop()

    # ---- Core calculations ----
    churn_rate_pct  = (predicted_churners / total_customers * 100
                       if total_customers > 0 else 0)
    ltv             = avg_monthly * avg_lifetime

    monthly_loss    = predicted_churners * avg_monthly
    annual_loss     = monthly_loss * 12
    lifetime_loss   = predicted_churners * ltv

    customers_saved = int(predicted_churners * retention_rate / 100)
    revenue_saved   = customers_saved * ltv
    retention_spend = predicted_churners * retention_cost
    net_benefit     = revenue_saved - retention_spend
    roi_pct         = ((net_benefit / retention_spend * 100)
                       if retention_spend > 0 else 0)

    st.divider()

    # ================================================
    # RESULTS — TOP METRICS
    # ================================================
    st.subheader("Revenue Impact Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Total customers at risk",
        f"{predicted_churners:,}",
        delta=f"{churn_rate_pct:.1f}% churn rate",
        delta_color="inverse"
    )
    c2.metric(
        "Monthly revenue at risk",
        f"${monthly_loss:,.0f}",
        delta="Per month",
        delta_color="inverse"
    )
    c3.metric(
        "Annual revenue at risk",
        f"${annual_loss:,.0f}",
        delta="If no action taken",
        delta_color="inverse"
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customer lifetime value (LTV)", f"${ltv:,.0f}")
    c2.metric("Total lifetime loss",           f"${lifetime_loss:,.0f}")
    c3.metric("Customers you can save",        f"{customers_saved:,}")
    c4.metric("Revenue you can save",          f"${revenue_saved:,.0f}")

    st.divider()

    # ================================================
    # ROI SECTION
    # ================================================
    st.subheader("Return on Investment")

    col1, col2 = st.columns(2)

    with col1:
        if net_benefit > 0:
            st.success(
                f"Net benefit of retention campaign: "
                f"${net_benefit:,.0f}"
            )
        else:
            st.warning(
                f"Net benefit: ${net_benefit:,.0f} — "
                "consider reducing retention cost or increasing success rate"
            )

        roi_data = {
            'Item'  : [
                'Retention campaign cost',
                'Revenue saved',
                'Net benefit',
                'ROI'
            ],
            'Amount': [
                f"${retention_spend:,.0f}",
                f"${revenue_saved:,.0f}",
                f"${net_benefit:,.0f}",
                f"{roi_pct:.0f}%"
            ]
        }
        st.dataframe(
            pd.DataFrame(roi_data),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        # ROI bar chart
        fig, ax = plt.subplots(figsize=(5, 4))
        categories = ['Campaign\ncost', 'Revenue\nsaved', 'Net\nbenefit']
        values     = [retention_spend, revenue_saved, net_benefit]
        colors     = [
            '#E24B4A',
            '#1A7A4A',
            '#1A7A4A' if net_benefit > 0 else '#E24B4A'
        ]
        bars = ax.bar(categories, values, color=colors, alpha=0.85)
        ax.set_title('ROI Breakdown', fontweight='bold')
        ax.set_ylabel('Amount ($)')
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h + max(values) * 0.01,
                f'${h:,.0f}',
                ha='center', fontsize=9, fontweight='bold'
            )
        ax.axhline(y=0, color='gray', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ================================================
    # MONTHLY PROJECTION CHART
    # ================================================
    st.subheader("12-Month Revenue Loss Projection")

    months      = list(range(1, 13))
    cumulative  = [monthly_loss * m for m in months]
    saved_line  = [revenue_saved * (m/12) for m in months]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(months, cumulative,
                    alpha=0.3, color='#E24B4A',
                    label='Revenue lost (no action)')
    ax.plot(months, cumulative,
            color='#E24B4A', linewidth=2)
    ax.fill_between(months, saved_line,
                    alpha=0.3, color='#1A7A4A',
                    label='Revenue saved (with ChurnGuard)')
    ax.plot(months, saved_line,
            color='#1A7A4A', linewidth=2, linestyle='--')

    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Cumulative Revenue ($)', fontsize=11)
    ax.set_title(
        'Cumulative Revenue: Lost vs Saved Over 12 Months',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.set_xticks(months)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x:,.0f}')
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ================================================
    # BUSINESS SUMMARY — PRINTABLE
    # ================================================
    st.subheader("Executive Summary")

    summary = f"""
**ChurnGuard Pro — Revenue Impact Report**

**Industry:** {industry.capitalize()}
**Total customers analysed:** {total_customers:,}
**Predicted to churn:** {predicted_churners:,} ({churn_rate_pct:.1f}%)

**Financial Impact:**
- Monthly revenue at risk: ${monthly_loss:,.0f}
- Annual revenue at risk: ${annual_loss:,.0f}
- Total lifetime value at risk: ${lifetime_loss:,.0f}

**Retention Campaign Results:**
- Customers that can be saved: {customers_saved:,}
- Revenue that can be saved: ${revenue_saved:,.0f}
- Campaign cost: ${retention_spend:,.0f}
- Net benefit: ${net_benefit:,.0f}
- Return on investment: {roi_pct:.0f}%

**Recommendation:** {"Immediately launch retention campaign — ROI is positive." if net_benefit > 0 else "Review retention strategy to reduce cost per customer."}
"""
    st.markdown(summary)

    # Download as CSV
    export_data = pd.DataFrame({
        'Metric': [
            'Total customers',
            'Predicted churners',
            'Churn rate',
            'Monthly loss',
            'Annual loss',
            'Lifetime loss',
            'Customers saved',
            'Revenue saved',
            'Campaign cost',
            'Net benefit',
            'ROI'
        ],
        'Value': [
            total_customers,
            predicted_churners,
            f"{churn_rate_pct:.1f}%",
            f"${monthly_loss:,.0f}",
            f"${annual_loss:,.0f}",
            f"${lifetime_loss:,.0f}",
            customers_saved,
            f"${revenue_saved:,.0f}",
            f"${retention_spend:,.0f}",
            f"${net_benefit:,.0f}",
            f"{roi_pct:.0f}%"
        ]
    })

    st.download_button(
        label="Download Revenue Report as CSV",
        data=export_data.to_csv(index=False),
        file_name="churnguard_revenue_impact.csv",
        mime="text/csv",
        use_container_width=True
    )