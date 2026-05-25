# pages/9_Segments.py — Customer Segmentation using K-Means

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster    import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Segments", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

st.title("Customer Segmentation")
st.write(
    "Group your customers into meaningful segments using "
    "K-Means clustering — so your team knows exactly who "
    "needs what type of action."
)
st.divider()

# ================================================
# WHAT IS SEGMENTATION — EXPLANATION
# ================================================
st.info(
    "**What is customer segmentation?** Instead of treating all "
    "at-risk customers the same way, segmentation groups them by "
    "behavior patterns. A loyal long-term customer needs a "
    "different retention approach than a new high-paying customer. "
    "K-Means clustering finds these groups automatically from the data."
)

st.divider()

# ================================================
# INDUSTRY SELECTOR
# ================================================
industry = st.selectbox(
    "Select Industry",
    options=['telecom', 'banking', 'ecommerce'],
    format_func=lambda x: {
        'telecom'  : '📡 Telecom',
        'banking'  : '🏦 Banking',
        'ecommerce': '🛒 E-Commerce'
    }[x]
)

# ================================================
# LOAD DATASET FOR SELECTED INDUSTRY
# ================================================
@st.cache_data
def load_data(ind):
    try:
        if ind == 'telecom':
            df = pd.read_csv(
                'datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv'
            )
            df['TotalCharges'] = pd.to_numeric(
                df['TotalCharges'], errors='coerce'
            )
            df['Churn_num'] = df['Churn'].map({'Yes': 1, 'No': 0})
            features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            label_map = {
                0: 'Low Value — New',
                1: 'High Value — Loyal',
                2: 'High Risk — Churning',
                3: 'Medium Value — Stable'
            }
        elif ind == 'banking':
            df = pd.read_csv('datasets/Churn_Modelling.csv')
            df['Churn_num'] = df['Exited']
            features = ['CreditScore', 'Age', 'Balance',
                        'EstimatedSalary', 'Tenure']
            label_map = {
                0: 'Low Balance — Young',
                1: 'High Value — Senior',
                2: 'Mid Tier — Active',
                3: 'At Risk — Inactive'
            }
        else:
            df = pd.read_excel(
                'datasets/E Commerce Dataset.xlsx',
                sheet_name='E Comm'
            )
            df['Churn_num'] = df['Churn']
            features = ['Tenure', 'SatisfactionScore',
                        'OrderCount', 'CashbackAmount']
            label_map = {
                0: 'New — Low Spend',
                1: 'Loyal — High Spend',
                2: 'Dissatisfied — At Risk',
                3: 'Occasional — Mid Tier'
            }
        return df, features, label_map
    except Exception as e:
        return None, None, str(e)

df, features, label_map = load_data(industry)

if df is None:
    st.error(f"Could not load data: {label_map}")
    st.stop()

st.success(
    f"Loaded {len(df):,} customers — "
    f"segmenting by: {', '.join(features)}"
)

# ================================================
# SETTINGS
# ================================================
st.subheader("Segmentation Settings")

col1, col2 = st.columns(2)
with col1:
    n_clusters = st.slider(
        "Number of segments (K)",
        min_value=2,
        max_value=6,
        value=4,
        help="How many customer groups to create. 4 is recommended."
    )
with col2:
    show_raw = st.checkbox("Show raw data preview", value=False)

if show_raw:
    st.dataframe(df[features + ['Churn_num']].head(10),
                 use_container_width=True)

st.divider()

# ================================================
# RUN SEGMENTATION
# ================================================
if st.button("Run Segmentation", use_container_width=True,
             type="primary"):

    with st.spinner("Running K-Means clustering..."):

        # Prepare data — drop rows with missing values
        seg_df = df[features + ['Churn_num']].dropna().copy()

        # Scale features
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(seg_df[features])

        # Run K-Means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        seg_df['Segment'] = kmeans.fit_predict(X_scaled)

        # Name segments based on churn rate
        seg_summary = seg_df.groupby('Segment').agg(
            Count      = ('Churn_num', 'count'),
            Churn_Rate = ('Churn_num', 'mean'),
            **{f: (f, 'mean') for f in features}
        ).round(2)
        seg_summary['Churn_Rate_Pct'] = (
            seg_summary['Churn_Rate'] * 100
        ).round(1)

        # Auto-name segments by churn rate
        seg_summary = seg_summary.sort_values(
            'Churn_Rate', ascending=False
        )
        seg_names = [
            'High Risk — Immediate Action',
            'Medium Risk — Monitor Closely',
            'Low Risk — Stable',
            'Very Low Risk — Loyal Champions',
            'Undecided — Needs Analysis',
            'New — Too Early to Tell'
        ]
        seg_summary['Segment_Name'] = seg_names[:len(seg_summary)]
        seg_summary = seg_summary.reset_index()

        # Map names back to original df
        name_map = dict(zip(
            seg_summary['Segment'],
            seg_summary['Segment_Name']
        ))
        seg_df['Segment_Name'] = seg_df['Segment'].map(name_map)

    st.divider()

    # ================================================
    # OVERVIEW METRICS
    # ================================================
    st.subheader("Segment Overview")

    cols = st.columns(min(n_clusters, 4))
    risk_colors = {
        'High Risk — Immediate Action'  : '#E24B4A',
        'Medium Risk — Monitor Closely' : '#EF9F27',
        'Low Risk — Stable'             : '#1A7A4A',
        'Very Low Risk — Loyal Champions': '#0C447C',
        'Undecided — Needs Analysis'    : '#8B5CF6',
        'New — Too Early to Tell'       : '#718096',
    }

    for i, (_, row) in enumerate(seg_summary.iterrows()):
        col_idx = i % len(cols)
        with cols[col_idx]:
            color = risk_colors.get(row['Segment_Name'], '#718096')
            st.markdown(
                f"<div style='background:{color}15; "
                f"border-left:4px solid {color}; "
                f"border-radius:8px; padding:12px; "
                f"margin-bottom:8px;'>"
                f"<b style='color:{color}'>"
                f"{row['Segment_Name']}</b><br>"
                f"<span style='font-size:1.4rem; font-weight:700'>"
                f"{int(row['Count']):,}</span> customers<br>"
                f"<span style='color:#666; font-size:0.85rem'>"
                f"Churn rate: {row['Churn_Rate_Pct']}%</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.divider()

    # ================================================
    # DETAILED SEGMENT TABLE
    # ================================================
    st.subheader("Segment Details")

    display_df = seg_summary[
        ['Segment_Name', 'Count', 'Churn_Rate_Pct'] + features
    ].copy()
    display_df.columns = (
        ['Segment', 'Customers', 'Churn Rate (%)'] + features
    )

    st.dataframe(
        display_df.style.background_gradient(
            subset=['Churn Rate (%)'],
            cmap='RdYlGn_r'
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ================================================
    # CHARTS
    # ================================================
    st.subheader("Visual Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Churn rate by segment bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        colors  = [
            risk_colors.get(name, '#718096')
            for name in seg_summary['Segment_Name']
        ]
        bars = ax.bar(
            range(len(seg_summary)),
            seg_summary['Churn_Rate_Pct'],
            color=colors, alpha=0.85
        )
        ax.set_xticks(range(len(seg_summary)))
        ax.set_xticklabels(
            [f"Seg {i+1}" for i in range(len(seg_summary))],
            fontsize=9
        )
        ax.set_ylabel('Churn Rate (%)')
        ax.set_title('Churn Rate by Segment',
                     fontweight='bold')
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, seg_summary['Churn_Rate_Pct']):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{val}%',
                ha='center', fontsize=9, fontweight='bold'
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # Customer count pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(
            seg_summary['Count'],
            labels=[
                f"Seg {i+1}\n({int(c):,})"
                for i, c in enumerate(seg_summary['Count'])
            ],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 9}
        )
        ax.set_title('Customer Distribution by Segment',
                     fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Scatter plot — first 2 features
    if len(features) >= 2:
        st.subheader(
            f"Segment Scatter — {features[0]} vs {features[1]}"
        )
        fig, ax = plt.subplots(figsize=(10, 5))

        for seg_id in seg_df['Segment'].unique():
            mask   = seg_df['Segment'] == seg_id
            name   = name_map.get(seg_id, f'Segment {seg_id}')
            color  = list(risk_colors.values())[
                seg_id % len(risk_colors)
            ]
            sample = seg_df[mask].sample(
                min(300, mask.sum()), random_state=42
            )
            ax.scatter(
                sample[features[0]],
                sample[features[1]],
                c=color, alpha=0.5,
                label=f"Seg {seg_id+1}: {name[:25]}",
                s=20
            )

        ax.set_xlabel(features[0], fontsize=11)
        ax.set_ylabel(features[1], fontsize=11)
        ax.set_title(
            f'Customer Segments — {features[0]} vs {features[1]}',
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ================================================
    # ACTION RECOMMENDATIONS PER SEGMENT
    # ================================================
    st.subheader("Recommended Actions per Segment")

    actions = {
        'High Risk — Immediate Action'  : {
            'icon'  : '🚨',
            'color' : '#E24B4A',
            'action': [
                "Call every customer in this segment within 24 hours",
                "Offer a significant loyalty discount (20-30%)",
                "Assign dedicated retention agent",
                "Escalate to management for special approval"
            ]
        },
        'Medium Risk — Monitor Closely' : {
            'icon'  : '⚠️',
            'color' : '#EF9F27',
            'action': [
                "Send personalized email with special offer",
                "Schedule check-in call within 1 week",
                "Offer minor discount or service upgrade",
                "Monitor for 30 days — escalate if no improvement"
            ]
        },
        'Low Risk — Stable'             : {
            'icon'  : '✅',
            'color' : '#1A7A4A',
            'action': [
                "Maintain current service quality",
                "Send monthly satisfaction survey",
                "Consider upsell to premium tier",
                "Routine monitoring — quarterly review"
            ]
        },
        'Very Low Risk — Loyal Champions': {
            'icon'  : '⭐',
            'color' : '#0C447C',
            'action': [
                "Enroll in VIP loyalty rewards program",
                "Ask for testimonial or referral",
                "Offer exclusive early access to new features",
                "Priority customer support line"
            ]
        },
    }

    for _, row in seg_summary.iterrows():
        name   = row['Segment_Name']
        info   = actions.get(name, {
            'icon'  : 'ℹ️',
            'color' : '#718096',
            'action': ["Analyse this segment further to define actions."]
        })
        with st.expander(
            f"{info['icon']} {name} — "
            f"{int(row['Count']):,} customers, "
            f"{row['Churn_Rate_Pct']}% churn rate"
        ):
            for act in info['action']:
                st.write(f"• {act}")

    st.divider()

    # ================================================
    # DOWNLOAD SEGMENTED DATA
    # ================================================
    st.subheader("Download Segmented Customer Data")

    download_df = seg_df[
        features + ['Churn_num', 'Segment', 'Segment_Name']
    ].copy()
    download_df.columns = (
        features + ['Churned', 'Segment_ID', 'Segment_Name']
    )

    st.download_button(
        label="Download Segmentation Results as CSV",
        data=download_df.to_csv(index=False),
        file_name=f"churnguard_{industry}_segments.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        f"K-Means clustering with K={n_clusters} segments. "
        f"Features used: {', '.join(features)}. "
        "Segments are sorted by churn rate — highest risk first."
    )