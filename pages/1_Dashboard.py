# pages/1_Dashboard.py — Multi-Industry Analytics Dashboard

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard", layout="wide")

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.page_link("app.py", label="Go to Login page")
    st.stop()

st.title("Analytics Dashboard")
st.write("Overview of customer churn data across all 3 industries.")
st.divider()

# ================================================
# LOAD ALL 3 DATASETS
# ================================================
@st.cache_data
def load_telecom():
    df = pd.read_csv(
        'datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce'
    )
    df['Churn_num'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

@st.cache_data
def load_banking():
    df = pd.read_csv('datasets/Churn_Modelling.csv')
    df['Churn_num'] = df['Exited']
    return df

@st.cache_data
def load_ecommerce():
    df = pd.read_excel(
        'datasets/E Commerce Dataset.xlsx',
        sheet_name='E Comm'
    )
    df['Churn_num'] = df['Churn']
    return df

# Load with error handling
datasets = {}
errors   = {}

try:
    datasets['telecom'] = load_telecom()
except Exception as e:
    errors['telecom'] = str(e)

try:
    datasets['banking'] = load_banking()
except Exception as e:
    errors['banking'] = str(e)

try:
    datasets['ecommerce'] = load_ecommerce()
except Exception as e:
    errors['ecommerce'] = str(e)

# ================================================
# TOP SUMMARY — ALL INDUSTRIES COMBINED
# ================================================
st.subheader("Platform Overview — All Industries")

total_customers = sum(len(df) for df in datasets.values())
total_churned   = sum(
    int(df['Churn_num'].sum()) for df in datasets.values()
)
total_stayed    = total_customers - total_churned
churn_rate      = round(total_churned / total_customers * 100, 1) \
                  if total_customers > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Customers",   f"{total_customers:,}")
c2.metric("Total Churned",     f"{total_churned:,}",
          delta=f"-{churn_rate}%", delta_color="inverse")
c3.metric("Total Retained",    f"{total_stayed:,}")
c4.metric("Overall Churn Rate",f"{churn_rate}%")
c5.metric("Industries",        "3")

st.divider()

# ================================================
# INDUSTRY TABS
# ================================================
st.subheader("Industry-wise Analysis")

tab1, tab2, tab3 = st.tabs([
    "📡 Telecom",
    "🏦 Banking",
    "🛒 E-Commerce"
])

# ---- TELECOM TAB ----
with tab1:
    if 'telecom' in errors:
        st.error(f"Could not load Telecom data: {errors['telecom']}")
    else:
        df = datasets['telecom']
        churned   = int(df['Churn_num'].sum())
        stayed    = len(df) - churned
        churn_pct = round(churned / len(df) * 100, 1)
        avg_bill  = round(df['MonthlyCharges'].mean(), 2)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers", f"{len(df):,}")
        m2.metric("Churned",         f"{churned:,}")
        m3.metric("Churn Rate",      f"{churn_pct}%")
        m4.metric("Avg Monthly Bill",f"${avg_bill}")

        # Charts
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Churn by Contract Type**")
            ct = df.groupby('Contract')['Churn_num'].mean() * 100
            ct = ct.round(1)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(ct.index, ct.values,
                   color=['#E24B4A', '#EF9F27', '#639922'],
                   alpha=0.85)
            ax.set_ylabel('Churn Rate (%)')
            ax.set_ylim(0, 55)
            for i, v in enumerate(ct.values):
                ax.text(i, v + 0.5, f'{v}%',
                        ha='center', fontsize=9, fontweight='bold')
            ax.tick_params(axis='x', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.write("**Churn by Internet Service**")
            it = df.groupby('InternetService')['Churn_num'].mean() * 100
            it = it.round(1)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(it.index, it.values,
                   color=['#E24B4A', '#639922', '#EF9F27'],
                   alpha=0.85)
            ax.set_ylabel('Churn Rate (%)')
            ax.set_ylim(0, 55)
            for i, v in enumerate(it.values):
                ax.text(i, v + 0.5, f'{v}%',
                        ha='center', fontsize=9, fontweight='bold')
            ax.tick_params(axis='x', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col3:
            st.write("**Tenure Distribution**")
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(df[df['Churn_num']==0]['tenure'],
                    bins=20, alpha=0.6,
                    color='#639922', label='Stayed')
            ax.hist(df[df['Churn_num']==1]['tenure'],
                    bins=20, alpha=0.6,
                    color='#E24B4A', label='Churned')
            ax.set_xlabel('Tenure (months)')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.write("**Dataset Preview**")
        st.dataframe(df.head(5), use_container_width=True)

# ---- BANKING TAB ----
with tab2:
    if 'banking' in errors:
        st.error(f"Could not load Banking data: {errors['banking']}")
    else:
        df = datasets['banking']
        churned   = int(df['Churn_num'].sum())
        stayed    = len(df) - churned
        churn_pct = round(churned / len(df) * 100, 1)
        avg_bal   = round(df['Balance'].mean(), 2)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers",  f"{len(df):,}")
        m2.metric("Churned",          f"{churned:,}")
        m3.metric("Churn Rate",       f"{churn_pct}%")
        m4.metric("Avg Balance",      f"${avg_bal:,.0f}")

        # Charts
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Churn by Geography**")
            geo = df.groupby('Geography')['Churn_num'].mean() * 100
            geo = geo.round(1)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(geo.index, geo.values,
                   color=['#3B82F6', '#8B5CF6', '#10B981'],
                   alpha=0.85)
            ax.set_ylabel('Churn Rate (%)')
            ax.set_ylim(0, 35)
            for i, v in enumerate(geo.values):
                ax.text(i, v + 0.3, f'{v}%',
                        ha='center', fontsize=9, fontweight='bold')
            ax.tick_params(axis='x', labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.write("**Churn by Number of Products**")
            prod = df.groupby('NumOfProducts')['Churn_num'].mean() * 100
            prod = prod.round(1)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar([str(x) for x in prod.index],
                   prod.values,
                   color=['#639922','#EF9F27','#E24B4A','#8B5CF6'],
                   alpha=0.85)
            ax.set_xlabel('Number of Products')
            ax.set_ylabel('Churn Rate (%)')
            for i, v in enumerate(prod.values):
                ax.text(i, v + 0.5, f'{v}%',
                        ha='center', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col3:
            st.write("**Age Distribution**")
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(df[df['Churn_num']==0]['Age'],
                    bins=20, alpha=0.6,
                    color='#639922', label='Stayed')
            ax.hist(df[df['Churn_num']==1]['Age'],
                    bins=20, alpha=0.6,
                    color='#E24B4A', label='Churned')
            ax.set_xlabel('Age')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.write("**Dataset Preview**")
        st.dataframe(df.head(5), use_container_width=True)

# ---- ECOMMERCE TAB ----
with tab3:
    if 'ecommerce' in errors:
        st.error(
            f"Could not load E-Commerce data: {errors['ecommerce']}"
        )
    else:
        df = datasets['ecommerce']
        churned   = int(df['Churn_num'].sum())
        stayed    = len(df) - churned
        churn_pct = round(churned / len(df) * 100, 1)
        avg_sat   = round(df['SatisfactionScore'].mean(), 1)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers",   f"{len(df):,}")
        m2.metric("Churned",           f"{churned:,}")
        m3.metric("Churn Rate",        f"{churn_pct}%")
        m4.metric("Avg Satisfaction",  f"{avg_sat}/5")

        # Charts
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Churn by Login Device**")
            dev = df.groupby('PreferredLoginDevice')['Churn_num'].mean() * 100
            dev = dev.round(1)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(dev.index, dev.values,
                   color=['#3B82F6','#8B5CF6','#10B981'],
                   alpha=0.85)
            ax.set_ylabel('Churn Rate (%)')
            for i, v in enumerate(dev.values):
                ax.text(i, v + 0.3, f'{v}%',
                        ha='center', fontsize=9, fontweight='bold')
            ax.tick_params(axis='x', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.write("**Churn by Satisfaction Score**")
            sat = df.groupby('SatisfactionScore')['Churn_num'].mean() * 100
            sat = sat.round(1)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar([str(x) for x in sat.index],
                   sat.values,
                   color=['#639922','#8B5CF6',
                          '#EF9F27','#E24B4A','#3B82F6'],
                   alpha=0.85)
            ax.set_xlabel('Satisfaction Score (1-5)')
            ax.set_ylabel('Churn Rate (%)')
            for i, v in enumerate(sat.values):
                ax.text(i, v + 0.3, f'{v}%',
                        ha='center', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col3:
            st.write("**Churn by Complain Status**")
            comp = df.groupby('Complain')['Churn_num'].mean() * 100
            comp = comp.round(1)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(['No Complain', 'Has Complain'],
                   comp.values,
                   color=['#639922', '#E24B4A'],
                   alpha=0.85)
            ax.set_ylabel('Churn Rate (%)')
            ax.set_ylim(0, 100)
            for i, v in enumerate(comp.values):
                ax.text(i, v + 1, f'{v}%',
                        ha='center', fontsize=10, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.write("**Dataset Preview**")
        st.dataframe(df.head(5), use_container_width=True)

st.divider()

# ================================================
# MODEL PERFORMANCE COMPARISON
# ================================================
st.subheader("Model Performance — All Industries")

try:
    results_df = pd.read_csv('models/industry_results.csv',
                             index_col=0)
    st.dataframe(
        results_df.style.highlight_max(
            subset=['Accuracy','Precision',
                    'Recall','F1-Score','AUC'],
            color='#C0DD97'
        ),
        use_container_width=True
    )
except Exception:
    try:
        comp = pd.read_csv('model_comparison.csv')
        st.dataframe(comp, use_container_width=True)
    except Exception:
        st.info("Run multi_train.py to see model comparison.")