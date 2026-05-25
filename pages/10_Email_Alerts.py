# pages/10_Email_Alerts.py — ChurnGuard Pro — Automated Email Alert System

import streamlit as st
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

st.set_page_config(page_title="Email Alerts", page_icon="📧", layout="wide")

st.markdown("## 📧 Automated Email Alert System")
st.markdown("Configure the system to automatically notify your retention team when high-risk churn customers are detected.")
st.markdown("---")

# Layout Split
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("⚙️ 1. Configure Sender Mail (Gmail)")
    st.caption("Note: For security in production, use an App Password, not your raw master password.")
    
    sender_email = st.text_input("Sender Gmail Address", placeholder="your_company_alerts@gmail.com")
    app_password = st.text_input("Gmail App Password", type="password", placeholder="xxxx xxxx xxxx xxxx")
    
    st.subheader("📥 2. Configure Recipient Team")
    team_email = st.text_input("Retention Team Email", placeholder="retention_team@yourcompany.com")
    risk_threshold = st.slider("Trigger Alert for Churn Probability ≥", 0.50, 0.95, 0.75, step=0.05)

with col2:
    st.subheader("📁 3. Upload Predicted Dataset")
    uploaded_file = st.file_uploader("Upload the predicted CSV from your Bulk Upload page", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Check if the file has the necessary prediction columns
        probability_col = [col for col in df.columns if 'probability' in col.lower() or 'prob' in col.lower()]
        
        if not probability_col:
            st.error("❌ Invalid File: Could not find a 'Churn_Probability' column. Please process this file through the Bulk Upload page first.")
        else:
            prob_name = probability_col[0]
            # Filter out high-risk customers based on the slider selection
            high_risk_df = df[df[prob_name] >= risk_threshold]
            
            st.metric("🚨 Critical Customers Identified", len(high_risk_df))
            
            if len(high_risk_df) > 0:
                st.dataframe(high_risk_df.head(10), use_container_width=True)
                
                # Activation Button
                if st.button("🚀 Dispatch Alerts to Team", use_container_width=True):
                    if not sender_email or not app_password or not team_email:
                        st.warning("⚠️ Please fill out all email configuration settings on the left panel before sending.")
                    else:
                        with st.spinner("Establishing secure SMTP connection..."):
                            try:
                                # Create Email Container
                                msg = MIMEMultipart()
                                msg['From'] = sender_email
                                msg['To'] = team_email
                                msg['Subject'] = f"🚨 UGENT: {len(high_risk_df)} High-Risk Churn Customers Detected"
                                
                                # Draft HTML Body for a professional enterprise look
                                html_body = f"""
                                <html>
                                <body style='font-family: Arial, sans-serif; color: #333;'>
                                    <h2 style='color: #d9534f;'>ChurnGuard Pro — Automated Risk Alert</h2>
                                    <p>Hello Retention Team,</p>
                                    <p>Our machine learning engine has detected <strong>{len(high_risk_df)} customers</strong> whose churn risk has exceeded your threshold of {risk_threshold:.0%}.</p>
                                    <p>Please log into the management portal immediately to review their service history and apply preventative retention offers.</p>
                                    <br>
                                    <p style='font-size: 0.85rem; color: #777;'>Generated automatically by ChurnGuard Pro Core Ecosystem.</p>
                                </body>
                                </html>
                                """
                                msg.attach(MIMEText(html_body, 'html'))
                                
                                # Connect to Google's Secure SMTP Server
                                server = smtplib.SMTP('smtp.gmail.com', 547 or 587)
                                server.starttls()  # Upgrade connection to secure encrypted TLS
                                server.login(sender_email, app_password)
                                server.sendmail(sender_email, team_email, msg.as_string())
                                server.quit()
                                
                                st.success(f"✅ Success! Notifications securely broadcasted to {team_email}.")
                                
                            except Exception as e:
                                st.error(f"❌ Mail System Failure: {str(e)}")
            else:
                st.info("✅ Excellent news! No customers exceed the selected churn risk threshold.")