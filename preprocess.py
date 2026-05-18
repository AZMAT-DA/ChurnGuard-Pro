# preprocess.py - FIXED VERSION for Phase 1

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

print("=" * 50)
print("STEP 1: LOADING RAW DATA")
print("=" * 50)

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Raw data shape: {df.shape}")

# =============================================
# STEP 2: FIX TOTALCHARGES COLUMN
# =============================================
print("\nFixing TotalCharges column...")
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# =============================================
# STEP 3: DROP CUSTOMERID
# =============================================
df.drop('customerID', axis=1, inplace=True)

# =============================================
# STEP 4: ENCODE TARGET COLUMN
# =============================================
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# =============================================
# STEP 5: ENCODE BINARY COLUMNS
# =============================================
print("Encoding binary columns...")

# gender
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Yes/No columns — map directly to 1/0
yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in yes_no_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# SeniorCitizen is already 0/1 — leave it

# MultipleLines has 3 values — encode as number
df['MultipleLines'] = df['MultipleLines'].map({
    'No phone service': 0,
    'No': 1,
    'Yes': 2
})

# =============================================
# STEP 6: ONE-HOT ENCODE MULTI-CATEGORY COLUMNS
# =============================================
print("One-hot encoding multi-category columns...")

multi_cols = [
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod'
]

df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# =============================================
# STEP 7: SCALE NUMERICAL COLUMNS
# =============================================
print("Scaling numerical columns...")

scaler   = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols] = scaler.fit_transform(df[num_cols])

joblib.dump(scaler, 'scaler.pkl')

# =============================================
# STEP 8: FEATURE ENGINEERING
# =============================================
print("Creating new features...")

# Count total services per customer
service_cols = [c for c in df.columns if any(s in c for s in [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]) and '_Yes' in c]

if service_cols:
    df['total_services'] = df[service_cols].sum(axis=1)
else:
    # fallback if column names differ
    df['total_services'] = 0

# =============================================
# STEP 9: CONVERT ALL COLUMNS TO FLOAT
# =============================================
# This is the critical fix for PyTorch Neural Network
print("Converting all columns to float...")

# Convert boolean columns (from get_dummies) to int first, then float
bool_cols = df.select_dtypes(include='bool').columns.tolist()
if bool_cols:
    df[bool_cols] = df[bool_cols].astype(int)
    print(f"  Converted {len(bool_cols)} boolean columns to int")

# Convert everything to float64
df = df.astype(float)

# Verify no object columns remain
obj_cols = df.select_dtypes(include='object').columns.tolist()
if obj_cols:
    print(f"WARNING: Still has object columns: {obj_cols}")
else:
    print("  All columns are now numeric — Neural Network ready!")

# =============================================
# STEP 10: FINAL CHECK AND SAVE
# =============================================
print("\n" + "=" * 50)
print("FINAL DATA CHECK")
print("=" * 50)
print(f"Shape          : {df.shape}")
print(f"Missing values : {df.isnull().sum().sum()}")
print(f"Data types     : {df.dtypes.value_counts().to_dict()}")
print(f"\nColumns: {df.columns.tolist()}")

df.to_csv('cleaned_data.csv', index=False)
print("\nPreprocessing complete!")
print("Saved: cleaned_data.csv")
print("Saved: scaler.pkl")

# Show feature stats
print("\nTotal services distribution:")
print(df['total_services'].describe())