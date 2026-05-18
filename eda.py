import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# --- Basic Info ---
print("Shape of data:", df.shape)       # How many rows and columns?
print("\nFirst 5 rows:")
print(df.head())                         # Preview the data

print("\nData types and missing values:")
print(df.info())                         # Check data types

print("\nBasic statistics:")
print(df.describe())                     # Min, max, mean for numbers

# --- Check target column (Churn) ---
print("\nChurn value counts:")
print(df['Churn'].value_counts())        # How many Yes vs No?

# --- Plot 1: Churn distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title('Churn Distribution')
plt.xlabel('Churn (Yes = Left, No = Stayed)')
plt.ylabel('Number of Customers')
plt.savefig('churn_distribution.png')
plt.show()
print("Churn plot saved.")

# --- Plot 2: Contract type vs Churn ---
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set1')
plt.title('Contract Type vs Churn')
plt.savefig('contract_churn.png')
plt.show()

# --- Plot 3: Monthly Charges vs Churn ---
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
plt.title('Monthly Charges vs Churn')
plt.savefig('monthly_charges_churn.png')
plt.show()

print("\nEDA complete!")