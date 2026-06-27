# ============================================================
# Level 1 - Task 2: Data Cleaning and Preprocessing
# Dataset: churn-bigml-80.csv
# ============================================================

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ============================================================
# Step 1: Load Dataset 
# ============================================================

# Load dataset
df = pd.read_csv("churn-bigml-80.csv")

# Display first few rows
print("First 5 rows:")
print(df.head())

# Check number of records and attributes
print("\nShape (rows, columns):")
print(df.shape)

# Check data types
print("\nData types:")
print(df.dtypes)

# ============================================================
# Step 2: Handle Missing Data 
# ============================================================

print("\nMissing values per column:")
print(df.isnull().sum())

# Remove rows with missing values (if any)
df_cleaned = df.dropna()
print("\nShape after removing rows with missing values:")
print(df_cleaned.shape)

# Fill missing values with column mean (alternative approach)
# df_filled = df.fillna(df.mean())

# ============================================================
# Step 3: Handle Duplicates 
# ============================================================

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# ============================================================
# Step 4: Detect and Handle Outliers 
# ============================================================

print("\nOutlier detection using IQR method:")

Q1 = df["Total day minutes"].quantile(0.25)
Q3 = df["Total day minutes"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["Total day minutes"] < (Q1 - 1.5 * IQR)) | (df["Total day minutes"] > (Q3 + 1.5 * IQR))]
print(f"Total day minutes outliers: {len(outliers)}")

Q1 = df["Total eve minutes"].quantile(0.25)
Q3 = df["Total eve minutes"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["Total eve minutes"] < (Q1 - 1.5 * IQR)) | (df["Total eve minutes"] > (Q3 + 1.5 * IQR))]
print(f"Total eve minutes outliers: {len(outliers)}")

Q1 = df["Customer service calls"].quantile(0.25)
Q3 = df["Customer service calls"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["Customer service calls"] < (Q1 - 1.5 * IQR)) | (df["Customer service calls"] > (Q3 + 1.5 * IQR))]
print(f"Customer service calls outliers: {len(outliers)}")

# Remove outliers from Customer service calls (most extreme column)
Q1 = df["Customer service calls"].quantile(0.25)
Q3 = df["Customer service calls"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["Customer service calls"] >= (Q1 - 1.5 * IQR)) & (df["Customer service calls"] <= (Q3 + 1.5 * IQR))]
print("\nShape after removing Customer service calls outliers:", df.shape)

# ============================================================
# Step 5: Convert Categorical Variables to Numerical
#         Using Label Encoding
# ============================================================

print("\nCategorical columns before encoding:")
print(df[['International plan', 'Voice mail plan', 'Churn']].head())

# Label encode 'International plan'  (Yes=1, No=0)
le = LabelEncoder()
df['International plan'] = le.fit_transform(df['International plan'])

# Label encode 'Voice mail plan'
df['Voice mail plan'] = le.fit_transform(df['Voice mail plan'])

# Label encode 'Churn'  (True=1, False=0)
df['Churn'] = le.fit_transform(df['Churn'])

# Drop 'State' and 'Area code' - not useful for modeling
df = df.drop(columns=['State', 'Area code'])

print("\nAfter label encoding:")
print(df[['International plan', 'Voice mail plan', 'Churn']].head())

# ============================================================
# Step 6: Normalize Numerical Data 
#         Using MinMaxScaler (0 to 1 range)
# ============================================================

# Columns to normalize
cols = ['Account length', 'Total day minutes', 'Total day calls',
        'Total day charge', 'Total eve minutes', 'Total eve calls',
        'Total eve charge', 'Total night minutes', 'Total night calls',
        'Total night charge', 'Total intl minutes', 'Total intl calls',
        'Total intl charge', 'Customer service calls']

scaler = MinMaxScaler()
df[cols] = scaler.fit_transform(df[cols])

print("\nAfter MinMax Normalization (first 5 rows):")
print(df[cols].head())

# ============================================================
# Step 7: Final Clean Dataset Summary
# ============================================================

print("\nFinal dataset shape:", df.shape)
print("\nFinal data types:")
print(df.dtypes)
print("\nFinal dataset preview:")
print(df.head())

# Save cleaned dataset
df.to_csv("churn_cleaned.csv", index=False)
print("\nCleaned dataset saved as: churn_cleaned.csv")

print("\n[Preprocessing Complete]")
