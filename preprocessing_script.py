# ============================================================
# Level 1 - Task 3: Exploratory Data Analysis (EDA)
# Dataset: iris.csv
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Step 1: Data Collection 
# ============================================================

# Load dataset
df = pd.read_csv("iris.csv")

# Display first few rows
print("First 5 rows:")
print(df.head())

# ============================================================
# Step 2: Data Description & Structure Analysis 
# ============================================================

# Check number of records and attributes
print("\nShape (rows, columns):")
print(df.shape)

# Check data types
print("\nData types:")
print(df.dtypes)

# Identify unique values in the target column (species)
print("\nUnique species:")
print(df['species'].nunique())
print(df['species'].unique())

# ============================================================
# Step 3: Descriptive Statistics 
# ============================================================

# Basic statistical summary
print("\nDescriptive Statistics:")
print(df.describe())

# Compute mean, median, and standard deviation for each column
mean_value = df['sepal_length'].mean()
median_value = df['sepal_length'].median()
std_dev = df['sepal_length'].std()
print(f"\nSepal Length -> Mean: {mean_value}, Median: {median_value}, Standard Deviation: {std_dev}")

mean_value = df['sepal_width'].mean()
median_value = df['sepal_width'].median()
std_dev = df['sepal_width'].std()
print(f"Sepal Width  -> Mean: {mean_value}, Median: {median_value}, Standard Deviation: {std_dev}")

mean_value = df['petal_length'].mean()
median_value = df['petal_length'].median()
std_dev = df['petal_length'].std()
print(f"Petal Length -> Mean: {mean_value}, Median: {median_value}, Standard Deviation: {std_dev}")

mean_value = df['petal_width'].mean()
median_value = df['petal_width'].median()
std_dev = df['petal_width'].std()
print(f"Petal Width  -> Mean: {mean_value}, Median: {median_value}, Standard Deviation: {std_dev}")

# Class distribution
print("\nClass distribution (species):")
print(df['species'].value_counts())

# ============================================================
# Step 4: Visualizations 
# ============================================================

# --- Histograms ---
df['sepal_length'].hist()
plt.title('Sepal Length Distribution')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()

df['sepal_width'].hist()
plt.title('Sepal Width Distribution')
plt.xlabel('Sepal Width')
plt.ylabel('Frequency')
plt.show()

df['petal_length'].hist()
plt.title('Petal Length Distribution')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.show()

df['petal_width'].hist()
plt.title('Petal Width Distribution')
plt.xlabel('Petal Width')
plt.ylabel('Frequency')
plt.show()

# --- Scatter Plots ---
plt.scatter(df['sepal_length'], df['sepal_width'])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width")
plt.show()

plt.scatter(df['petal_length'], df['petal_width'])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal Length vs Petal Width")
plt.show()

# --- Box Plots ---
df.boxplot(column='sepal_length', by='species')
plt.title('Sepal Length by Species')
plt.suptitle('')
plt.show()

df.boxplot(column='petal_length', by='species')
plt.title('Petal Length by Species')
plt.suptitle('')
plt.show()

# ============================================================
# Step 5: Correlation Analysis 
# ============================================================

# Compute correlation matrix
correlation_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize correlation using a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

# ============================================================
# Step 6: Data Quality Assessment 
# ============================================================

# 1. Handling Missing Values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values with column mean (if any)
df_filled = df.fillna(df.mean(numeric_only=True))

# 2. Handling Duplicates
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# 3. Handling Outliers using IQR method
print("\nOutlier detection using IQR method:")

Q1 = df["sepal_length"].quantile(0.25)
Q3 = df["sepal_length"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["sepal_length"] < (Q1 - 1.5 * IQR)) | (df["sepal_length"] > (Q3 + 1.5 * IQR))]
print(f"Sepal Length outliers: {len(outliers)}")

Q1 = df["sepal_width"].quantile(0.25)
Q3 = df["sepal_width"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["sepal_width"] < (Q1 - 1.5 * IQR)) | (df["sepal_width"] > (Q3 + 1.5 * IQR))]
print(f"Sepal Width outliers: {len(outliers)}")

Q1 = df["petal_length"].quantile(0.25)
Q3 = df["petal_length"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["petal_length"] < (Q1 - 1.5 * IQR)) | (df["petal_length"] > (Q3 + 1.5 * IQR))]
print(f"Petal Length outliers: {len(outliers)}")

Q1 = df["petal_width"].quantile(0.25)
Q3 = df["petal_width"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["petal_width"] < (Q1 - 1.5 * IQR)) | (df["petal_width"] > (Q3 + 1.5 * IQR))]
print(f"Petal Width outliers: {len(outliers)}")

print("\n[EDA Complete]")
