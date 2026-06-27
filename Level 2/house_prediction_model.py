# ============================================================
# Level 2 - Task 1: Predictive Modeling (Regression)
# Dataset: house Prediction Data Set.csv (Boston Housing)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# Step 1: Load Dataset 
# ============================================================

# Column names for Boston Housing dataset
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
        'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df = pd.read_csv("house Prediction Data Set.csv",
                 sep=r'\s+', header=None, names=cols, engine='python')

print("First 5 rows:")
print(df.head())

# ============================================================
# Step 2: Data Understanding 
# ============================================================

print("\nDescriptive Statistics:")
print(df.describe())

print("\nColumn names:")
print(df.columns.tolist())

print("\nShape (rows, columns):")
print(df.shape)

print("\nMissing values per column:")
print(df.isnull().sum())

# ============================================================
# Step 3: Select Features 
# Target: MEDV = Median house price (in $1000s)
# Best predictor from correlation: RM (avg number of rooms)
#                                  LSTAT (% lower status population)
# ============================================================

cdf = df[['RM', 'LSTAT', 'PTRATIO', 'MEDV']]
print("\nSelected features preview:")
print(cdf.head())

# ============================================================
# Step 4: Visualization - Feature vs Target 
# ============================================================

# RM vs House Price
plt.scatter(cdf['RM'], cdf['MEDV'])
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("House Price in $1000s (MEDV)")
plt.title("Rooms vs House Price")
plt.show()

# LSTAT vs House Price
plt.scatter(cdf['LSTAT'], cdf['MEDV'])
plt.xlabel("% Lower Status Population (LSTAT)")
plt.ylabel("House Price in $1000s (MEDV)")
plt.title("LSTAT vs House Price")
plt.show()

# ============================================================
# Step 5: Train / Test Split 
# ============================================================

# Simple Linear Regression using RM only 
train, test = train_test_split(cdf, test_size=0.2)

train_x = np.asanyarray(train[['RM']])
train_y = np.asanyarray(train[['MEDV']])
test_x  = np.asanyarray(test[['RM']])
test_y  = np.asanyarray(test[['MEDV']])

print("\nTraining set size:", len(train))
print("Testing set size:", len(test))

# ============================================================
# Step 6: Build Simple Linear Regression Model 
# ============================================================

regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

# ============================================================
# Step 7: Model Parameters 
# ============================================================

print("\nSimple Linear Regression (RM -> MEDV):")
print("Coefficient (Slope):", regr.coef_)
print("Intercept:", regr.intercept_)

# ============================================================
# Step 8: Plot Regression Line 
# ============================================================

plt.scatter(train_x, train_y, color='blue')
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("House Price in $1000s (MEDV)")
plt.title("Simple Linear Regression - Regression Line")
plt.show()

# ============================================================
# Step 9: Prediction and Evaluation 
# ============================================================

y_hat = regr.predict(test_x)

# Evaluation Metrics 
mse = mean_squared_error(test_y, y_hat)
mae = mean_absolute_error(test_y, y_hat)
r2  = r2_score(test_y, y_hat)

print("\n--- Simple Linear Regression Results (RM only) ---")
print("MSE:", mse)
print("MAE:", mae)
print("R2 Score:", r2)

# ============================================================
# Step 10: Multiple Linear Regression 
# Using RM + LSTAT + PTRATIO as features
# ============================================================

X = df[['RM', 'LSTAT', 'PTRATIO']]
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regr2 = linear_model.LinearRegression()
regr2.fit(X_train, y_train)
y_pred2 = regr2.predict(X_test)

mse2 = mean_squared_error(y_test, y_pred2)
mae2 = mean_absolute_error(y_test, y_pred2)
r2_2 = r2_score(y_test, y_pred2)

print("\n--- Multiple Linear Regression Results (RM + LSTAT + PTRATIO) ---")
print("Coefficients:", regr2.coef_)
print("Intercept:", regr2.intercept_)
print("MSE:", mse2)
print("MAE:", mae2)
print("R2 Score:", r2_2)

# ============================================================
# Step 11: Compare Both Models 
# ============================================================

print("\n--- Model Comparison ---")
print(f"Simple Linear Regression  -> MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
print(f"Multiple Linear Regression -> MSE: {mse2:.2f}, MAE: {mae2:.2f}, R2: {r2_2:.2f}")

if r2_2 > r2:
    print(">> Multiple Linear Regression performs better (higher R2 score)")
else:
    print(">> Simple Linear Regression performs better")

# ============================================================
# Step 12: Actual vs Predicted Plot
# ============================================================

plt.scatter(y_test, y_pred2, color='green')
plt.xlabel("Actual House Price (MEDV)")
plt.ylabel("Predicted House Price")
plt.title("Multiple Linear Regression - Actual vs Predicted")
plt.show()

print("\n[Regression Complete]")
