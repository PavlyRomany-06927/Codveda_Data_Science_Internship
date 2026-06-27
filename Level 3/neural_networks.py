# ============================================================
# Level 3 - Task 3: Neural Networks with TensorFlow/Keras
# Dataset: churn-bigml-80.csv (Structured Data Input Matrix)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ============================================================
# Step 1: Load and Preprocess Structured Matrix Data
# ============================================================

df = pd.read_csv("churn-bigml-80.csv")

# Encode categories to binary flags
le = LabelEncoder()
df['International plan'] = le.fit_transform(df['International plan'])
df['Voice mail plan'] = le.fit_transform(df['Voice mail plan'])
df['Churn'] = le.fit_transform(df['Churn'])

# Drop structural ID codes
df = df.drop(columns=['State', 'Area code'])

X = df.drop(columns=['Churn']).values
y = df['Churn'].values

# Split data into training and validation folds
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical input matrices (Essential step for Neural Networks gradients)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ============================================================
# Step 2: Design Feed-Forward Neural Network Architecture
# ============================================================

model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),  # Input Layer
    Dropout(0.2),                                                   # Prevents Overfitting
    Dense(16, activation='relu'),                                   # Hidden Layer
    Dense(1, activation='sigmoid')                                  # Output Layer for Binary Classification
])

# Compile optimization algorithms and metrics
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================================================
# Step 3: Train Network with Backpropagation
# ============================================================

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=32,
                    verbose=1)

# ============================================================
# Step 4: Evaluate Performance and Loss Curves
# ============================================================

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Final Validation Loss: {val_loss:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")

# Generate simple ASCII learning progress graph
print("\n--- Training Logs (Final Epochs Summary) ---")
print(f"Training Accuracy Peak: {history.history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy Peak: {history.history['val_accuracy'][-1]:.4f}")
print("\nNeural Network Execution Completed Successfully!")
