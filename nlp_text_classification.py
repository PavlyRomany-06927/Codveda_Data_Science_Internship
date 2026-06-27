# ============================================================
# Level 3 - Task 2: NLP Text Classification
# Dataset: Sentiment dataset.csv
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ============================================================
# Step 1: Load and Inspect Dataset
# ============================================================

df = pd.read_csv("Sentiment dataset.csv")

# Ensure column strings are clean
df['Text'] = df['Text'].astype(str).str.strip()
df['Sentiment'] = df['Sentiment'].astype(str).str.strip()

print(f"Dataset shape: {df.shape}")
print("\nClass distribution of Sentiment target:")
print(df['Sentiment'].value_counts())

# ============================================================
# Step 2: Text Preprocessing & Text Cleaning
# ============================================================

# Convert to lower case for uniformity
df['Cleaned_Text'] = df['Text'].str.lower()

X = df['Cleaned_Text']
y = df['Sentiment']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# Step 3: Feature Extraction (TF-IDF Numerical Representation)
# ============================================================

# Limit features to prevent matrix memory overflow while removing standard English stopwords
tfidf = TfidfVectorizer(max_features=2500, stop_words='english')

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Train feature matrix shape: {X_train_tfidf.shape}")

# ============================================================
# Step 4: Model Training (Naive Bayes Classifier)
# ============================================================

nlp_model = MultinomialNB()
nlp_model.fit(X_train_tfidf, y_train)
print("Classifier training complete.")

# ============================================================
# Step 5: Prediction & Advanced Performance Evaluation
# ============================================================

y_pred = nlp_model.predict(X_test_tfidf)

# Generate scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n================ NLP TEXT CLASSIFICATION REPORT ================")
print(f"Overall Accuracy:  {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall:    {recall:.4f}")
print(f"Weighted F1-Score:  {f1:.4f}")
print("=================================================================\n")
print(classification_report(y_test, y_pred))