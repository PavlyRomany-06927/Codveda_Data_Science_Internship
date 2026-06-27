# ============================================================
# Level 2 - Task 3: K-Means Clustering
# Dataset: iris.csv
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================
# Step 1: Load and Explore the Dataset 
# ============================================================

# Load dataset
df = pd.read_csv("iris.csv")

# Display first few rows
print("First 5 rows:")
print(df.head())

# Check shape
print("\nShape (rows, columns):")
print(df.shape)

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# ============================================================
# Step 2: Preprocessing Data 
# Feature Selection: use only numerical columns for clustering
# ============================================================

# Select numerical features (drop species - clustering is unsupervised)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

print("\nFeatures selected for clustering:")
print(X.head())

# Normalize the data using MinMaxScaler 
cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
scaler = MinMaxScaler()
X[cols] = scaler.fit_transform(X[cols])

print("\nAfter MinMax Normalization:")
print(X.head())

# ============================================================
# Step 3: Elbow Method - Find Optimal K 
# ============================================================

print("\n--- Elbow Method: Inertia for each K ---")
inertia = []
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    print(f"  K={k:2d}  ->  Inertia: {kmeans.inertia_:.4f}")

# Plot Elbow Curve
plt.plot(k_values, inertia, marker='o', color='blue')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Optimal K")
plt.show()

# ============================================================
# Step 4: Apply K-Means Clustering 
# Based on elbow method, K=3 is the optimal number
# (matches the 3 real species in the dataset)
# ============================================================

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Add cluster labels to the dataframe
df['Cluster'] = kmeans.labels_

print("\nCluster assignments (first 10 rows):")
print(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species', 'Cluster']].head(10))

# Count of samples per cluster
print("\nSamples per cluster:")
print(df['Cluster'].value_counts())

# ============================================================
# Step 5: Evaluate Clusters
# ============================================================

# Silhouette Score
sil_score = silhouette_score(X, kmeans.labels_)
print(f"\nSilhouette Score (K=3): {sil_score:.4f}")
print("(Closer to 1.0 = better clustering)")

# Inertia
print(f"Inertia (K=3): {kmeans.inertia_:.4f}")
print("(Lower inertia = tighter clusters)")

# Compare silhouette scores for different K values
print("\n--- Silhouette Scores for different K values ---")
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    score = silhouette_score(X, km.labels_)
    print(f"  K={k}  ->  Silhouette Score: {score:.4f}")

# ============================================================
# Step 6: Visualize Clusters 
# Scatter plot using petal_length vs petal_width
# (best features for separation as found in EDA task)
# ============================================================

colors = {0: 'red', 1: 'blue', 2: 'green'}
cluster_colors = df['Cluster'].map(colors)

plt.scatter(df['petal_length'], df['petal_width'], c=cluster_colors)
plt.xlabel("Petal Length (normalized)")
plt.ylabel("Petal Width (normalized)")
plt.title("K-Means Clustering (K=3) - Petal Features")
plt.show()

# Scatter plot using sepal features
plt.scatter(df['sepal_length'], df['sepal_width'], c=cluster_colors)
plt.xlabel("Sepal Length (normalized)")
plt.ylabel("Sepal Width (normalized)")
plt.title("K-Means Clustering (K=3) - Sepal Features")
plt.show()

# ============================================================
# Step 7: Cluster Summary
# ============================================================

print("\n--- Cluster Summary ---")
print(df.groupby('Cluster')[['sepal_length', 'sepal_width',
                              'petal_length', 'petal_width']].mean())

print("\n--- Species distribution per Cluster ---")
print(df.groupby(['Cluster', 'species']).size())

print("\n[Clustering Complete]")
