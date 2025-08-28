import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------
# Step 1: Data Preprocessing
# ---------------------------

# Load dataset
df = pd.read_csv("HeartDiseaseTrain-Test.csv")
print("Dataset shape:", df.shape)
print(df.head())

# Check missing values
print("\nMissing values per column:\n", df.isnull().sum())

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)
print("\nAfter encoding, dataset shape:", df_encoded.shape)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded.drop("target", axis=1))
y = df_encoded["target"]

# Save cleaned dataset
cleaned_df = pd.DataFrame(X_scaled, columns=df_encoded.drop("target", axis=1).columns)
cleaned_df["target"] = y.values
cleaned_df.to_csv("cleaned_heart_disease.csv", index=False)
print("\nCleaned dataset saved as cleaned_heart_disease.csv")

# ---------------------------
# Step 2: PCA
# ---------------------------

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Print explained variance
print("\nExplained variance ratio per component:\n", explained_variance)
print("\nCumulative variance explained:\n", cumulative_variance)

# Plot cumulative variance
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--')
plt.title("Cumulative Variance Explained by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.grid(True)
plt.show()

# Plot first two PCA components
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA - First Two Principal Components")
plt.colorbar(label="Target")
plt.show()

print("\nPCA transformation completed.")
