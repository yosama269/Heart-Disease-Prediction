import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# ==============================
# 1. Load cleaned dataset
# ==============================
df = pd.read_csv("cleaned_heart_disease.csv")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# ==============================
# 2. Random Forest Feature Importance
# ==============================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# Get feature importance
importances = rf.feature_importances_
feat_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feat_importance_df = feat_importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_importance_df)
plt.title("Feature Importance (Random Forest)")
plt.show()

print("\nTop Features by Random Forest:\n", feat_importance_df.head(10))

# ==============================
# 3. Recursive Feature Elimination (RFE)
# ==============================
lr = LogisticRegression(max_iter=1000, solver='liblinear')
rfe = RFE(estimator=lr, n_features_to_select=10)  # Select top 10 features
rfe.fit(X, y)

rfe_selected = X.columns[rfe.support_]
print("\nSelected Features by RFE:\n", rfe_selected)

# ==============================
# 4. Chi-Square Test
# ==============================
# Scale features to [0,1] since chi2 requires non-negative values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

chi2_selector = SelectKBest(score_func=chi2, k=10)  # Select top 10
chi2_selector.fit(X_scaled, y)

chi2_selected = X.columns[chi2_selector.get_support()]
chi2_scores = chi2_selector.scores_

chi2_df = pd.DataFrame({"Feature": X.columns, "Chi2 Score": chi2_scores})
chi2_df = chi2_df.sort_values(by="Chi2 Score", ascending=False)

print("\nTop Features by Chi-Square Test:\n", chi2_df.head(10))

# ==============================
# 5. Final Feature Selection
# ==============================
# Union of top features from all 3 methods
final_features = list(set(feat_importance_df.head(12)["Feature"]) | 
                      set(rfe_selected) | 
                      set(chi2_selected))

print("\nFinal Selected Features for Modeling:\n", final_features)

# Reduced dataset
X_reduced = X[final_features]
reduced_df = pd.concat([X_reduced, y], axis=1)
reduced_df.to_csv("reduced_heart_disease.csv", index=False)

print("\nReduced dataset saved as reduced_heart_disease.csv")
print("Shape of reduced dataset:", reduced_df.shape)
