# --- Model Export Code (Decision Tree) ---

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load dataset
data = pd.read_csv("cleaned_heart_disease.csv")

# 2. Split features & target
X = data.drop("target", axis=1)
y = data["target"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Build pipeline (scaler + decision tree)
pipeline = Pipeline([
    ("scaler", StandardScaler()), 
    ("classifier", DecisionTreeClassifier(
        criterion="gini", 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1,
        random_state=42
    ))
])

# 5. Train model
pipeline.fit(X_train, y_train)

# 6. Evaluate model
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Export pipeline (preprocessing + model)
joblib.dump(pipeline, "heart_disease_dt_model.pkl")

print("[OK] Decision Tree model exported as 'heart_disease_dt_model.pkl'")
