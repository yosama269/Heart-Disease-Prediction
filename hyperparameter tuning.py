# Step 6: Hyperparameter Tuning
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load cleaned dataset
df = pd.read_csv("cleaned_heart_disease.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------ Logistic Regression ------------------
log_reg = LogisticRegression(max_iter=5000)
param_grid_lr = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}
grid_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring="accuracy", n_jobs=-1)
grid_lr.fit(X_train, y_train)

print("Best Logistic Regression:", grid_lr.best_params_)
y_pred_lr = grid_lr.predict(X_test)
print("Accuracy (LogReg):", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# ------------------ Decision Tree ------------------
dt = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}
grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring="accuracy", n_jobs=-1)
grid_dt.fit(X_train, y_train)

print("Best Decision Tree:", grid_dt.best_params_)
y_pred_dt = grid_dt.predict(X_test)
print("Accuracy (Decision Tree):", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# ------------------ Random Forest ------------------
rf = RandomForestClassifier(random_state=42)
param_dist_rf = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}
rand_rf = RandomizedSearchCV(rf, param_dist_rf, n_iter=20, cv=5, scoring="accuracy", random_state=42, n_jobs=-1)
rand_rf.fit(X_train, y_train)

print("Best Random Forest:", rand_rf.best_params_)
y_pred_rf = rand_rf.predict(X_test)
print("Accuracy (Random Forest):", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ------------------ Support Vector Machine ------------------
svm = SVC(probability=True)
param_grid_svm = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf", "linear"]
}
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring="accuracy", n_jobs=-1)
grid_svm.fit(X_train, y_train)

print("Best SVM:", grid_svm.best_params_)
y_pred_svm = grid_svm.predict(X_test)
print("Accuracy (SVM):", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
