# ❤️ Heart Disease Prediction Project

A Machine Learning project that predicts the likelihood of heart disease using patient health data.

---

## 📌 Features
- Data preprocessing & cleaning  
- Supervised Learning (Logistic Regression, Decision Tree, Random Forest, SVM)  
- Model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)  
- Unsupervised Learning (K-Means, Hierarchical Clustering)  
- Hyperparameter tuning with GridSearchCV & RandomizedSearchCV  
- Exported trained Decision Tree model (`.pkl`)  
- Deployment-ready script for predictions  

---

## 📂 Project Structure
- `data/` → Dataset  
- `notebooks/` → Jupyter notebooks for each step  
- `scripts/` → Python scripts (preprocessing, training, clustering, deployment)  
- `models/` → Trained models in `.pkl` format  

---

## ⚙️ Setup Instructions
```bash
# Clone the repository
git clone https://github.com/your-username/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
