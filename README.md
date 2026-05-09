# 📡 Telecom Customer Churn Prediction

An end-to-end machine learning project to predict customer churn for a telecom company using the IBM Telco Customer Churn dataset.

---

## 🎯 Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. This project builds a classification model to identify customers likely to churn, enabling the business to take proactive retention actions before losing them.

---

## 📁 Dataset

- **Source:** IBM Telco Customer Churn Dataset (Kaggle)
- **Size:** 7,043 customers, 35 features after preprocessing
- **Target:** `Churn Label` (Yes/No)
- **Class Distribution:** ~26% churn (imbalanced)

---

## 🔧 Tech Stack

- Python 3.13
- pandas, NumPy
- scikit-learn
- XGBoost
- matplotlib, seaborn
- joblib

---

## 🚀 Project Pipeline

### 1. Data Cleaning
- Converted `Total Charges` from string to numeric (hidden space issue)
- Dropped irrelevant columns (location data, IDs)
- Removed data leakage columns (`Customer Status`, `Satisfaction Score`)
- Filled missing values in `Offer` and `Internet Type`

### 2. Exploratory Data Analysis
- Churn rate analysis (~26% imbalance identified)
- Key findings:
  - Month-to-month contract customers churn the most
  - Fiber Optic internet users have highest churn rate
  - Short tenure (0–10 months) customers are highest risk
  - Higher monthly charges correlate with churn

### 3. Preprocessing
- Label Encoding for all categorical features
- StandardScaler for numerical features
- 80/20 train-test split

### 4. Model Comparison

| Model | Accuracy | Churn Recall |
|---|---|---|
| Logistic Regression | 83.7% | 68% |
| Random Forest | 83.8% | 61% |
| XGBoost | 83.0% | 75% |
| **Tuned XGBoost** | **79.2%** | **87%** |

> Churn Recall was chosen as the primary metric because missing an actual churner is more costly to the business than a false alarm.

### 5. Hyperparameter Tuning
- Used `RandomizedSearchCV` with 5-fold cross-validation
- Optimized for recall on the minority class
- Best params: `n_estimators=200, max_depth=5, learning_rate=0.01, subsample=0.8`

### 6. Class Imbalance Handling
- Used `scale_pos_weight` in XGBoost to penalize misclassification of churners

---

## 📊 Key Results

- **Final Model:** Tuned XGBoost
- **Churn Recall: 87%** — catches 87 out of every 100 actual churners
- **Top Features Driving Churn:**
  1. Contract type (Month-to-Month = highest risk)
  2. Internet Service type (Fiber Optic)
  3. Number of Dependents & Referrals
  4. Monthly Charge
  5. Tenure in Months

---

## 📂 Project Structure

```
telecom-churn-prediction/
│
├── telco.csv               # Dataset
├── Telco.py                # Main ML pipeline
├── churn_model.pkl         # Saved XGBoost model
├── scaler.pkl              # Saved StandardScaler
└── README.md
```

---

## ▶️ How to Run

```bash
# Install dependencies
pip install pandas scikit-learn xgboost matplotlib seaborn joblib

# Run the pipeline
python Telco.py
```

---

## 💡 Business Insights

- Customers on **month-to-month contracts** are 3x more likely to churn — offer incentives to switch to annual plans
- **Fiber Optic** users churn despite paying premium — service quality needs improvement
- Customers who **refer friends** are significantly more loyal — referral programs reduce churn
- **New customers (0–10 months)** are the highest risk group — early engagement is critical

---

## 👤 Author

**Pradyumna**  
Electronics and Telecommunications Engineering  
M S Ramaiah Institute of Technology, Bengaluru  
