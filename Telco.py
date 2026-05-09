import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── STEP 1: Load ──────────────────────────────────────────────
df = pd.read_csv("telco.csv")

# ── STEP 2: Clean ─────────────────────────────────────────────
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df.dropna(subset=['Total Charges'], inplace=True)

cols_to_drop = ['Customer ID', 'Country', 'State', 'City', 'Zip Code',
                'Latitude', 'Longitude', 'Population', 'Quarter',
                'Churn Category', 'Churn Reason', 'Churn Score', 'CLTV',
                'Customer Status', 'Satisfaction Score']

df.drop(cols_to_drop, axis=1, inplace=True)

df['Offer'] = df['Offer'].fillna('None')
df['Internet Type'] = df['Internet Type'].fillna('None')

print("Shape:", df.shape)
print("Nulls:", df.isnull().sum().sum())
print(df.isnull().sum()[df.isnull().sum() > 0])
# ── STEP 3: EDA ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Telco Churn EDA', fontsize=16)

df['Churn Label'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'], ax=axes[0,0])
axes[0,0].set_title('Churn Distribution')
axes[0,0].tick_params(axis='x', rotation=0)

sns.histplot(data=df, x='Tenure in Months', hue='Churn Label', bins=30, ax=axes[0,1])
axes[0,1].set_title('Tenure vs Churn')

sns.boxplot(data=df, x='Churn Label', y='Monthly Charge', ax=axes[0,2])
axes[0,2].set_title('Monthly Charge vs Churn')

sns.countplot(data=df, x='Contract', hue='Churn Label', ax=axes[1,0])
axes[1,0].set_title('Contract Type vs Churn')
axes[1,0].tick_params(axis='x', rotation=15)

sns.countplot(data=df, x='Internet Type', hue='Churn Label', ax=axes[1,1])
axes[1,1].set_title('Internet Type vs Churn')
axes[1,1].tick_params(axis='x', rotation=15)

sns.countplot(data=df, x='Payment Method', hue='Churn Label', ax=axes[1,2])
axes[1,2].set_title('Payment Method vs Churn')
axes[1,2].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()
plt.close()
# ── STEP 4: Preprocessing ─────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

df['Churn Label'] = df['Churn Label'].map({'Yes': 1, 'No': 0})

cat_cols = df.select_dtypes(include='object').columns.tolist()
print("Categorical columns:", cat_cols)

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("Shape after encoding:", df.shape)
print(df.dtypes)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ── STEP 5: Split ─────────────────────────────────────────────
X = df.drop('Churn Label', axis=1)
y = df['Churn Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ── STEP 6: Train & Evaluate Models ───────────────────────────

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("── Logistic Regression ──")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("── Random Forest ──")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

from xgboost import XGBClassifier

# ── STEP 7: XGBoost with class imbalance handling ─────────────
scale = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(n_estimators=100, scale_pos_weight=scale, 
                    random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("── XGBoost ──")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

from sklearn.model_selection import RandomizedSearchCV

# ── STEP 8: Hyperparameter Tuning ─────────────────────────────
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_tuned = XGBClassifier(scale_pos_weight=scale, random_state=42, eval_metric='logloss')

search = RandomizedSearchCV(xgb_tuned, params, n_iter=20, 
                            scoring='recall', cv=5, random_state=42, n_jobs=-1)
search.fit(X_train, y_train)

print("Best Params:", search.best_params_)
best_pred = search.best_estimator_.predict(X_test)
print("── Tuned XGBoost ──")
print("Accuracy:", accuracy_score(y_test, best_pred))
print(classification_report(y_test, best_pred))

import joblib

# ── STEP 9: Save Model ────────────────────────────────────────
joblib.dump(search.best_estimator_, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved!")

# ── STEP 10: Feature Importance ───────────────────────────────
import numpy as np

feature_names = X.columns
importances = search.best_estimator_.feature_importances_
indices = np.argsort(importances)[-15:]  # top 15 features

plt.figure(figsize=(10, 6))
plt.barh(range(15), importances[indices], color='steelblue')
plt.yticks(range(15), [feature_names[i] for i in indices])
plt.title('Top 15 Important Features')
plt.tight_layout()
plt.show()