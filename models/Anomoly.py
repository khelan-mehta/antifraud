import os
import pandas as pd
import numpy as np
import json
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore") 

# 🔹 Define file path
file_path = os.path.abspath("processed_Dataset.csv")

# 🔹 Check if dataset exists
if not os.path.exists(file_path):
    print(f"❌ Dataset not found at {file_path}")
    exit(1)

print("✅ Dataset found. Loading...")
df = pd.read_csv(file_path)

# 🔹 Check for missing values
if df.isnull().sum().sum() > 0:
    print("⚠️ Dataset contains missing values. Filling with median.")
    df.fillna(df.median(numeric_only=True), inplace=True)

# 🔹 Create Cross_Border_Flag feature
df['Cross_Border_Flag'] = (df['Sender_bank_location_encoded'] != df['Receiver_bank_location_encoded']).astype(int)

# 🔹 Define feature set
features = [
    'Transaction_Frequency', 'Amount_Mean', 'Time_Diff',
    'Sender_In_Degree', 'Sender_Out_Degree', 'Unique_Receivers',
    'Sender_PageRank', 'Cross_Border_Flag'
]
X = df[features]
y = df['Is_laundering']

# 🔹 Apply SMOTE for balancing
print("\n🔄 Applying SMOTE for balancing...")
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 🔹 Load pre-determined best hyperparameters
best_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'max_depth': 5,
    'n_estimators': 300,
    'scale_pos_weight': 1,
    'subsample': 0.8
}

# 🔹 Initialize and train XGBoost model with best hyperparameters
print("\n🚀 Loading XGBoost Model with Best Parameters...")
best_xgb = xgb.XGBClassifier(
    objective='binary:logistic', 
    eval_metric='logloss',
    random_state=42,
    **best_params
)
best_xgb.fit(X_train, y_train)

# 🔹 Make Predictions
y_pred_probs = best_xgb.predict_proba(X_test)[:, 1]
threshold = 0.7  # Decision threshold
y_pred = (y_pred_probs >= threshold).astype(int)

# 🔹 Model Evaluation
print("\n📊 Model Evaluation (Threshold =", threshold, ")")
print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("🔹 Precision:", precision_score(y_test, y_pred, zero_division=1))
print("🔹 Recall:", recall_score(y_test, y_pred, zero_division=1))
print("🔹 F1 Score:", f1_score(y_test, y_pred, zero_division=1))
print("🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# 🔹 Retrieve anomaly scores with transaction details
print("\n🔍 Generating Anomaly Scores...")
anomaly_scores = []
for idx, prob in zip(y_test.index, y_pred_probs):
    original_idx = idx % len(df)  # Map back to original dataset
    anomaly_scores.append({
        "Transaction_ID": str(df.loc[original_idx, "Transaction_ID"]),
        "Sender_account": str(df.loc[original_idx, "Sender_account"]),
        "Receiver_account": str(df.loc[original_idx, "Receiver_account"]),
        "Amount": float(df.loc[original_idx, "Amount"]),
        "Transaction_Frequency": int(df.loc[original_idx, "Transaction_Frequency"]),
        "Time_Diff": int(df.loc[original_idx, "Time_Diff"]),
        "Unique_Receivers": int(df.loc[original_idx, "Unique_Receivers"]),
        "Anomaly_Score": round(float(prob), 4)
    })

# 🔹 Load existing anomaly scores if available
output_file = os.path.abspath("anomaly_scores.json")
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        existing_scores = json.load(f)
    anomaly_scores.extend(existing_scores)  # Append new scores

# 🔹 Save updated anomaly scores to JSON
with open(output_file, "w") as f:
    json.dump(anomaly_scores, f, indent=4)

print(f"\n✅ Anomaly scores updated and saved to {output_file}")