import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

def classify_transactions():
    input_csv = "reduced500_dataset.csv"  # Hardcoded input file
    output_json = "predictions.json"  # Hardcoded output file

    # Load the dataset
    df = pd.read_csv(input_csv)

    # Select features
    features = [
        "Amount", "RollingSum_Amount_5", "TimeSinceLastTx", "Repeat_Tx_Count_Sender_Receiver",
        "Transaction_Frequency", "Amount_Mean", "Time_Diff", "Sender_In_Degree",
        "Sender_Out_Degree", "Sender_PageRank", "Unique_Receivers"
    ]
    X = df[features]
    y = df["Is_laundering"]
    transaction_ids = df["Transaction_ID"]  # Keep transaction IDs for output

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Train SVM on full dataset
    param_grid = {'C': [1, 10], 'gamma': [0.01, 0.1], 'kernel': ['rbf']}
    svc = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='precision', cv=5, n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)

    # Best model
    best_svc = grid_search.best_estimator_
    print("Best parameters found:", grid_search.best_params_)

    # Predict on the full dataset
    y_proba = best_svc.predict_proba(X_scaled)[:, 1]  # Probability of fraud (Is_laundering = 1)

    # Normality Score (Higher = More Normal)
    normality_scores = {txn_id: 1 - score for txn_id, score in zip(transaction_ids, y_proba)}

    # Save as JSON
    with open(output_json, "w") as json_file:
        json.dump(normality_scores, json_file, indent=4)
    
    print(f"Predictions saved to {output_json}")
    
def compare(layering_result):
    # Add logic to determine if the transaction is normal
    return isinstance(layering_result, bool) and layering_result


if __name__ == "__main__":
    classify_transactions()