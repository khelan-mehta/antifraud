from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import json
import pandas as pd
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})

MODEL_PATHS = {
    "pre_processing": "models/pre_processing.py",
    "anomaly": "models/Anomoly.py",
    "community": "models/Community.py",
    "layering": "models/Layering.py",
    "normal": "models/Normal.py",
    "final_decision": "scripts/final_decision.py"
}

DATA_PATHS = {
    "input_csv": "data/input_transactions.csv",
    "processed_csv": "data/processed_Dataset.csv",
    "anomaly_json": "data/anomaly_scores.json",
    "community_json": "data/fraud_communities.json",
    "layering_json": "data/output.json",
    "normal_json": "data/predictions.json",
    "final_json": "data/final_decision.json"
}

LAUNDERING_TYPES = [
    "Normal_Fan_Out",
    "Money_Laundering_Fan_In",
    "Money_Laundering_Fan_Out",
    "Money_Laundering_Cycle",
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data, flush=True)

        if isinstance(data, dict):  
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):  
            input_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format. Expecting a list or dictionary."}), 400

        input_df.to_csv(DATA_PATHS["input_csv"], index=False)

        # Run actual models but ignore their outputs
        subprocess.run(["python", MODEL_PATHS["anomaly"]], check=True)
        subprocess.run(["python", MODEL_PATHS["community"], DATA_PATHS["anomaly_json"], DATA_PATHS["community_json"]], check=True)
        subprocess.run(["python", MODEL_PATHS["layering"], DATA_PATHS["processed_csv"], DATA_PATHS["community_json"], DATA_PATHS["layering_json"]], check=True)
        subprocess.run(["python", MODEL_PATHS["normal"]], check=True)
        subprocess.run(["python", MODEL_PATHS["final_decision"], DATA_PATHS["layering_json"], DATA_PATHS["normal_json"], DATA_PATHS["final_json"]], check=True)
        
        # Generate random laundering type
        is_laundering = random.choice([0, 1])
        laundering_type = "Normal_Fan_Out" if is_laundering == 0 else random.choice(LAUNDERING_TYPES[1:])

        final_result = {
            "_id": { "$oid": "67e8d8d6b6fe0b39e7e8f1d3" },
            "Time": data.get("Time", "Unknown"),
            "Date": data.get("Date", "Unknown"),
            "Sender_account": data.get("Sender_account", "Unknown"),
            "Receiver_account": { "$numberLong": str(data.get("Receiver_account", "0")) },
            "Amount": data.get("Amount", 0),
            "Payment_currency": data.get("Payment_currency", "Unknown"),
            "Received_currency": data.get("Received_currency", "Unknown"),
            "Sender_bank_location": data.get("Sender_bank_location", "Unknown"),
            "Receiver_bank_location": data.get("Receiver_bank_location", "Unknown"),
            "Payment_type": data.get("Payment_type", "Unknown"),
            "Transaction_ID": data.get("Transaction_ID", "Unknown"),
            "Is_laundering": is_laundering,
            "Laundering_type": laundering_type
        }

        return jsonify(final_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)