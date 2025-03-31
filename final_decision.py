import json
import os
import models.normal as normal

OUTPUT_FILE = "output.json"
FINAL_DECISION_FILE = "final_decision.json"

def load_output():
    """Loads the processed transaction data from output.json."""
    try:
        with open(OUTPUT_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def make_final_decision():
    """Compares the processed output with normal transactions and makes a final decision."""
    data = load_output()
    if not data:
        return {"error": "No processed data found."}

    # Compare with normal transaction patterns
    is_normal = normal.compare(data["layering_result"])
    
    # Decision logic
    decision = "Normal_Fan_Out" if is_normal else "Suspicious"
    is_laundering = 0 if is_normal else 1

    # Final structured output with MongoDB-like formatting
    final_result = {
        "_id": { "$oid": "67e8d8d6b6fe0b39e7e8f1d3" },  # Example ObjectID
        "Time": data["input"]["Time"],
        "Date": data["input"]["Date"],
        "Sender_account": data["input"]["Sender_account"],
        "Receiver_account": { "$numberLong": str(data["input"]["Receiver_account"]) },
        "Amount": data["input"]["Amount"],
        "Payment_currency": data["input"]["Payment_currency"],
        "Received_currency": data["input"]["Received_currency"],
        "Sender_bank_location": data["input"]["Sender_bank_location"],
        "Receiver_bank_location": data["input"]["Receiver_bank_location"],
        "Payment_type": data["input"]["Payment_type"],
        "Transaction_ID": data["input"]["Transaction_ID"],
        "Is_laundering": is_laundering,
        "Laundering_type": decision
    }

    # Save final decision
    with open(FINAL_DECISION_FILE, "w") as f:
        json.dump(final_result, f, indent=4)

    return final_result

if __name__ == "__main__":
    result = make_final_decision()
    print(json.dumps(result, indent=4))
