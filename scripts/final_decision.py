import json
import os
import sys

# Ensure the parent directory is added to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import models.normal as normal
except ImportError:
    print("Error: Could not import 'models.normal'. Make sure the module exists.")
    sys.exit(1)

OUTPUT_FILE = "output.json"
FINAL_DECISION_FILE = "final_decision.json"

def load_output():
    """Loads the processed transaction data from output.json."""
    if not os.path.exists(OUTPUT_FILE):
        print(f"Error: {OUTPUT_FILE} not found.")
        return None
    
    try:
        with open(OUTPUT_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {OUTPUT_FILE} contains invalid JSON.")
        return None

def make_final_decision():
    """Compares the processed output with normal transactions and makes a final decision."""
    data = load_output()
    if not data:
        return {"error": "No processed data found."}

    if "layering_result" not in data:
        return {"error": "Missing 'layering_result' in data."}

    # Handle if layering_result is a list
    layering_result = data["layering_result"]
    if isinstance(layering_result, list):
        is_normal = all(normal.compare(item) for item in layering_result if isinstance(item, dict))
    elif isinstance(layering_result, dict):
        is_normal = normal.compare(layering_result)
    else:
        return {"error": "'layering_result' must be a list or dictionary."}

    # Decision logic
    decision = "Normal_Fan_Out" if is_normal else "Suspicious"
    is_laundering = 0 if is_normal else 1

    # Ensure "input" exists in data
    if "input" not in data:
        return {"error": "Missing 'input' in data."}

    # Final structured output with MongoDB-like formatting
    try:
        final_result = {
            "_id": { "$oid": "67e8d8d6b6fe0b39e7e8f1d3" },  # Example ObjectID
            "Time": data["input"].get("Time", "Unknown"),
            "Date": data["input"].get("Date", "Unknown"),
            "Sender_account": data["input"].get("Sender_account", "Unknown"),
            "Receiver_account": { "$numberLong": str(data["input"].get("Receiver_account", "0")) },
            "Amount": data["input"].get("Amount", 0),
            "Payment_currency": data["input"].get("Payment_currency", "Unknown"),
            "Received_currency": data["input"].get("Received_currency", "Unknown"),
            "Sender_bank_location": data["input"].get("Sender_bank_location", "Unknown"),
            "Receiver_bank_location": data["input"].get("Receiver_bank_location", "Unknown"),
            "Payment_type": data["input"].get("Payment_type", "Unknown"),
            "Transaction_ID": data["input"].get("Transaction_ID", "Unknown"),
            "Is_laundering": is_laundering,
            "Laundering_type": decision
        }
    except Exception as e:
        return {"error": f"Error constructing final result: {str(e)}"}

    # Save final decision
    try:
        with open(FINAL_DECISION_FILE, "w") as f:
            json.dump(final_result, f, indent=4)
    except IOError:
        return {"error": f"Failed to write to {FINAL_DECISION_FILE}."}

    return final_result

if __name__ == "__main__":
    result = make_final_decision()
    print(json.dumps(result, indent=4))
 