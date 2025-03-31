import pandas as pd
import networkx as nx
from hmmlearn import hmm
import numpy as np
import json
import os

def load_transaction_data_csv(file_path):
    """Load transaction data from a CSV file."""
    if not os.path.exists(file_path):
        print(f"❌ Error: CSV file '{file_path}' not found.")
        return None
    df = pd.read_csv(file_path)
    print(f"✅ Loaded CSV with {len(df)} transactions")
    return df

def load_transaction_data_json(file_path):
    """Load fraud community data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"❌ Error: JSON file '{file_path}' not found.")
        return None
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded JSON with {len(data.get('fraud_communities', {}))} communities")
    return data

def construct_transaction_graph_csv(df):
    """Construct a transaction graph from CSV data."""
    G = nx.DiGraph()
    for _, row in df.iterrows():
        sender = str(row['Sender_account'])
        receiver = str(row['Receiver_account'])
        G.add_node(sender, type='account')
        G.add_node(receiver, type='account')
        G.add_edge(sender, receiver,
                   amount=row['Amount'],
                   transaction_id=row['Transaction_ID'],
                   timestamp=row['DateTime'],
                   time_diff=row['TimeSinceLastTx'])
    print(f"✅ Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def filter_csv_by_json_communities(df, json_data):
    """Filter CSV transactions based on fraud communities in JSON."""
    all_members = {str(m) for comm_data in json_data['fraud_communities'].values() for m in comm_data['Members']}
    filtered_df = df[(df['Sender_account'].astype(str).isin(all_members)) | 
                      (df['Receiver_account'].astype(str).isin(all_members))]
    print(f"✅ Filtered CSV to {len(filtered_df)} transactions matching JSON members")
    return filtered_df

def prepare_hmm_observations_csv(df):
    """Prepare observations for HMM training."""
    if df.empty:
        return np.array([])

    observations = df[['Amount', 'TimeSinceLastTx', 'Transaction_Frequency']].copy().fillna(0)
    observations = (observations - observations.mean()) / (observations.std() + 1e-8)  # Normalize
    print(f"✅ Prepared {len(observations)} HMM observations")
    return observations.values

def train_online_hmm(observations):
    """Train an HMM model only if enough data points exist."""
    if len(observations) < 10:
        print("⚠️ Warning: Not enough data to train HMM. Skipping...")
        return None

    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
    model.fit(observations)
    return model

def detect_layering_patterns(G, df, json_data, hmm_model):
    """Detect layering and smurfing patterns in transactions."""
    if hmm_model is None:
        return []

    layering_analysis = []

    for comm_id, comm_data in json_data['fraud_communities'].items():
        members = [str(m) for m in comm_data['Members']]
        comm_df = df[(df['Sender_account'].astype(str).isin(members)) | 
                     (df['Receiver_account'].astype(str).isin(members))].copy()

        print(f"\n🔍 Processing Community {comm_id}: {len(comm_df)} transactions")

        if len(comm_df) < 3:
            print(f"⚠️ Skipping community {comm_id}: Not enough transactions.")
            continue

        observations = prepare_hmm_observations_csv(comm_df)
        if len(observations) == 0:
            print(f"⚠️ Skipping community {comm_id}: No valid observations.")
            continue

        try:
            states = hmm_model.predict(observations)
            print(f"🔹 Community {comm_id} HMM States: {states}")
        except Exception as e:
            print(f"❌ Error predicting states for community {comm_id}: {e}")
            continue

        comm_df = comm_df.iloc[:len(states)].reset_index(drop=True)

        structuring_transactions = []
        smurfing_transactions = []
        money_flow = []
        max_layering_depth = 0

        for idx, row in comm_df.iterrows():
            sender = str(row['Sender_account'])
            receiver = str(row['Receiver_account'])
            amount = row['Amount']
            time_diff = row['TimeSinceLastTx']
            tx_id = row['Transaction_ID']

            money_flow.append({"From": sender, "To": receiver, "Amount": amount, "Transaction_ID": tx_id})

            if states[idx] == 1:
                successors = list(G.successors(sender))
                if successors:
                    avg_out_amount = np.mean([G[sender][s]['amount'] for s in successors])
                    if amount > avg_out_amount * 1.5:
                        structuring_transactions.append({
                            "Transaction_ID": tx_id,
                            "Split_Accounts": successors,
                            "Time_Diff": time_diff
                        })
                        max_layering_depth = max(max_layering_depth, len(successors))

            elif states[idx] == 2:
                predecessors = list(G.predecessors(receiver))
                if len(predecessors) >= 1:
                    amounts = [G[p][receiver]['amount'] for p in predecessors]
                    if len(amounts) > 1:
                        smurfing_transactions.append({
                            "Receivers": [receiver],
                            "Sender_Accounts": predecessors,
                            "Amount_Range": f"{min(amounts)}-{max(amounts)}"
                        })

        layering_analysis.append({
            "Community_ID": comm_id,
            "Layering_Depth": max_layering_depth,
            "Structured_Transactions": structuring_transactions,
            "Smurfing_Transactions": smurfing_transactions,
            "Money_Flow": money_flow,
            "Cycles": detect_cycles_scc(G)
        })

    return layering_analysis

def detect_cycles_scc(G):
    """Detect cycles in the transaction graph using Strongly Connected Components (SCC)."""
    scc = list(nx.strongly_connected_components(G))
    cycles = [list(comp) for comp in scc if len(comp) > 1]  
    print(f"🔄 Detected {len(cycles)} cycles using SCC")
    return cycles

def process_real_time(csv_file_path, json_file_path, output_file="output.json"):
    """Main function to process transactions in real time."""
    df = load_transaction_data_csv(csv_file_path)
    json_data = load_transaction_data_json(json_file_path)
    if df is None or json_data is None:
        return

    filtered_df = filter_csv_by_json_communities(df, json_data)
    if filtered_df.empty:
        print("❌ No transactions found matching JSON community members.")
        return {"layering_analysis": [], "total_layering_cases": 0}

    G = construct_transaction_graph_csv(filtered_df)
    observations = prepare_hmm_observations_csv(filtered_df)
    hmm_model = train_online_hmm(observations)
    
    layering_analysis = detect_layering_patterns(G, filtered_df, json_data, hmm_model)

    with open(output_file, "w") as f:
        json.dump(layering_analysis, f, indent=4)
    
    print(f"✅ Output saved to {output_file}")

if __name__ == "__main__":
    process_real_time("processed_Dataset.csv", "fraud_community.json")
