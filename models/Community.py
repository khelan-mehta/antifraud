import json
import networkx as nx
from igraph import Graph
import numpy as np
import os
import warnings
import argparse
import logging
from collections import defaultdict

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Constants
ANOMALY_SCORE_THRESHOLD = 0.85
TRANSACTION_FREQ_THRESHOLD = 3  # Lowered since most transactions have frequency 1-2
UNIQUE_RECEIVERS_THRESHOLD = 3  # Lowered since most have 1-2 unique receivers
TIME_DIFF_THRESHOLD = 3600  # 1 hour in seconds
MIN_COMMUNITY_SIZE = 3  # Minimum nodes to consider a community

def load_transactions_from_json(file_path = "anomaly_scores.json"):
    """Load and preprocess transactions from JSON file"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        with open("anomaly_scores.json", 'r') as f:
            content = f.read().strip()
            if not content:
                raise ValueError("JSON file is empty.")
            transactions = json.loads(content)


        if not isinstance(transactions, list):
            raise ValueError("JSON file should contain an array of transactions")

        # Filter suspicious transactions
        flagged = [
            tx for tx in transactions if (
                tx['Anomaly_Score'] > ANOMALY_SCORE_THRESHOLD or
                tx['Transaction_Frequency'] > TRANSACTION_FREQ_THRESHOLD or
                (tx['Unique_Receivers'] > UNIQUE_RECEIVERS_THRESHOLD and 
                 tx['Time_Diff'] < TIME_DIFF_THRESHOLD)
            )
        ]

        senders = {str(tx['Sender_account']) for tx in transactions}
        receivers = {str(tx['Receiver_account']) for tx in transactions}
        all_nodes = senders | receivers

        avg_anomaly = np.mean([tx['Anomaly_Score'] for tx in flagged]) if flagged else 0

        logging.info(f"Loaded {len(flagged)} flagged transactions from {len(transactions)} total.")
        return flagged, all_nodes, avg_anomaly

    except Exception as e:
        logging.error(f"Error loading transactions: {e}")
        return [], set(), 0

def build_transaction_graph(transactions, all_nodes, avg_anomaly):
    """Build NetworkX graph"""
    G = nx.DiGraph()
    
    # Add all nodes first with default anomaly score
    for node in all_nodes:
        node_str = str(node).strip()
        if node_str:
            G.add_node(node_str, anomaly_score=avg_anomaly)

    # Add edges and update node anomaly scores
    edge_count = 0
    for tx in transactions:
        try:
            sender = str(tx['Sender_account']).strip()
            receiver = str(tx['Receiver_account']).strip()
            if sender and receiver:  # Ensure nodes are not empty
                # Update anomaly scores to the maximum between existing and new
                G.nodes[sender]['anomaly_score'] = max(G.nodes[sender].get('anomaly_score', avg_anomaly), tx['Anomaly_Score'])
                G.nodes[receiver]['anomaly_score'] = max(G.nodes[receiver].get('anomaly_score', avg_anomaly), tx['Anomaly_Score'])
                
                # Add edge with transaction details
                if G.has_edge(sender, receiver):
                    # If edge exists, accumulate amount
                    G[sender][receiver]['amount'] += tx['Amount']
                    G[sender][receiver]['transaction_ids'].append(tx['Transaction_ID'])
                else:
                    G.add_edge(sender, receiver, amount=tx['Amount'], transaction_ids=[tx['Transaction_ID']])
                    edge_count += 1
        except (KeyError, AttributeError) as e:
            logging.warning(f"Skipping transaction due to error: {e}")
            continue

    logging.info(f"Built graph with {G.number_of_nodes()} nodes and {edge_count} edges")
    return G

def convert_to_igraph(nx_graph):
    """Convert NetworkX graph to igraph"""
    nodes = list(nx_graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nx_graph.edges()]

    g = Graph(directed=True)
    g.add_vertices(len(nodes))

    for node, idx in node_to_idx.items():
        g.vs[idx]['name'] = node
        g.vs[idx]['anomaly_score'] = nx_graph.nodes[node].get('anomaly_score', 0)

    g.add_edges(edges)
    g.es['weight'] = [nx_graph[u][v]['amount'] for u, v in nx_graph.edges()]
    return g

def detect_fraud_communities(igraph_graph):
    """Detect fraud communities using Infomap"""
    try:
        partition = igraph_graph.community_infomap(
            edge_weights='weight', 
            trials=10, 
            vertex_weights='anomaly_score'
        )
        communities = {
            f"C{i+1}": [igraph_graph.vs[idx]['name'] for idx in community] 
            for i, community in enumerate(partition) 
            if len(community) >= MIN_COMMUNITY_SIZE
        }
        logging.info(f"Detected {len(communities)} communities (original: {len(partition)})")
        return communities
    except Exception as e:
        logging.error(f"Error in community detection: {e}")
        return {}

def filter_suspicious_communities(nx_graph, communities):
    """Filter suspicious communities"""
    suspicious_communities = {}
    metrics_list = []

    for comm_id, nodes in communities.items():
        metrics = calculate_community_metrics(nx_graph, nodes)
        if metrics:
            metrics['id'] = comm_id
            metrics_list.append(metrics)

    if not metrics_list:
        return suspicious_communities

    # Calculate thresholds based on median values
    anomaly_threshold = np.median([m['avg_anomaly'] for m in metrics_list]) if metrics_list else 0
    amount_threshold = np.median([m['avg_amount'] for m in metrics_list]) if metrics_list else 0
    density_threshold = np.median([m['density'] for m in metrics_list]) if metrics_list else 0

    logging.info(f"\nThresholds - Anomaly: {anomaly_threshold:.2f}, Amount: {amount_threshold:.2f}, Density: {density_threshold:.2f}")

    for metrics in metrics_list:
        if (metrics['avg_anomaly'] >= anomaly_threshold and 
            metrics['avg_amount'] >= amount_threshold and 
            metrics['density'] >= density_threshold):
            suspicious_communities[metrics['id']] = {
                'Members': metrics['nodes'],
                'Avg_Anomaly': round(metrics['avg_anomaly'], 2),
                'Avg_Amount': round(metrics['avg_amount'], 2),
                'Edge_Count': metrics['edge_count'],
                'Density': round(metrics['density'], 2),
                'Transaction_Count': metrics['transaction_count']
            }

    return suspicious_communities

def calculate_community_metrics(nx_graph, community_nodes):
    """Calculate community metrics"""
    valid_nodes = [node for node in community_nodes if node in nx_graph.nodes]
    if len(valid_nodes) < MIN_COMMUNITY_SIZE:
        return None

    # Calculate node-based metrics
    total_anomaly = sum(nx_graph.nodes[node].get('anomaly_score', 0) for node in valid_nodes)
    
    # Calculate edge-based metrics
    edge_amounts = []
    transaction_count = 0
    edge_count = 0
    
    for u in valid_nodes:
        for v in nx_graph.successors(u):
            if v in valid_nodes:
                edge_data = nx_graph[u][v]
                edge_amounts.append(edge_data['amount'])
                transaction_count += len(edge_data.get('transaction_ids', []))
                edge_count += 1

    total_amount = sum(edge_amounts) if edge_amounts else 0
    node_count = len(valid_nodes)

    return {
        'avg_anomaly': total_anomaly / node_count,
        'avg_amount': total_amount / edge_count if edge_count else 0,
        'edge_count': edge_count,
        'transaction_count': transaction_count,
        'node_count': node_count,
        'density': edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0,
        'nodes': valid_nodes
    }

def main(input_file, output_file):
    logging.info("Starting fraud community detection pipeline...")

    flagged_transactions, all_nodes, avg_anomaly = load_transactions_from_json(input_file)
    if not flagged_transactions:
        logging.info("No suspicious transactions found. Exiting.")
        return

    nx_graph = build_transaction_graph(flagged_transactions, all_nodes, avg_anomaly)
    if nx_graph.number_of_edges() == 0:
        logging.info("No edges in graph. Exiting.")
        return

    igraph_graph = convert_to_igraph(nx_graph)
    communities = detect_fraud_communities(igraph_graph)
    suspicious_communities = filter_suspicious_communities(nx_graph, communities)

    with open(output_file, 'w') as f:
        json.dump({"fraud_communities": suspicious_communities}, f, indent=4)

    logging.info(f"Results saved to {output_file}")
    logging.info(f"Found {len(suspicious_communities)} suspicious communities")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect fraud communities in transaction data")
    parser.add_argument("anomaly_scores", help="Path to the JSON file containing transactions")
    parser.add_argument("fraud_community", help="Path to save the results")
    args = parser.parse_args()
    
    # For testing in VS Code, you can hardcode paths like this:
    # args.anomaly_scores = "anomaly_scores.json"
    # args.fraud_community = "fraud_communities.json"
    
    main(args.anomaly_scores, args.fraud_community)