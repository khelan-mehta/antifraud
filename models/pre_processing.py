import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder

def process_transaction_data():
    input_csv = "stratified_sampled_dataset.csv"  # Hardcoded file name

    df = pd.read_csv(input_csv)
    df = df[df['Is_laundering'].notnull() & (df['Is_laundering'].astype(str).str.strip() != '')]

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df = df.sort_values(by=['Sender_account', 'DateTime'])

    df['HourOfDay'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek

    df['TimeSinceLastTx'] = df.groupby('Sender_account')['DateTime']\
                              .diff().dt.total_seconds().fillna(0)

    df['RollingSum_Amount_5'] = df.groupby('Sender_account')['Amount']\
                                  .transform(lambda x: x.rolling(window=5, min_periods=1).sum())

    df = pd.get_dummies(df, columns=['Payment_type'], prefix='PayType')

    df['IsInternational'] = (df['Sender_bank_location'] != df['Receiver_bank_location']).astype(int)

    df['Repeat_Tx_Count_Sender_Receiver'] = df.groupby(['Sender_account','Receiver_account'])\
                                              .cumcount() + 1

    account_encoder = LabelEncoder()
    all_accounts = pd.concat([df['Sender_account'], df['Receiver_account']]).unique()
    account_encoder.fit(all_accounts)
    df['Sender_account_encoded'] = account_encoder.transform(df['Sender_account'])
    df['Receiver_account_encoded'] = account_encoder.transform(df['Receiver_account'])

    location_encoder = LabelEncoder()
    all_locations = pd.concat([df['Sender_bank_location'], df['Receiver_bank_location']]).unique()
    location_encoder.fit(all_locations)
    df['Sender_bank_location_encoded'] = location_encoder.transform(df['Sender_bank_location'])
    df['Receiver_bank_location_encoded'] = location_encoder.transform(df['Receiver_bank_location'])

    df['Transaction_Frequency'] = df.groupby('Sender_account')['DateTime'].transform('count')

    df['Amount_Mean'] = df.groupby('Sender_account')['Amount'].transform('mean')

    df['Time_Diff'] = df.groupby('Sender_account')['DateTime']\
                        .diff().dt.total_seconds().fillna(0)

    G = nx.from_pandas_edgelist(df, source='Sender_account', target='Receiver_account', create_using=nx.DiGraph())

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    df['Sender_In_Degree'] = df['Sender_account'].map(in_degrees)
    df['Sender_Out_Degree'] = df['Sender_account'].map(out_degrees)

    pagerank_scores = nx.pagerank(G, alpha=0.85)
    df['Sender_PageRank'] = df['Sender_account'].map(pagerank_scores)

    df['Unique_Receivers'] = df.groupby('Sender_account')['Receiver_account'].transform('nunique')

    # Generate Unique Transaction IDs
    df['Transaction_ID'] = 'TXN' + (df.index + 1).astype(str)

    output_csv = "processed_reduced500_Dataset.csv"
    df.to_csv(output_csv, index=False)

    print(f"Processed dataset saved as '{output_csv}'")

    return output_csv


if __name__ == "__main__":
    process_transaction_data()