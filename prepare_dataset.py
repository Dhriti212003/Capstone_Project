"""
NSL-KDD Dataset Preparation for DP-FedAvg IoT Threat Detection
ADVANCED VERSION: Non-IID Partitioning for Final Year Project
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import urllib.request

# Column names for NSL-KDD dataset
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

def download_nslkdd():
    """Download NSL-KDD dataset"""
    print("Downloading NSL-KDD dataset...")
    urls = {
        'train': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt',
        'test': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'
    }
    os.makedirs('data', exist_ok=True)
    for name, url in urls.items():
        filepath = f'data/{name}.txt'
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(url, filepath)
            print(f"-> {name} data downloaded")
        else:
            print(f"-> {name} data already exists")

def load_and_preprocess():
    """Load and preprocess NSL-KDD dataset"""
    print("\nLoading dataset...")
    train_df = pd.read_csv('data/train.txt', names=COLUMN_NAMES, header=None)
    test_df = pd.read_csv('data/test.txt', names=COLUMN_NAMES, header=None)
    
    # Remove difficulty level
    train_df = train_df.iloc[:, :-1]
    test_df = test_df.iloc[:, :-1]
    
    # Binary classification: Normal (0) vs Attack (1)
    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label'].values
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label'].values
    
    # Encoding categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    combined_df = pd.concat([train_df.drop('label', axis=1), test_df.drop('label', axis=1)], ignore_index=True)
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(combined_df[col])
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le
        
    # Normalizing features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    with open('data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train, y_train, X_test, y_test

def partition_data(X_train, y_train, num_clients=3):
    """
    Partition data for federated learning (Non-IID Distribution)
    This simulates a realistic IoT scenario where devices see different traffic.
    """
    print(f"\n[ADVANCED] Partitioning data NON-IID for {num_clients} clients...")
    
    # Separate indices for Normal and Attack classes
    normal_indices = np.where(y_train == 0)[0]
    attack_indices = np.where(y_train == 1)[0]
    
    # Shuffle indices to ensure variety
    np.random.shuffle(normal_indices)
    np.random.shuffle(attack_indices)

    # Calculate split sizes
    # We want each client to have roughly 1/3 of the total data, but different ratios
    samples_per_client = len(X_train) // num_clients
    
    client_indices = []

    # --- CLIENT 0: The "Safe Device" (85% Normal, 15% Attack) ---
    c0_norm_count = int(samples_per_client * 0.85)
    c0_att_count = samples_per_client - c0_norm_count
    c0_indices = np.concatenate([
        normal_indices[:c0_norm_count], 
        attack_indices[:c0_att_count]
    ])
    client_indices.append(c0_indices)

    # --- CLIENT 1: The "Targeted Device" (15% Normal, 85% Attack) ---
    c1_norm_count = int(samples_per_client * 0.15)
    c1_att_count = samples_per_client - c1_norm_count
    c1_indices = np.concatenate([
        normal_indices[c0_norm_count : c0_norm_count + c1_norm_count], 
        attack_indices[c0_att_count : c0_att_count + c1_att_count]
    ])
    client_indices.append(c1_indices)

    # --- CLIENT 2: The "Gateway Device" (Balanced 50/50 mix) ---
    # Take the remaining data
    c2_norm_start = c0_norm_count + c1_norm_count
    c2_att_start = c0_att_count + c1_att_count
    c2_indices = np.concatenate([
        normal_indices[c2_norm_start:c2_norm_start + (samples_per_client//2)], 
        attack_indices[c2_att_start:c2_att_start + (samples_per_client//2)]
    ])
    client_indices.append(c2_indices)

    # Save partitions
    os.makedirs('data/clients', exist_ok=True)
    for i, indices in enumerate(client_indices):
        np.random.shuffle(indices) # Shuffle the client's local set
        X, y = X_train[indices], y_train[indices]
        np.save(f'data/clients/client_{i}_X.npy', X)
        np.save(f'data/clients/client_{i}_y.npy', y)
        
        # Calculate ratio for the report
        attack_ratio = (np.sum(y) / len(y)) * 100
        print(f"-> Client {i}: {len(X)} samples, Attack Ratio: {attack_ratio:.2f}%")
    
    print("\n[OK] Non-IID Client data saved to data/clients/")

def save_test_data(X_test, y_test):
    """Save test data for global evaluation"""
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    print("-> Global Test data saved")

def main():
    print("="*60)
    print("NSL-KDD Dataset Preparation: Non-IID Experimental Setup")
    print("="*60)
    
    download_nslkdd()
    X_train, y_train, X_test, y_test = load_and_preprocess()
    partition_data(X_train, y_train, num_clients=3)
    save_test_data(X_test, y_test)
    
    print("\n" + "="*60)
    print("[SUCCESS] Dataset is ready for Non-IID DP-FedAvg Experiment")
    print("="*60)

if __name__ == "__main__":
    main()