import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_telecom_italia(data_dir, n_cells=100, seq_len=12, horizon=3):
    # Find all data files in the directory
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    if not files:
        raise FileNotFoundError(f"No .txt data files found in {data_dir}")
    
    all_data = []
    for file in files:
        filepath = os.path.join(data_dir, file)
        print(f"Loading {filepath}...")
        # Note: Dataset has 8 columns: squareid, timeinterval, countrycode, smsin, smsout, callin, callout, internet
        df = pd.read_csv(filepath, sep='\t', header=None,
                         names=['grid', 'interval', 'country', 'sms_in', 'sms_out',
                                'call_in', 'call_out', 'internet'])
        
        # Filter for the first n_cells and fill NaNs
        df = df[df['grid'] <= n_cells].fillna(0)
        
        # Calculate total activity
        df['activity'] = df['internet'] + df['call_in'] + df['call_out']
        
        # Pivot to get (interval, grid) matrix
        pivot = df.pivot_table(index='interval', columns='grid',
                               values='activity', fill_value=0)
        all_data.append(pivot)

    # Combine all days
    full_pivot = pd.concat(all_data).sort_index()
    data = full_pivot.values.astype(np.float32)   # (T, N)
    
    print(f"Total timesteps: {len(data)}, Nodes: {data.shape[1]}")
    
    # Normalise to [0, 1]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    X, y = [], []
    for t in range(len(data) - seq_len - horizon):
        X.append(data[t : t+seq_len])
        y.append(data[t+seq_len : t+seq_len+horizon])
    
    X, y = np.stack(X), np.stack(y)          # (samples, seq, N)
    
    # Split into train/val/test (70/10/20)
    s1, s2 = int(0.7*len(X)), int(0.8*len(X))
    return (X[:s1], y[:s1]), (X[s1:s2], y[s1:s2]), (X[s2:], y[s2:]), scaler

if __name__ == "__main__":
    # Test loading
    data_path = './wireless dataset'
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), scaler = load_telecom_italia(data_path, n_cells=100)
    print(f"X_train shape: {X_tr.shape}, y_train shape: {y_tr.shape}")
