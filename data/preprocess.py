import pandas as pd
import numpy as np

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label'
]

print("Loading dataset...")
df = pd.read_csv('data/kdd_data.csv', names=columns)

print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts().head(10)}")

# normal traffic = 0, attack = 1
df['is_anomaly'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)

# keep only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'is_anomaly']

df_clean = df[numeric_cols + ['is_anomaly']].copy()

# sample 50000 rows to keep it manageable
df_sample = df_clean.sample(n=50000, random_state=42)

df_sample.to_csv('data/processed_data.csv', index=False)
print(f"Processed data saved! Shape: {df_sample.shape}")
print(f"Normal traffic: {(df_sample['is_anomaly']==0).sum()}")
print(f"Attacks detected: {(df_sample['is_anomaly']==1).sum()}")
