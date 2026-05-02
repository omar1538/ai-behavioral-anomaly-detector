import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("Loading processed data...")
df = pd.read_csv('data/processed_data.csv')

X = df.drop('is_anomaly', axis=1)
y = df['is_anomaly']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model 1 - Isolation Forest (unsupervised)
print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(contamination=0.5, random_state=42, n_jobs=-1)
iso_forest.fit(X_train)
iso_pred = iso_forest.predict(X_test)
iso_pred = np.where(iso_pred == -1, 1, 0)

# Model 2 - Random Forest (supervised)
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Model 3 - LOF (unsupervised)
print("Training Local Outlier Factor...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.5, n_jobs=-1)
lof_pred = lof.fit_predict(X_test)
lof_pred = np.where(lof_pred == -1, 1, 0)

print("\n--- Isolation Forest Results ---")
print(classification_report(y_test, iso_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, iso_pred):.4f}")

print("\n--- Random Forest Results ---")
print(classification_report(y_test, rf_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_pred):.4f}")

print("\n--- Local Outlier Factor Results ---")
print(classification_report(y_test, lof_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, lof_pred):.4f}")

print("\nSaving best model (Random Forest)...")
joblib.dump(rf, 'model/anomaly_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("Model saved.")
