from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd

app = FastAPI(title="AI Behavioral Anomaly Detector", version="1.0")

model = joblib.load("model/anomaly_model.pkl")
scaler = joblib.load("model/scaler.pkl")

class ConnectionData(BaseModel):
    duration: float
    src_bytes: float
    dst_bytes: float
    land: float
    wrong_fragment: float
    urgent: float
    hot: float
    num_failed_logins: float
    logged_in: float
    num_compromised: float
    root_shell: float
    su_attempted: float
    num_root: float
    num_file_creations: float
    num_shells: float
    num_access_files: float
    num_outbound_cmds: float
    is_host_login: float
    is_guest_login: float
    count: float
    srv_count: float
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: float
    dst_host_srv_count: float
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float

@app.get("/")
def root():
    return {"message": "AI Behavioral Anomaly Detector API", "status": "running"}

@app.post("/predict")
def predict(data: ConnectionData):
    input_data = pd.DataFrame([data.dict()])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    risk_level = "LOW"
    if probability > 0.7:
        risk_level = "HIGH"
    elif probability > 0.4:
        risk_level = "MEDIUM"

    return {
        "prediction": int(prediction),
        "is_attack": bool(prediction == 1),
        "risk_score": round(float(probability), 4),
        "risk_level": risk_level
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
