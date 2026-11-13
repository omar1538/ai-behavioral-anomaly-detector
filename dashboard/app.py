import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Anomaly Detector", layout="wide")

# Paths resolved relative to the repo root so it works locally and on Streamlit Cloud
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "model", "anomaly_model.pkl")
SCALER_PATH = os.path.join(ROOT, "model", "scaler.pkl")
DATA_PATH = os.path.join(ROOT, "data", "processed_data.csv")


@st.cache_resource
def load_model():
    """Load the trained model, or train it once if the .pkl files are missing.

    Streamlit Cloud runs only this process (no FastAPI), and the repo does not
    ship the .pkl files, so we build them on first launch and cache in memory.
    """
    df = pd.read_csv(DATA_PATH)
    feature_names = [c for c in df.columns if c != "is_anomaly"]

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, feature_names

    X = df[feature_names]
    y = df["is_anomaly"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    except Exception:
        pass  # read-only FS is fine; model stays cached in memory
    return model, scaler, feature_names


model, scaler, FEATURES = load_model()

st.title("AI-Powered Network Anomaly Detector")
st.markdown("Detects malicious network connections using Machine Learning trained on KDD Cup 1999 dataset")

st.sidebar.header("Connection Parameters")

inputs = {
    "duration": st.sidebar.slider("Duration", 0.0, 100.0, 0.0),
    "src_bytes": st.sidebar.number_input("Source Bytes", 0.0, 1000000.0, 0.0),
    "dst_bytes": st.sidebar.number_input("Destination Bytes", 0.0, 1000000.0, 0.0),
    "num_failed_logins": st.sidebar.slider("Failed Logins", 0.0, 10.0, 0.0),
    "logged_in": st.sidebar.selectbox("Logged In", [0.0, 1.0]),
    "count": st.sidebar.slider("Count", 0.0, 512.0, 1.0),
    "srv_count": st.sidebar.slider("Srv Count", 0.0, 512.0, 1.0),
    "serror_rate": st.sidebar.slider("Error Rate", 0.0, 1.0, 0.0),
    "same_srv_rate": st.sidebar.slider("Same Srv Rate", 0.0, 1.0, 1.0),
    "diff_srv_rate": st.sidebar.slider("Diff Srv Rate", 0.0, 1.0, 0.0),
}

# Build a full feature row: every model feature, defaulting to 0.0, overridden by the sidebar
row = {feat: 0.0 for feat in FEATURES}
row.update({k: v for k, v in inputs.items() if k in row})

col1, col2, col3 = st.columns(3)

if st.button("Analyze Connection", type="primary"):
    X = pd.DataFrame([row])[FEATURES]
    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])

    risk_level = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.4 else "LOW"

    with col1:
        st.metric("Prediction", "ATTACK" if prediction == 1 else "NORMAL")
    with col2:
        st.metric("Risk Score", f"{probability * 100:.1f}%")
    with col3:
        if risk_level == "HIGH":
            st.error(f"Risk Level: {risk_level}")
        elif risk_level == "MEDIUM":
            st.warning(f"Risk Level: {risk_level}")
        else:
            st.success(f"Risk Level: {risk_level}")

    st.json({
        "prediction": prediction,
        "is_attack": bool(prediction == 1),
        "risk_score": round(probability, 4),
        "risk_level": risk_level,
    })

st.markdown("---")
st.markdown("**Model:** Random Forest | **Dataset:** KDD Cup 1999 | **ROC-AUC:** 0.9993")
