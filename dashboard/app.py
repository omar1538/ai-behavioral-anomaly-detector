import streamlit as st
import requests
import json

st.set_page_config(page_title="AI Anomaly Detector", layout="wide")
st.title("AI-Powered Network Anomaly Detector")
st.markdown("Detects malicious network connections using Machine Learning trained on KDD Cup 1999 dataset")

st.sidebar.header("Connection Parameters")

duration = st.sidebar.slider("Duration", 0.0, 100.0, 0.0)
src_bytes = st.sidebar.number_input("Source Bytes", 0.0, 1000000.0, 0.0)
dst_bytes = st.sidebar.number_input("Destination Bytes", 0.0, 1000000.0, 0.0)
num_failed_logins = st.sidebar.slider("Failed Logins", 0.0, 10.0, 0.0)
logged_in = st.sidebar.selectbox("Logged In", [0.0, 1.0])
count = st.sidebar.slider("Count", 0.0, 512.0, 1.0)
srv_count = st.sidebar.slider("Srv Count", 0.0, 512.0, 1.0)
serror_rate = st.sidebar.slider("Error Rate", 0.0, 1.0, 0.0)
same_srv_rate = st.sidebar.slider("Same Srv Rate", 0.0, 1.0, 1.0)
diff_srv_rate = st.sidebar.slider("Diff Srv Rate", 0.0, 1.0, 0.0)

payload = {
    "duration": duration,
    "src_bytes": src_bytes,
    "dst_bytes": dst_bytes,
    "land": 0.0,
    "wrong_fragment": 0.0,
    "urgent": 0.0,
    "hot": 0.0,
    "num_failed_logins": num_failed_logins,
    "logged_in": logged_in,
    "num_compromised": 0.0,
    "root_shell": 0.0,
    "su_attempted": 0.0,
    "num_root": 0.0,
    "num_file_creations": 0.0,
    "num_shells": 0.0,
    "num_access_files": 0.0,
    "num_outbound_cmds": 0.0,
    "is_host_login": 0.0,
    "is_guest_login": 0.0,
    "count": count,
    "srv_count": srv_count,
    "serror_rate": serror_rate,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": same_srv_rate,
    "diff_srv_rate": diff_srv_rate,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 0.0,
    "dst_host_srv_count": 0.0,
    "dst_host_same_srv_rate": 0.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.0,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0
}

col1, col2, col3 = st.columns(3)

if st.button("Analyze Connection", type="primary"):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()

        with col1:
            st.metric("Prediction", "ATTACK" if result["is_attack"] else "NORMAL")
        with col2:
            st.metric("Risk Score", f"{result['risk_score']*100:.1f}%")
        with col3:
            risk = result["risk_level"]
            if risk == "HIGH":
                st.error(f"Risk Level: {risk}")
            elif risk == "MEDIUM":
                st.warning(f"Risk Level: {risk}")
            else:
                st.success(f"Risk Level: {risk}")

        st.json(result)

    except Exception as e:
        st.error(f"API not reachable. Make sure FastAPI is running. Error: {e}")

st.markdown("---")
st.markdown("**Model:** Random Forest | **Dataset:** KDD Cup 1999 | **ROC-AUC:** 0.9993")
