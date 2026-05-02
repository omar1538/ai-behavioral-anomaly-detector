# AI-Powered Network Anomaly Detector

A machine learning system that detects malicious network connections in real time, trained on the KDD Cup 1999 intrusion detection dataset.

## Overview

This project implements a behavioral anomaly detection pipeline combining:
- **Data preprocessing** of real network traffic logs
- **Three ML models compared:** Isolation Forest, Local Outlier Factor, Random Forest
- **REST API** built with FastAPI to serve predictions
- **Interactive dashboard** built with Streamlit

## Results

| Model | ROC-AUC |
|-------|---------|
| Isolation Forest | 0.1795 |
| Local Outlier Factor | 0.2261 |
| **Random Forest** | **0.9993** |

Random Forest was selected as the production model.

## Project Structure# ai-behavioral-anomaly-detector
## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/omar1538/ai-behavioral-anomaly-detector
cd ai-behavioral-anomaly-detector
```

**2. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Start the API**
```bash
uvicorn api.main:app --reload
```

**4. Start the dashboard (new terminal)**
```bash
streamlit run dashboard/app.py
```

**5. Open browser**
- API docs: http://127.0.0.1:8000/docs
- Dashboard: http://localhost:8501

## Tech Stack

Python, scikit-learn, FastAPI, Streamlit, Pandas, Docker
