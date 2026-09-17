import pytest
from fastapi.testclient import TestClient
import pandas as pd
import sqlite3
import os
from src.api.main import app

def test_api_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_auth_login():
    with TestClient(app) as client:
        response = client.post(
            "/auth/login",
            data={"username": "admin", "password": "secure_password_456"}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

def test_api_predict():
    with TestClient(app) as client:
        # Login to get token
        login_response = client.post(
            "/auth/login",
            data={"username": "admin", "password": "secure_password_456"}
        )
        token = login_response.json()["access_token"]
        
        # Test prediction
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": "29.85"
        }
        
        response = client.post("/api/v1/predict", json=payload, headers=headers)
        
        # If the model is not trained yet, it returns 503
        if response.status_code == 503:
            pytest.skip("Model not loaded, skipping prediction test.")
            
        assert response.status_code == 200
        data = response.json()
        assert "ChurnProbability" in data
        assert "RiskTier" in data
        assert "TopDrivers" in data

def test_data_cleaning_output():
    # Assuming the pipeline has been run at least once to create the DB
    db_path = "data/processed/churn_db.sqlite"
    if not os.path.exists(db_path):
        pytest.skip("Database not found, skipping data validation test.")
        
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM cleaned_customer_data", conn)
    conn.close()
    
    assert len(df) > 0
    assert "TotalCharges" in df.columns
    assert "Churn" in df.columns
    assert "IsOutlierCharge" in df.columns
    
    # Churn should be binary
    assert set(df['Churn'].unique()).issubset({0, 1})
