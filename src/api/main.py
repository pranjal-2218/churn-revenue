from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import jwt
from jwt import PyJWTError as JWTError
from datetime import datetime, timedelta
import os
import pickle
import pandas as pd
import numpy as np
import time
import json
import logging

# Setup Logging
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("predictions")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/predictions.log")
fh.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(fh)

# Security Configuration
SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'default_dev_key')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="Churn Prediction API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Load Model
model_path = "models/xgboost_churn_model.pkl"
model = None
model_features = None

@app.on_event("startup")
def load_model():
    global model, model_features
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            model_features = data['features']
    except Exception as e:
        print(f"Warning: Model not found or failed to load. Run pipeline first. ({e})")

# Pydantic Models
class CustomerPayload(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

class PredictionResponse(BaseModel):
    ChurnProbability: float
    RiskTier: str
    TopDrivers: list[str]

# Auth Helper
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Dummy auth for demonstration
    if form_data.username == "admin" and form_data.password == "password":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": form_data.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Incorrect username or password")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(payload: CustomerPayload, current_user: str = Depends(get_current_user)):
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    # 1. Preprocess payload to match training features
    df = pd.DataFrame([payload.dict()])
    
    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = np.where(
        df['TotalCharges'].isna(),
        np.where(df['tenure'] > 0, df['MonthlyCharges'] * df['tenure'], 0),
        df['TotalCharges']
    )
    
    # Outlier Charge Flag (Simplified static threshold for API, normally use trained scaler)
    df['IsOutlierCharge'] = (df['TotalCharges'] > 8000).astype(int) # Approximation
    
    # Tenure Bucket
    df['TenureBucket'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 200], labels=['<1 yr', '1-2 yrs', '2-4 yrs', '4+ yrs'])
    
    # Multi-Product Adoption Index
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['MultiProductAdoptionIndex'] = sum((df[service] == 'Yes').astype(int) for service in services if service in df.columns)
    
    # Financial Features
    df['ChargeToTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['LifetimeDiscrepancyRatio'] = df['TotalCharges'] / (df['tenure'] * df['MonthlyCharges'] + 1)
    
    # One-hot encoding (aligning to model_features)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Ensure all columns match model_features exactly
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]
    
    # 2. Predict
    prob = model.predict_proba(df)[0][1]
    
    if prob > 0.7:
        risk_tier = "High"
    elif prob > 0.3:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"
        
    # Top Drivers (Simplified local explanation based on global feature importance * feature value)
    feature_importances = model.feature_importances_
    # Multiply feature values by their global importance to approximate local impact
    impacts = df.iloc[0].values * feature_importances
    top_indices = np.argsort(impacts)[::-1][:3]
    top_drivers = [model_features[i] for i in top_indices]
    
    latency = time.time() - start_time
    
    # 3. Log Request & Response
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "payload": payload.dict(),
        "churn_probability": float(prob),
        "risk_tier": risk_tier,
        "latency_sec": latency
    }
    logger.info(json.dumps(log_data))
    
    return PredictionResponse(
        ChurnProbability=float(prob),
        RiskTier=risk_tier,
        TopDrivers=top_drivers
    )
