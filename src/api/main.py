from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from pydantic_settings import BaseSettings
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
import shap
import warnings

# Add src to sys path if not present, to ensure transformers can be unpickled
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Setup Logging
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("predictions")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')

fh = logging.FileHandler("logs/predictions.log")
fh.setFormatter(formatter)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)

# Configuration using pydantic-settings
class Settings(BaseSettings):
    JWT_SECRET_KEY: str = "default_dev_key"
    ADMIN_USER: str = "admin"
    ADMIN_PASSWORD: str = "password"
    
    class Config:
        env_file = ".env"

settings = Settings()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="Churn Prediction API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Load Model Pipeline
model_path = "models/xgboost_churn_pipeline.pkl"
pipeline = None
optimal_threshold = 0.5
explainer = None

@app.on_event("startup")
def load_model():
    global pipeline, optimal_threshold, explainer
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            pipeline = data['pipeline']
            optimal_threshold = data.get('optimal_threshold', 0.5)
            
            # Initialize SHAP explainer on the classifier step
            # Note: We need a background dataset if it's not tree-based, but for XGBoost TreeExplainer is fine without data
            classifier = pipeline.named_steps['classifier']
            explainer = shap.TreeExplainer(classifier)
            
    except Exception as e:
        print(f"Warning: Pipeline not found or failed to load. Run pipeline first. ({e})")

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
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username == settings.ADMIN_USER and form_data.password == settings.ADMIN_PASSWORD:
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": form_data.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Incorrect username or password")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(payload: CustomerPayload, current_user: str = Depends(get_current_user)):
    start_time = time.time()
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    # 1. Preprocess payload to DataFrame
    df = pd.DataFrame([payload.dict()])
    
    # 2. Predict using the pipeline directly on the raw df
    prob = pipeline.predict_proba(df)[0][1]
    
    # Risk Tiering based on optimal threshold
    # Example logic: > optimal + 0.15 is High, > optimal - 0.15 is Medium
    if prob >= min(optimal_threshold + 0.15, 0.9):
        risk_tier = "High"
    elif prob >= max(optimal_threshold - 0.15, 0.1):
        risk_tier = "Medium"
    else:
        risk_tier = "Low"
        
    # 3. Top Drivers using SHAP
    top_drivers = []
    if explainer is not None:
        # Transform the data through the pipeline up to the classifier
        transformed_df = df
        for name, step in pipeline.steps[:-1]:
            transformed_df = step.transform(transformed_df)
            
        shap_values = explainer.shap_values(transformed_df)
        
        # Ensure shap_values is a 1D array for single prediction
        if isinstance(shap_values, list): # For some multiclass or objective setups
            shap_values = shap_values[1] 
        
        shap_values_1d = shap_values[0] if shap_values.ndim > 1 else shap_values
        
        # Get feature names from the ColumnTransformer step (preprocessor)
        preprocessor = pipeline.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        
        # Sort by absolute impact
        top_indices = np.argsort(np.abs(shap_values_1d))[::-1][:3]
        top_drivers = [feature_names[i] for i in top_indices]
    
    latency = time.time() - start_time
    
    # 4. Log Request & Response
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
