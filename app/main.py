import os
from typing import Optional, Union
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, field_validator

from app.utils import format_prediction_response

app = FastAPI(title="Customer Churn Prediction API")
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_pipeline.joblib")
pipeline = None

class CustomerData(BaseModel):
    model_config = ConfigDict(extra="ignore")
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
    TotalCharges: Union[float, str]
    customerID: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to Customer Churn Prediction API", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy" if pipeline is not None else "degraded", "model_loaded": pipeline is not None}

@app.post("/predict")
def predict_churn(data: CustomerData):
    global pipeline
    if pipeline is None:
        if os.path.exists(MODEL_PATH):
            pipeline = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=500, detail="Model pipeline not loaded.")
    input_dict = data.model_dump()
    input_dict.pop('customerID', None)
    df = pd.DataFrame([input_dict])
    pred_raw = pipeline.predict(df)[0]
    prob_raw = pipeline.predict_proba(df)[0][1]
    return format_prediction_response(pred_raw, prob_raw)
