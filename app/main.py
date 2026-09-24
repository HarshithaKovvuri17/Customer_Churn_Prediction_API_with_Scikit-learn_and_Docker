import os
from typing import Optional, Union
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

app = FastAPI(title="Customer Churn Prediction API")
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_pipeline.joblib")
pipeline = None

class CustomerData(BaseModel):
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

    @field_validator('TotalCharges', mode='before')
    @classmethod
    def validate_total_charges(cls, v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            v_str = v.strip()
            if v_str == "":
                return 0.0
            try:
                return float(v_str)
            except ValueError:
                raise ValueError("TotalCharges must be a valid numeric string or number.")
        raise ValueError("TotalCharges must be a numeric value or string.")
