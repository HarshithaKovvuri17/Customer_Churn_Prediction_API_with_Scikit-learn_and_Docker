import os
from typing import Optional, Union
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

app = FastAPI(title="Customer Churn Prediction API")
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_pipeline.joblib")
pipeline = None

@app.get("/")
def read_root():
    return {"message": "Welcome to Customer Churn Prediction API", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy" if pipeline is not None else "degraded", "model_loaded": pipeline is not None}
