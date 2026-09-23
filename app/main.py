import os
import joblib
from fastapi import FastAPI

app = FastAPI(title="Customer Churn Prediction API")
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_pipeline.joblib")
pipeline = None
