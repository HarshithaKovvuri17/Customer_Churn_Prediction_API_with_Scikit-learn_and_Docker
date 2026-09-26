"""
FastAPI application defining RESTful API routes, schema validation, and prediction handling.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional, Union
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, field_validator

from app.utils import format_prediction_response, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# Global model variable loaded during lifespan startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_pipeline.joblib")
pipeline = None


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """
    Lifespan context manager for loading serialized model into memory on startup
    and managing app resources cleanly.
    """
    global pipeline
    if os.path.exists(MODEL_PATH):
        try:
            pipeline = joblib.load(MODEL_PATH)
            print(f"Model pipeline successfully loaded from '{MODEL_PATH}'")
        except Exception as e:
            print(f"Error loading model pipeline: {e}")
            pipeline = None
    else:
        print(f"Warning: Model file not found at '{MODEL_PATH}'. Ensure model is trained.")
    yield


# Initialize FastAPI App with lifespan context manager
app = FastAPI(
    title="Customer Churn Prediction API",
    description="RESTful API for real-time customer churn prediction using Scikit-Learn and FastAPI.",
    version="1.0.0",
    lifespan=lifespan
)


class CustomerData(BaseModel):
    """
    Pydantic schema for strict input validation of prediction requests.
    """
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

    @field_validator('SeniorCitizen')
    @classmethod
    def validate_senior_citizen(cls, v):
        if v not in (0, 1):
            raise ValueError("SeniorCitizen must be 0 or 1.")
        return v

    @field_validator('tenure')
    @classmethod
    def validate_tenure(cls, v):
        if v < 0:
            raise ValueError("tenure must be a non-negative integer.")
        return v


@app.get("/")
def read_root():
    """
    Root health check endpoint.
    """
    return {
        "message": "Welcome to Customer Churn Prediction API",
        "status": "healthy",
        "model_loaded": pipeline is not None
    }


@app.get("/health")
def health_check():
    """
    Health check status endpoint.
    """
    return {
        "status": "healthy" if pipeline is not None else "degraded",
        "model_loaded": pipeline is not None
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict_churn(data: CustomerData):
    """
    Predict customer churn risk for incoming customer profile payload.
    """
    global pipeline
    if pipeline is None:
        # Fallback reload attempt if pipeline wasn't loaded on startup
        if os.path.exists(MODEL_PATH):
            pipeline = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model pipeline is not loaded on server."
            )

    try:
        # Convert validated Pydantic model to dict
        input_dict = data.model_dump()
        input_dict.pop('customerID', None)

        # Convert to Pandas DataFrame
        df = pd.DataFrame([input_dict])

        # Execute prediction and probability
        pred_raw = pipeline.predict(df)[0]
        prob_raw = pipeline.predict_proba(df)[0][1]

        # Format exact JSON output response schema
        return format_prediction_response(pred_raw, prob_raw)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )
