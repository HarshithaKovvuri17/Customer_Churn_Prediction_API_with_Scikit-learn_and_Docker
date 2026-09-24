"""
Helper functions and constants for data loading, preprocessing, and API responses.
"""

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]

NUMERICAL_FEATURES = [
    'SeniorCitizen',
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

TARGET_COLUMN = 'Churn'

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw DataFrame by handling missing values in TotalCharges,
    converting data types, and ensuring consistent columns.
    """
    df = df.copy()

    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Handle TotalCharges missing or empty space strings
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill missing values (for new customers with tenure = 0) with 0.0 or median
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    # Ensure numerical types are numeric
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

def format_prediction_response(prediction_val: int, proba_val: float) -> dict:
    """
    Formats the raw model prediction and probability into API schema.
    """
    label = "Yes" if prediction_val == 1 else "No"
    return {
        "prediction": label,
        "probability": round(float(proba_val), 4)
    }
