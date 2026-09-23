import os
import pytest
import pandas as pd
import numpy as np
from app.model import load_data, build_pipeline, train_and_save_model, load_model_pipeline
from app.utils import clean_data, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN

def test_load_data():
    df = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    assert isinstance(df, pd.DataFrame)

def test_clean_data_total_charges_spaces():
    raw_sample = pd.DataFrame([{'customerID': '1234', 'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No', 'tenure': 0, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check', 'MonthlyCharges': 20.0, 'TotalCharges': ' ', 'Churn': 'No'}])
    cleaned = clean_data(raw_sample)
    assert cleaned['TotalCharges'].iloc[0] == 0.0

def test_build_pipeline_structure():
    pipeline = build_pipeline()
    assert 'preprocessor' in pipeline.named_steps
    assert 'classifier' in pipeline.named_steps
