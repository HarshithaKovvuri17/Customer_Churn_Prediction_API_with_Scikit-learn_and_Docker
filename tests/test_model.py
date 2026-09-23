"""
Unit tests for data preprocessing and model engineering pipeline logic.
"""

import os
import pytest
import pandas as pd
import numpy as np
from app.model import load_data, build_pipeline, train_and_save_model, load_model_pipeline
from app.utils import clean_data, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN


def test_load_data():
    """
    Test loading dataset returns valid pandas DataFrame with required columns.
    """
    data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    assert os.path.exists(data_path), f"Dataset file missing at {data_path}"

    df = load_data(data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert TARGET_COLUMN in df.columns
    for col in NUMERICAL_FEATURES:
        assert col in df.columns
    for col in CATEGORICAL_FEATURES:
        assert col in df.columns


def test_clean_data_total_charges_spaces():
    """
    Test clean_data properly converts empty string ' ' in TotalCharges to numeric.
    """
    raw_sample = pd.DataFrame([{
        'customerID': '1234-TEST',
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 0,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 20.0,
        'TotalCharges': ' ',
        'Churn': 'No'
    }])

    cleaned = clean_data(raw_sample)
    assert 'customerID' not in cleaned.columns
    assert isinstance(cleaned['TotalCharges'].iloc[0], (float, int, np.floating))
    assert cleaned['TotalCharges'].iloc[0] == 0.0


def test_build_pipeline_structure():
    """
    Test model pipeline creation contains preprocessor and classifier steps.
    """
    pipeline = build_pipeline()
    assert 'preprocessor' in pipeline.named_steps
    assert 'classifier' in pipeline.named_steps


def test_train_and_save_model(tmp_path):
    """
    Test full model training workflow and pipeline artifact serialization.
    """
    model_file = os.path.join(tmp_path, "churn_pipeline_test.joblib")
    metrics = train_and_save_model(
        data_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
        model_save_path=model_file
    )

    assert os.path.exists(model_file)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] > 0.5

    # Test loading serialized model artifact
    loaded_pipeline = load_model_pipeline(model_file)
    assert loaded_pipeline is not None
