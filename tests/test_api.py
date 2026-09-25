"""
Integration tests for FastAPI RESTful API endpoints, request validation, and inference response formats.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Sample valid customer payload 1 (Likely Non-Churner)
VALID_CUSTOMER_PAYLOAD_1 = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 45,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 65.6,
    "TotalCharges": 2950.0,
    "customerID": "7590-VHVEG"
}

# Sample valid customer payload 2 (High Churn Risk)
VALID_CUSTOMER_PAYLOAD_2 = {
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 99.8,
    "TotalCharges": 99.8
}


def test_read_root():
    """
    Test GET / endpoint returns 200 OK and welcome status.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "message" in data


def test_health_check():
    """
    Test GET /health endpoint returns 200 OK.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_valid_customer_1():
    """
    Test POST /predict with valid customer payload 1 returns 200 OK
    and schema with prediction and probability fields.
    """
    response = client.post("/predict", json=VALID_CUSTOMER_PAYLOAD_1)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in ["Yes", "No"]
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_valid_customer_2():
    """
    Test POST /predict with valid customer payload 2.
    """
    response = client.post("/predict", json=VALID_CUSTOMER_PAYLOAD_2)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in ["Yes", "No"]
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_missing_required_field():
    """
    Test POST /predict with missing required field (e.g. tenure) returns 422 Unprocessable Entity.
    """
    invalid_payload = VALID_CUSTOMER_PAYLOAD_1.copy()
    del invalid_payload["tenure"]

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_invalid_data_type():
    """
    Test POST /predict with invalid field data type (e.g. string for integer tenure) returns 422.
    """
    invalid_payload = VALID_CUSTOMER_PAYLOAD_1.copy()
    invalid_payload["tenure"] = "invalid_string_instead_of_int"

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_invalid_senior_citizen_val():
    """
    Test POST /predict with out-of-bounds SeniorCitizen value (e.g. 5) returns 422.
    """
    invalid_payload = VALID_CUSTOMER_PAYLOAD_1.copy()
    invalid_payload["SeniorCitizen"] = 5

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
