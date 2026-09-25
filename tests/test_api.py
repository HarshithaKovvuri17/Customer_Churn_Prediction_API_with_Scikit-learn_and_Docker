import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_CUSTOMER_PAYLOAD_1 = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 45,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.6, "TotalCharges": 2950.0, "customerID": "7590-VHVEG"
}

VALID_CUSTOMER_PAYLOAD_2 = {
    "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No", "tenure": 1,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 99.8, "TotalCharges": 99.8
}

def test_predict_valid_customer_1():
    response = client.post("/predict", json=VALID_CUSTOMER_PAYLOAD_1)
    assert response.status_code == 200

def test_predict_valid_customer_2():
    response = client.post("/predict", json=VALID_CUSTOMER_PAYLOAD_2)
    assert response.status_code == 200
