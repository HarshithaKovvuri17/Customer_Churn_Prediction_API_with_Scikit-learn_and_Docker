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

def test_predict_missing_required_field():
    invalid_payload = VALID_CUSTOMER_PAYLOAD_1.copy()
    del invalid_payload["tenure"]
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
