# Testing Commands & Verification Guide (PowerShell & Bash)

This document contains all commands and PowerShell scripts required to set up, test, verify, and evaluate the **Customer Churn Prediction API** across Docker containers, automated tests, model training, and live REST endpoints.

---

## 🐳 1. Docker & Docker Compose Container Deployment

### A. Option 1: Single-Command Setup (Docker Compose - Recommended)

```powershell
# Build image and start containerized API service on port 8000
docker-compose up --build -d

# Check running containers
docker ps

# View container logs
docker-compose logs --tail 50

# Stop container service
docker-compose down
```

### B. Option 2: Standalone Docker Commands

```powershell
# Build Docker Image
docker build -t churn-prediction-api .

# Remove existing container if present (prevents name conflict)
docker rm -f churn_api_container

# Run Docker Container on Port 8000
docker run -d -p 8000:8000 --name churn_api_container churn-prediction-api

# Inspect Container Logs
docker logs churn_api_container --tail 50

# Stop & Remove Container when finished
docker stop churn_api_container
docker rm churn_api_container
```

---

## 🤖 2. Model Training & Pipeline Serialization

Train the Scikit-learn model pipeline on `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` and save `models/churn_pipeline.joblib`:

```powershell
python -m app.model
```

**Expected Output:**
```text
Model Evaluation Metrics on Test Set:
  Accuracy: 0.7991
  Precision: 0.6512
  Recall: 0.5241
  F1_score: 0.5807
Saved trained pipeline artifact to: models/churn_pipeline.joblib
```

---

## 🧪 3. Running Pytest Suite (PowerShell)

Run all 11 automated unit and integration tests:

```powershell
# Run all tests using python module syntax
python -m pytest -v

# Or run pytest executable directly
.\.venv\Scripts\pytest.exe -v
```

**Expected Output:**
```text
======================== 11 passed in 3.77s ========================
```

---

## 🌐 4. Local Development Environment & API Startup

```powershell
# Create & Activate Virtual Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Start Uvicorn Server locally
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

> [!NOTE]
> Running with `--host 127.0.0.1` binds the server locally, allowing direct access in your web browser at **`http://127.0.0.1:8000`** or **`http://localhost:8000`**.

### 🔗 Web Browser Links:
- **Interactive Swagger Documentation**: 👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
- **API Health Check**: 👉 **[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)**
- **API Root**: 👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📡 5. PowerShell API Request Commands (`Invoke-RestMethod`)

### A. Health Check (`GET /health`)

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

**Expected Output:**
```json
status  model_loaded
------  ------------
healthy         True
```

---

### B. High Churn Risk Customer (`POST /predict`)

```powershell
$body = @{
    gender = "Female"
    SeniorCitizen = 0
    Partner = "No"
    Dependents = "No"
    tenure = 1
    PhoneService = "Yes"
    MultipleLines = "No"
    InternetService = "Fiber optic"
    OnlineSecurity = "No"
    OnlineBackup = "No"
    DeviceProtection = "No"
    TechSupport = "No"
    StreamingTV = "Yes"
    StreamingMovies = "Yes"
    Contract = "Month-to-month"
    PaperlessBilling = "Yes"
    PaymentMethod = "Electronic check"
    MonthlyCharges = 89.85
    TotalCharges = 89.85
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

**Expected Output:**
```json
prediction probability
---------- -----------
Yes             0.8419
```

---

### C. Low Churn Risk Customer (`POST /predict`)

```powershell
$body = @{
    gender = "Male"
    SeniorCitizen = 0
    Partner = "Yes"
    Dependents = "Yes"
    tenure = 60
    PhoneService = "Yes"
    MultipleLines = "Yes"
    InternetService = "DSL"
    OnlineSecurity = "Yes"
    OnlineBackup = "Yes"
    DeviceProtection = "Yes"
    TechSupport = "Yes"
    StreamingTV = "No"
    StreamingMovies = "No"
    Contract = "Two year"
    PaperlessBilling = "No"
    PaymentMethod = "Bank transfer (automatic)"
    MonthlyCharges = 64.20
    TotalCharges = 3850.00
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

**Expected Output:**
```json
prediction probability
---------- -----------
No              0.0091
```

---

### D. Invalid Payload Validation Test - Missing Field (`POST /predict`)

```powershell
$body = @{
    gender = "Female"
    SeniorCitizen = 0
    # tenure is omitted intentionally to trigger validation error
    Partner = "No"
    Dependents = "No"
    PhoneService = "Yes"
    MultipleLines = "No"
    InternetService = "DSL"
    OnlineSecurity = "No"
    OnlineBackup = "No"
    DeviceProtection = "No"
    TechSupport = "No"
    StreamingTV = "No"
    StreamingMovies = "No"
    Contract = "Month-to-month"
    PaperlessBilling = "Yes"
    PaymentMethod = "Electronic check"
    MonthlyCharges = 29.85
    TotalCharges = 29.85
} | ConvertTo-Json

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $body -ContentType "application/json"
} catch {
    $_.Exception.Response
}
```

**Expected Output (`422 Unprocessable Content`):**
```text
StatusCode        : 422
StatusDescription : Unprocessable Content
ResponseUri       : http://127.0.0.1:8000/predict
Method            : POST
```
