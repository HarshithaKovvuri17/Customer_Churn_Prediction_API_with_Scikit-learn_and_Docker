# 📊 Customer Churn Prediction API with Scikit-learn & Docker

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![Pytest](https://img.shields.io/badge/Pytest-11%20Tests-0A9EDC?logo=pytest)
![License](https://img.shields.io/badge/License-MIT-green)

> An end-to-end machine learning application that predicts whether a telecommunications customer is likely to churn, exposes the trained model through a FastAPI REST API, validates incoming requests with Pydantic, and provides reproducible deployment using Docker.

---

# 🔎 Overview

Customer churn is an important business problem for telecommunications companies.

A customer may leave a service because of factors such as:

* Contract type
* Monthly charges
* Customer tenure
* Internet service
* Payment method
* Technical support
* Online security
* Additional services

This project uses **machine learning** to predict customer churn based on customer information.

The trained Scikit-learn model is exposed through a **FastAPI REST API**, allowing applications such as:

* Web applications
* Mobile applications
* Business dashboards
* Postman
* cURL
* Other backend services

to send customer information and receive a churn prediction.

The entire application can also be packaged and deployed using **Docker**.

---

# 🎯 Problem Statement

Telecommunication companies need to identify customers who are at a higher risk of leaving their services.

The objective of this project is to build a machine learning system that:

1. Accepts customer information.
2. Preprocesses numerical and categorical features.
3. Uses a trained Random Forest classifier.
4. Predicts whether the customer will churn.
5. Returns a churn probability.
6. Exposes the prediction through a REST API.
7. Validates incoming API requests.
8. Supports automated testing.
9. Runs consistently inside a Docker container.

---

# 🎯 Project Objectives

The main objectives are:

* Build a complete machine learning pipeline.
* Clean and preprocess the Telco Customer Churn dataset.
* Handle numerical and categorical features.
* Train a Random Forest classification model.
* Evaluate the model using standard classification metrics.
* Serialize the complete Scikit-learn pipeline.
* Build a REST API using FastAPI.
* Validate incoming customer data using Pydantic.
* Provide health-check endpoints.
* Containerize the application using Docker.
* Support Docker Compose deployment.
* Implement unit and integration tests using Pytest.

---

# ✨ Key Features

### 🤖 Machine Learning

* Random Forest classification
* Numerical feature preprocessing
* Categorical feature preprocessing
* Missing-value handling
* Feature scaling
* One-hot encoding
* Train/test split
* Model evaluation
* Serialized Scikit-learn pipeline

### 🌐 REST API

* FastAPI-based REST service
* `/` root endpoint
* `/health` health-check endpoint
* `/predict` prediction endpoint
* Automatic Swagger documentation
* JSON request/response format

### 🛡️ Input Validation

Pydantic validates incoming customer information before it reaches the machine learning pipeline.

The API validates values such as:

* `SeniorCitizen`
* `tenure`
* `MonthlyCharges`
* `TotalCharges`
* Required customer fields
* Data types

### 🐳 Deployment

* Dockerfile included
* Docker Compose configuration included
* Port `8000`
* Environment-variable based model configuration
* Container health verification

### 🧪 Testing

The project includes:

* Unit tests
* API integration tests
* Input validation tests
* Model pipeline tests
* Model serialization tests

---

# 🏗️ System Architecture

```text
                         ┌───────────────────────┐
                         │      API Client       │
                         │                       │
                         │ Postman / cURL / Web  │
                         └───────────┬───────────┘
                                     │
                                     │ HTTP Request
                                     ▼
                         ┌───────────────────────┐
                         │      FastAPI API      │
                         │       main.py         │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Pydantic Validation │
                         │     CustomerData      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Pandas DataFrame   │
                         └───────────┬───────────┘
                                     │
                                     ▼
              ┌─────────────────────────────────────────┐
              │       Scikit-learn Pipeline             │
              │                                         │
              │  ┌───────────────────────────────────┐  │
              │  │      ColumnTransformer            │  │
              │  │                                   │  │
              │  │ Numerical Features                │  │
              │  │ → Imputation                      │  │
              │  │ → StandardScaler                  │  │
              │  │                                   │  │
              │  │ Categorical Features              │  │
              │  │ → Imputation                      │  │
              │  │ → OneHotEncoder                   │  │
              │  └──────────────────┬────────────────┘  │
              │                     │                   │
              │                     ▼                   │
              │        ┌────────────────────────┐      │
              │        │ RandomForestClassifier │      │
              │        └────────────┬───────────┘      │
              └─────────────────────┼──────────────────┘
                                    │
                                    ▼
                         ┌───────────────────────┐
                         │ Prediction +          │
                         │ Churn Probability     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │      JSON Response    │
                         └───────────────────────┘
```

---

# 🔄 Machine Learning Workflow

The complete ML workflow is:

```text
Raw CSV Dataset
       │
       ▼
Data Loading
       │
       ▼
Data Cleaning
       │
       ▼
Feature Selection
       │
       ▼
Train / Test Split
       │
       ▼
Preprocessing Pipeline
       │
       ├── Numerical Features
       │      ├── Missing Value Imputation
       │      └── StandardScaler
       │
       └── Categorical Features
              ├── Missing Value Imputation
              └── OneHotEncoder
       │
       ▼
Random Forest Classifier
       │
       ▼
Model Evaluation
       │
       ▼
Pipeline Serialization
       │
       ▼
churn_pipeline.joblib
       │
       ▼
FastAPI Inference
```

---

# 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

Dataset file:

```text
data/
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contains customer information related to:

* Demographics
* Account information
* Services
* Billing
* Contract information
* Customer tenure
* Churn status

The target variable is:

```text
Churn
```

with two possible values:

```text
Yes
No
```

---

# 🧮 Features

## Numerical Features

The model uses:

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
```

### Numerical preprocessing

```text
Missing Values
      │
      ▼
Median Imputation
      │
      ▼
StandardScaler
```

---

## Categorical Features

The categorical features include:

```text
gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
```

### Categorical preprocessing

```text
Missing Values
      │
      ▼
Most Frequent Imputation
      │
      ▼
OneHotEncoder
      │
      ▼
handle_unknown = ignore
```

---

# 🤖 Model & Preprocessing

The project uses:

```text
RandomForestClassifier
```

Configuration:

```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42
)
```

The preprocessing and classifier are combined into one Scikit-learn `Pipeline`.

This is important because the same preprocessing used during training is automatically applied during prediction.

The complete pipeline is saved as:

```text
models/churn_pipeline.joblib
```

---

# 📈 Model Evaluation

The model is evaluated using a 20% test split.

The training workflow uses:

```text
train_test_split(
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The current documented training run produced approximately:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **79.91%** |
| Precision | **65.12%** |
| Recall    | **52.41%** |
| F1 Score  | **58.07%** |

### Metric Explanation

**Accuracy**

Measures the percentage of total predictions that are correct.

**Precision**

Measures how many customers predicted as churners actually churn.

**Recall**

Measures how many actual churners were successfully identified.

**F1 Score**

Provides a balance between precision and recall.

> These values are model-evaluation results for the documented training run. Results can change if the dataset, preprocessing, model configuration, or training process changes.

---

# 🌐 REST API

The FastAPI application provides three main endpoints.

| Method | Endpoint   | Purpose                    |
| ------ | ---------- | -------------------------- |
| `GET`  | `/`        | API root/status            |
| `GET`  | `/health`  | Health and model readiness |
| `POST` | `/predict` | Predict customer churn     |

---

# 🏠 Root Endpoint

### Request

```http
GET /
```

### Example Response

```json
{
  "message": "Welcome to Customer Churn Prediction API",
  "status": "healthy",
  "model_loaded": true
}
```

---

# ❤️ Health Endpoint

### Request

```http
GET /health
```

### Example Response

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

This endpoint can be used by developers, monitoring systems, or deployment platforms to verify that the API and model are available.

---

# 🔮 Prediction Endpoint

### Request

```http
POST /predict
```

The endpoint accepts customer information as JSON.

### Example Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85,
  "customerID": "7590-VHVEG"
}
```

### Example Response

```json
{
  "prediction": "No",
  "probability": 0.3867
}
```

Where:

```text
prediction  → predicted churn status
probability → probability of churn
```

---

# 📡 cURL Example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Female",
       "SeniorCitizen": 0,
       "Partner": "Yes",
       "Dependents": "No",
       "tenure": 1,
       "PhoneService": "No",
       "MultipleLines": "No phone service",
       "InternetService": "DSL",
       "OnlineSecurity": "No",
       "OnlineBackup": "Yes",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "No",
       "StreamingMovies": "No",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 29.85,
       "TotalCharges": 29.85
     }'
```

---

# 📖 Swagger API Documentation

FastAPI automatically generates interactive API documentation.

After starting the server, open:

```text
http://127.0.0.1:8000/docs
```

You can use Swagger UI to:

1. View available endpoints.
2. View request schemas.
3. Enter customer information.
4. Execute `/predict`.
5. View the JSON response.

Alternative documentation:

```text
http://127.0.0.1:8000/redoc
```

---

# 📁 Project Structure

```text
Customer_Churn_Prediction_API_with_Scikit-learn_and_Docker/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── utils.py
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_pipeline.joblib
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── conftest.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── testing.md
├── .env.example
└── README.md
```

---

# 🧩 Important Files

### `app/main.py`

Contains:

* FastAPI application
* API endpoints
* Pydantic request model
* Input validation
* Model loading
* Prediction logic
* Error handling

---

### `app/model.py`

Contains:

* Dataset loading
* Pipeline construction
* Model training
* Model evaluation
* Model serialization
* Model loading

---

### `app/utils.py`

Contains:

* Feature definitions
* Data cleaning
* Prediction response formatting

---

### `models/churn_pipeline.joblib`

Serialized machine learning pipeline containing:

```text
Preprocessing
     +
Random Forest Model
```

---

### `tests/test_api.py`

Tests:

* Root endpoint
* Health endpoint
* Valid predictions
* Missing fields
* Invalid data types
* Invalid feature values

---

### `tests/test_model.py`

Tests:

* Dataset loading
* Data cleaning
* Pipeline structure
* Model training
* Model serialization
* Model loading

---

# 🛠️ Technology Stack

| Technology     | Purpose                 |
| -------------- | ----------------------- |
| Python         | Programming language    |
| Pandas         | Data manipulation       |
| NumPy          | Numerical operations    |
| Scikit-learn   | Machine learning        |
| FastAPI        | REST API                |
| Pydantic       | Request validation      |
| Uvicorn        | ASGI server             |
| Joblib         | Model serialization     |
| Pytest         | Automated testing       |
| Docker         | Containerization        |
| Docker Compose | Container orchestration |

---

# 💻 Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/Customer_Churn_Prediction_API_with_Scikit-learn_and_Docker.git
```

Move into the project:

```bash
cd Customer_Churn_Prediction_API_with_Scikit-learn_and_Docker
```

---

# 2. Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Train the Model

Run:

```bash
python -m app.model
```

The model will:

1. Load the dataset.
2. Clean the data.
3. Separate features and target.
4. Split the dataset.
5. Build the preprocessing pipeline.
6. Train the Random Forest model.
7. Calculate evaluation metrics.
8. Save the trained pipeline.

The serialized model will be created at:

```text
models/churn_pipeline.joblib
```

---

# 🚀 Run the API

Start the FastAPI application:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

# 🐳 Docker Deployment

Docker allows the application to run in an isolated and reproducible environment.

## Build the Docker Image

```bash
docker build -t churn-prediction-api .
```

---

## Run the Docker Container

```bash
docker run -d \
  -p 8000:8000 \
  --name churn_api \
  churn-prediction-api
```

Check running containers:

```bash
docker ps
```

View logs:

```bash
docker logs churn_api
```

---

# 🐳 Docker Compose

Docker Compose provides a simpler way to build and start the application.

Run:

```bash
docker-compose up --build
```

For detached mode:

```bash
docker-compose up --build -d
```

Check containers:

```bash
docker ps
```

View logs:

```bash
docker-compose logs --tail 50
```

Stop the service:

```bash
docker-compose down
```

---

# 🔍 Verify Docker Deployment

After starting the container, open:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# 🧪 Testing

The project contains automated tests using **Pytest**.

Run all tests:

```bash
python -m pytest -v
```

or:

```bash
pytest -v
```

The documented test suite contains **11 automated tests** covering:

```text
API endpoints
     │
     ├── Root endpoint
     ├── Health endpoint
     ├── Valid prediction
     ├── Invalid payload
     └── Data validation

Model pipeline
     │
     ├── Dataset loading
     ├── Data cleaning
     ├── Pipeline structure
     ├── Model training
     ├── Model serialization
     └── Model loading
```

Expected result:

```text
======================== 11 passed ========================
```

For detailed manual testing commands, see:

```text
testing.md
```

---

# 🧪 Example Testing Workflow

A complete test can be performed as follows:

### Step 1 — Start the application

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Step 2 — Check health

```text
GET /health
```

Expected:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Step 3 — Send customer data

```text
POST /predict
```

### Step 4 — Receive prediction

```json
{
  "prediction": "Yes",
  "probability": 0.8419
}
```

### Step 5 — Run automated tests

```bash
python -m pytest -v
```

---

# 🛡️ Error Handling & Validation

The API uses Pydantic to validate incoming requests.

For example, `SeniorCitizen` must be:

```text
0 or 1
```

Invalid:

```json
{
  "SeniorCitizen": 5
}
```

will result in a validation error.

Similarly, `tenure` cannot be negative.

Invalid:

```json
{
  "tenure": -5
}
```

will also fail validation.

Missing required fields produce:

```text
422 Unprocessable Entity
```

This prevents invalid data from reaching the machine learning pipeline.

---

# ⚙️ Environment Variables

The application supports configurable environment variables.

| Variable      | Default                        | Purpose                   |
| ------------- | ------------------------------ | ------------------------- |
| `MODEL_PATH`  | `models/churn_pipeline.joblib` | Location of trained model |
| `PORT`        | `8000`                         | API port                  |
| `HOST`        | `0.0.0.0`                      | API host                  |
| `ENVIRONMENT` | `production` in Docker         | Runtime environment       |

Example:

```env
MODEL_PATH=models/churn_pipeline.joblib
PORT=8000
HOST=0.0.0.0
ENVIRONMENT=production
```

---

# 🔐 Why Serialize the Complete Pipeline?

Instead of saving only the Random Forest model, this project saves the complete Scikit-learn pipeline.

```text
Input Data
    │
    ▼
Preprocessing
    │
    ├── Numerical Transformation
    │
    └── Categorical Transformation
    │
    ▼
Random Forest
    │
    ▼
Prediction
```

This provides an important benefit:

> The exact preprocessing logic used during training is reused during inference.

Therefore, the API does not need to manually recreate the preprocessing steps every time a prediction is requested.

---

# 🧠 MLOps Concepts Demonstrated

This project demonstrates several practical machine learning engineering concepts:

### 1. Reproducible Training

The model uses:

```python
random_state=42
```

to make the training process reproducible.

### 2. Pipeline-Based ML

Preprocessing and model training are combined into one pipeline.

### 3. Model Serialization

The trained pipeline is stored using Joblib.

### 4. API-Based Inference

The model can be accessed through HTTP requests.

### 5. Input Validation

Pydantic protects the API from invalid request data.

### 6. Automated Testing

Pytest verifies both model and API behavior.

### 7. Containerization

Docker packages the application and its dependencies together.

---

# 🔄 Complete End-to-End Flow

```text
                    ┌─────────────────────┐
                    │ Telco Churn Dataset │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Data Cleaning    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Train Random Forest │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    └──────────┬──────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ churn_pipeline.joblib    │
                  └────────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FastAPI Server   │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
             GET /health              POST /predict
                  │                         │
                  │                         ▼
                  │                Pydantic Validation
                  │                         │
                  │                         ▼
                  │                ML Pipeline Inference
                  │                         │
                  │                         ▼
                  │                 Prediction + Probability
                  │
                  ▼
             API Status
```

---

# 🚀 Future Enhancements

Possible improvements for future versions include:

* Add authentication and authorization.
* Add request logging.
* Add structured application logging.
* Add model versioning.
* Add experiment tracking with MLflow.
* Add CI/CD using GitHub Actions.
* Add model monitoring.
* Add data drift detection.
* Add prediction history storage.
* Add a web-based frontend.
* Add batch prediction support.
* Add cloud deployment.
* Add automated model retraining.
* Add performance monitoring.
* Add API rate limiting.

---

# 📌 Production Considerations

For production deployment, the following improvements should be considered:

### Security

* Add API authentication.
* Use HTTPS.
* Validate and sanitize all inputs.
* Avoid exposing sensitive customer information.

### Monitoring

Monitor:

* API latency
* Request volume
* Error rate
* Model prediction distribution
* Model performance
* Data drift

### Model Management

Maintain:

```text
Model Version
Training Dataset Version
Training Date
Model Metrics
Feature Version
Deployment Version
```

This makes the ML system easier to reproduce and maintain.

---

# 💡 Use Cases

This API can be integrated into:

### Customer Service Platforms

Identify customers who may require proactive support.

### Business Dashboards

Display churn-risk information to business teams.

### CRM Systems

Add churn prediction to customer profiles.

### Retention Campaigns

Use predicted churn risk to support targeted retention workflows.

### Web Applications

A frontend can send customer information to `/predict` and display the result.

---

# 📚 Learning Outcomes

By completing this project, you gain practical experience with:

* Machine Learning classification
* Random Forest
* Scikit-learn pipelines
* Feature preprocessing
* Pandas
* FastAPI
* REST APIs
* Pydantic
* Model serialization
* Pytest
* Docker
* Docker Compose
* API testing
* MLOps fundamentals

---

# 🏁 Quick Command Reference

## Local Development

```bash
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

```bash
python -m app.model
```

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## Testing

```bash
python -m pytest -v
```

---

## Docker

```bash
docker build -t churn-prediction-api .
```

```bash
docker run -d -p 8000:8000 --name churn_api churn-prediction-api
```

---

## Docker Compose

```bash
docker-compose up --build -d
```

```bash
docker-compose logs --tail 50
```

```bash
docker-compose down
```

---

# 📍 API URLs

After starting the application:

| Resource       | URL                             |
| -------------- | ------------------------------- |
| API Root       | `http://127.0.0.1:8000/`        |
| Health Check   | `http://127.0.0.1:8000/health`  |
| Swagger UI     | `http://127.0.0.1:8000/docs`    |
| ReDoc          | `http://127.0.0.1:8000/redoc`   |
| Prediction API | `http://127.0.0.1:8000/predict` |

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👩‍💻 Author

**Harshitha**

Built as an end-to-end Machine Learning and API deployment project using:

```text
Python
Scikit-learn
FastAPI
Pandas
Pydantic
Pytest
Docker
Docker Compose
```

