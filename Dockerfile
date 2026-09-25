# Use official lightweight Python image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV MODEL_PATH=models/churn_pipeline.joblib

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application directories
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

# Expose port 8000
EXPOSE 8000

# Start Uvicorn web server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
