"""
Model training, evaluation, loading, and serialization module.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.utils import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN, clean_data


def load_data(filepath: str = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    """
    Load Telco Customer Churn dataset from CSV and clean data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_csv(filepath)
    df = clean_data(df)
    return df


def build_pipeline() -> Pipeline:
    """
    Constructs a Scikit-Learn Pipeline encapsulating ColumnTransformer
    preprocessing (StandardScaler, OneHotEncoder) and RandomForestClassifier.
    """
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, NUMERICAL_FEATURES),
            ('cat', cat_transformer, CATEGORICAL_FEATURES)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, max_depth=12))
    ])

    return pipeline


def train_and_save_model(
    data_path: str = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    model_save_path: str = 'models/churn_pipeline.joblib'
) -> dict:
    """
    Trains the churn prediction model, evaluates it, prints metrics,
    and serializes the full pipeline artifact to disk.
    """
    df = load_data(data_path)

    # 1. Define Features (X) and Target (y)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET_COLUMN] == 'Yes').astype(int)

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Create full pipeline
    pipeline = build_pipeline()

    # 4. Train model pipeline
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on test set
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
    }

    print("Model Evaluation Metrics on Test Set:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.capitalize()}: {value:.4f}")

    # 6. Save serialized pipeline
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    print(f"Saved trained pipeline artifact to: {model_save_path}")

    return metrics


def load_model_pipeline(model_path: str = 'models/churn_pipeline.joblib') -> Pipeline:
    """
    Loads a saved serialized model pipeline from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)


if __name__ == '__main__':
    train_and_save_model()
