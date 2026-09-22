import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.utils import clean_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

def load_data(filepath: str = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return clean_data(df)

def build_pipeline() -> Pipeline:
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, NUMERICAL_FEATURES), ('cat', cat_transformer, CATEGORICAL_FEATURES)])
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, max_depth=12))])

def train_and_save_model(data_path: str = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv', model_save_path: str = 'models/churn_pipeline.joblib') -> dict:
    df = load_data(data_path)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET_COLUMN] == 'Yes').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {"accuracy": float(accuracy_score(y_test, y_pred)), "precision": float(precision_score(y_test, y_pred, zero_division=0)), "recall": float(recall_score(y_test, y_pred, zero_division=0)), "f1_score": float(f1_score(y_test, y_pred, zero_division=0))}
