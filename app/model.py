import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from app.utils import clean_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def load_data(filepath: str = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return clean_data(df)

def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[])
    return Pipeline(steps=[('preprocessor', preprocessor)])
