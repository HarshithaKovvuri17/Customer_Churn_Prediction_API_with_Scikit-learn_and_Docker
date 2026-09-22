import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from app.utils import clean_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def load_data(filepath: str = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return clean_data(df)

def build_pipeline() -> Pipeline:
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, NUMERICAL_FEATURES)])
    return Pipeline(steps=[('preprocessor', preprocessor)])
