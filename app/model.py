import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from app.utils import clean_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def load_data(filepath: str = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return clean_data(df)

def build_pipeline() -> Pipeline:
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, NUMERICAL_FEATURES), ('cat', cat_transformer, CATEGORICAL_FEATURES)])
    return Pipeline(steps=[('preprocessor', preprocessor)])
