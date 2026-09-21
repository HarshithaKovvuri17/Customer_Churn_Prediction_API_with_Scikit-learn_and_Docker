import pandas as pd
import numpy as np

CATEGORICAL_FEATURES = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
NUMERICAL_FEATURES = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
TARGET_COLUMN = 'Churn'

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    return df
