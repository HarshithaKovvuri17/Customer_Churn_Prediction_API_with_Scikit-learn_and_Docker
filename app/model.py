import os
import pandas as pd
from app.utils import clean_data

def load_data(filepath: str = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return clean_data(df)
