import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def check_structure(df: pd.DataFrame):
    return df.info(), df.dtypes

def check_missing(df: pd.DataFrame):
    return df.isnull().sum()
