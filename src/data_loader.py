import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='|',
    skipinitialspace=False,
    engine='python',
    skiprows=0 
    )
def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def check_structure(df: pd.DataFrame):
    return df.info(), df.dtypes

def save_raw_data(df: pd.DataFrame):
    df.to_csv('../../data/raw/raw_data.csv', index=False)

def check_missing(df: pd.DataFrame):
    return df.isnull().sum()

def extract_numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def extract_categorical_cols(df: pd.DataFrame):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def extract_date_time_cols(df: pd.DataFrame):
    return df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()

