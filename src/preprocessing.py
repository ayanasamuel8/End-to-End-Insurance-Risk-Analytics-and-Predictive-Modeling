import pandas as pd
import numpy as np
from scipy import stats
from src.data_loader import extract_numeric_cols, extract_categorical_cols, extract_date_time_cols


def clean_numeric_strings(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Convert string-formatted numbers with commas to proper floats.
    """
    for col in cols:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by performing the following steps:
      - Drop duplicate records.
      - Fill missing numerical values with the median.
      - Standardize and fill missing categorical values with the mode.
      - Clean numeric strings (e.g., "1,000").
      - Cap numerical outliers between 1st and 99th percentile.
      - Convert datetime columns.
    
    Returns the cleaned DataFrame.
    """
    # 1. Remove Duplicate Rows
    df_clean = df.drop_duplicates().copy()
    print(print(df['TotalClaims'].value_counts()))

    # 2. Handle Numeric Columns
    num_cols = extract_numeric_cols(df_clean)
    print('TotalClaims' in num_cols)
    for col in num_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

    # 3. Handle Categorical Columns
    cat_cols = extract_categorical_cols(df_clean)
    for col in cat_cols:
        # Ensure string formatting and remove invalid 'nan' strings
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        df_clean[col] = df_clean[col].replace('nan', np.nan)

    for col in cat_cols:
        if df_clean[col].isnull().all():
            continue  # Skip if entire column is NaN
        mode_vals = df_clean[col].mode()
        if not mode_vals.empty:
            mode_val = mode_vals[0]
            df_clean[col] = df_clean[col].fillna(mode_val)

    # 4. Clean Numeric Strings
    numeric_string_cols = [
        col for col in df_clean.columns
        if df_clean[col].dtype == 'object' and df_clean[col].str.contains(',', na=False).any()
    ]
    df_clean = clean_numeric_strings(df_clean, numeric_string_cols)
    print('after string clean:')
    print(print(df['TotalClaims'].value_counts()))

    # 5. Cap Outliers to 1st and 99th percentiles
    for col in num_cols:
        if col != 'TotalClaims':
            lower, upper = df_clean[col].quantile([0.01, 0.99])
            df_clean[col] = np.clip(df_clean[col], lower, upper)

    # 6. Convert Date/Time Columns
    date_time_cols = extract_date_time_cols(df_clean)
    for col in date_time_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    df_clean = df_clean.dropna(thresh=len(df) * 0.2, axis=1)
    # df_clean.drop_duplicates(inplace=True)

    return df_clean
def save_cleaned_data(df: pd.DataFrame, file_path: str = '../../data/cleaned/cleaned_data.csv') -> None:
    """
    Save the cleaned DataFrame to a CSV file.
    """
    df.to_csv(file_path, index=False)
    print(f"Cleaned data saved to {file_path}")