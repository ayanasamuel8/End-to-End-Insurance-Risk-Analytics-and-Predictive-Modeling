import pandas as pd
def clean_numeric_strings(df: pd.DataFrame, cols: list):
    """
    Convert string-formatted numbers with commas to proper floats.
    """
    for col in cols:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '', regex=False)  # Remove thousands separator
            df[col] = pd.to_numeric(df[col], errors='coerce')    # Convert to float
    return df
