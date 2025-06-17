import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from src.preprocessing.cleaning import clean_data
from src.task_4.feature_engineering import add_features  # you defined this earlier

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label encode all categorical columns, safely handling NaNs and type issues.
    """
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        # If NaNs exist, temporarily fill with placeholder
        if df_encoded[col].isnull().any():
            df_encoded[col] = df_encoded[col].fillna('___missing___')

        try:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        except (ValueError, TypeError):
            # Fallback for very dirty columns
            df_encoded[col] = df_encoded[col].astype('category').cat.codes

    return df_encoded

def is_date_column(series: pd.Series) -> bool:
    try:
        parsed = pd.to_datetime(series, errors='coerce')
        non_null_ratio = parsed.notna().mean()
        return non_null_ratio > 0.9  # 90%+ of values must be valid dates
    except:
        return False




def prepare_claim_severity_data(df):
    target = 'TotalClaims'
    drop_cols = ['UnderwrittenCoverID', 'PolicyID', 'claim_indicator']

    # Filter only likely date columns
    date_cols = [col for col in df.columns if df[col].dtype == 'object' and is_date_column(df[col])]

    # Convert and extract
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df.drop(columns=[col], inplace=True)

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df.drop(columns=[col], inplace=True)

    # Keep only positive-claim rows
    df = df[df[target] > 0]

    # Encode categoricals
    df = encode_categoricals(df)

    # Drop unnecessary columns
    df = df.drop(columns=drop_cols)



    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)





def prepare_claim_probability_data(df: pd.DataFrame) -> tuple:
    """
    Load, clean, engineer, encode, and split the data for claim probability classification.
    Assumes a `claim_indicator` column exists (boolean or binary 0/1).
    """

    # Ensure claim_indicator is binary (0/1)
    if df['claim_indicator'].dtype == 'bool':
        df['claim_indicator'] = df['claim_indicator'].astype(int)
    elif df['claim_indicator'].nunique() == 2 and sorted(df['claim_indicator'].unique()) == [False, True]:
        df['claim_indicator'] = df['claim_indicator'].map({False: 0, True: 1})

    df = encode_categoricals(df)
    df = df.dropna()

    X = df.drop(columns=['claim_indicator'])
    y = df['claim_indicator']

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

