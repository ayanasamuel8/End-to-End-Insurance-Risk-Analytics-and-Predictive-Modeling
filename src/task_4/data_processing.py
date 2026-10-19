import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

SEVERITY_TARGET = 'TotalClaims'
PROBABILITY_TARGET = 'claim_indicator'

# A policy appears in many rows. Rows of one policy always land on the same side of a split.
GROUP_COLUMN = 'PolicyID'

# Identifiers carry no pricing signal. ClaimRatio is derived from the claim amount.
SEVERITY_DROP = ['UnderwrittenCoverID', 'claim_indicator', 'ClaimRatio']
# TotalClaims decides claim_indicator, so keeping it would leak the answer.
PROBABILITY_DROP = ['UnderwrittenCoverID', 'TotalClaims', 'ClaimRatio']

MISSING_TOKEN = '___missing___'
DATE_PATTERN = r'^\d{4}-\d{2}-\d{2}'


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace every text or categorical column with integer codes.

    Codes follow the sorted order of the distinct values, so the result does not
    depend on row order. Missing values get their own code.
    """
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        values = df_encoded[col].astype(object).where(df_encoded[col].notna(), MISSING_TOKEN).astype(str)
        df_encoded[col] = pd.Categorical(values, categories=sorted(values.unique())).codes
    for col in df_encoded.select_dtypes(include=['bool']).columns:
        df_encoded[col] = df_encoded[col].astype(int)
    return df_encoded


def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def is_date_column(series: pd.Series) -> bool:
    """True when at least 90% of the values start with an ISO date (YYYY-MM-DD)."""
    if not _is_text(series):
        return False
    values = series.dropna().astype(str)
    if values.empty:
        return False
    return values.str.match(DATE_PATTERN).mean() > 0.9


def expand_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace each ISO date text column with ``<col>_year`` and ``<col>_month``."""
    df = df.copy()
    for col in [c for c in df.columns if is_date_column(df[c])]:
        parsed = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_year'] = parsed.dt.year
        df[f'{col}_month'] = parsed.dt.month
        df = df.drop(columns=[col])
    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how='all')


def build_severity_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a positive claim, numeric features, and the target column."""
    df = drop_empty_columns(expand_date_columns(df))
    df = df[df[SEVERITY_TARGET] > 0]
    df = encode_categoricals(df)
    return df.drop(columns=[c for c in SEVERITY_DROP if c in df.columns])


def build_probability_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All rows, numeric features without claim-derived columns, and the 0/1 target."""
    df = df.copy()
    if PROBABILITY_TARGET not in df.columns:
        df[PROBABILITY_TARGET] = df[SEVERITY_TARGET] > 0
    df[PROBABILITY_TARGET] = df[PROBABILITY_TARGET].astype(int)
    df = drop_empty_columns(expand_date_columns(df))
    df = encode_categoricals(df)
    df = df.drop(columns=[c for c in PROBABILITY_DROP if c in df.columns])
    return df.dropna()


def _split(frame: pd.DataFrame, target: str, test_size: float, random_state: int, stratify: bool):
    """
    Split ``frame`` into train and test parts.

    When the frame has a ``PolicyID`` column, all rows of a policy go to the same
    side. Returns X_train, X_test, y_train, y_test, groups_train, groups_test
    (the group lists are None when there is no ``PolicyID``).
    """
    y = frame[target]
    if GROUP_COLUMN not in frame.columns:
        X = frame.drop(columns=[target])
        strata = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=strata, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, None, None

    groups = frame[GROUP_COLUMN]
    X = frame.drop(columns=[target, GROUP_COLUMN])
    if stratify:
        splitter = StratifiedGroupKFold(n_splits=round(1 / test_size), shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups))
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx],
        groups.iloc[train_idx], groups.iloc[test_idx],
    )


def split_claim_severity_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Positive-claim rows split by policy. Returns the 6-part result of ``_split``."""
    return _split(build_severity_frame(df), SEVERITY_TARGET, test_size, random_state, stratify=False)


def split_claim_probability_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """All rows split by policy with similar claim rates on both sides."""
    return _split(build_probability_frame(df), PROBABILITY_TARGET, test_size, random_state, stratify=True)


def prepare_claim_severity_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split positive-claim rows into train and test sets for severity regression."""
    return split_claim_severity_data(df, test_size, random_state)[:4]


def prepare_claim_probability_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split all rows into train and test sets for claim classification."""
    return split_claim_probability_data(df, test_size, random_state)[:4]
