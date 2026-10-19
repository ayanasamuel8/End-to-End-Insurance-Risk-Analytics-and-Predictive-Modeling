import pandas as pd

TERM_MULTIPLIERS = {
    'monthly': 12,
    'quarterly': 4,
    'semi-annual': 2,
    'annual': 1,
}


def add_features(df: pd.DataFrame, reference_year: int = 2025) -> pd.DataFrame:
    """Return a copy of ``df`` with the derived pricing features added.

    ``TermFrequency`` text is matched case-insensitively. Values that are already
    numeric, or that are not recognised, fall back to a multiplier of 1.
    """
    df = df.copy()
    df['ClaimRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, 1)
    df['VehicleAge'] = reference_year - df['RegistrationYear']
    df['IsNew'] = (df['VehicleAge'] <= 1).astype(int)
    df['PowerPerCylinder'] = df['kilowatts'] / df['Cylinders'].replace(0, 1)
    df['IsHighValue'] = (df['CustomValueEstimate'] > df['CustomValueEstimate'].median()).astype(int)

    term = df['TermFrequency'].astype(str).str.strip().str.lower()
    df['TermFrequency'] = term.map(TERM_MULTIPLIERS).fillna(1)

    df['MonthlyPremium'] = df['CalculatedPremiumPerTerm'] / df['TermFrequency'].replace(0, 1)

    return df
