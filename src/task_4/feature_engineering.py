import pandas as pd
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['ClaimRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, 1)
    df['VehicleAge'] = 2025 - df['RegistrationYear']
    df['IsNew'] = (df['VehicleAge'] <= 1).astype(int)
    df['PowerPerCylinder'] = df['kilowatts'] / df['Cylinders'].replace(0, 1)
    df['IsHighValue'] = (df['CustomValueEstimate'] > df['CustomValueEstimate'].median()).astype(int)

    # 🔁 Map TermFrequency from string to numeric multiplier
    term_map = {
        'Monthly': 12,
        'Quarterly': 4,
        'Semi-Annual': 2,
        'Annual': 1
    }
    df['TermFrequency'] = df['TermFrequency'].map(term_map).fillna(1)

    df['MonthlyPremium'] = df['CalculatedPremiumPerTerm'] / df['TermFrequency'].replace(0, 1)

    return df
