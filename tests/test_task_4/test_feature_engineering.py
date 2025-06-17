import pandas as pd
from src.task_4.feature_engineering import add_features

def test_add_features():
    df = pd.DataFrame({
        'TotalClaims': [1000, 2000, 0, 5000],
        'TotalPremium': [5000, 4000, 0, 1000],
        'RegistrationYear': [2024, 2023, 2025, 2010],
        'kilowatts': [100, 200, 150, 300],
        'Cylinders': [4, 0, 3, 6],
        'CustomValueEstimate': [10000, 20000, 15000, 30000],
        'CalculatedPremiumPerTerm': [1200, 800, 0, 400],
        'TermFrequency': [12, 4, 0, 1]
    })
    
    df_new = add_features(df)
    
    expected_claim_ratio = df['TotalClaims'] / df['TotalPremium'].replace(0, 1)
    expected_vehicle_age = 2025 - df['RegistrationYear']
    expected_is_new = (expected_vehicle_age <= 1).astype(int)
    expected_power_per_cylinder = df['kilowatts'] / df['Cylinders'].replace(0, 1)
    median_value = df['CustomValueEstimate'].median()
    expected_is_high_value = (df['CustomValueEstimate'] > median_value).astype(int)
    expected_monthly_premium = df['CalculatedPremiumPerTerm'] / df['TermFrequency'].replace(0, 1)

    assert all(df_new['ClaimRatio'] == expected_claim_ratio)
    assert all(df_new['VehicleAge'] == expected_vehicle_age)
    assert all(df_new['IsNew'] == expected_is_new)
    assert all(df_new['PowerPerCylinder'] == expected_power_per_cylinder)
    assert all(df_new['IsHighValue'] == expected_is_high_value)
    assert all(df_new['MonthlyPremium'] == expected_monthly_premium)
