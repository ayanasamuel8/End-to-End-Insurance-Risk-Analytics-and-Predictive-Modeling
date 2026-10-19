import pandas as pd

from src.task_4.feature_engineering import add_features


def make_frame(term_frequency):
    return pd.DataFrame({
        'TotalClaims': [1000, 2000, 0, 5000],
        'TotalPremium': [5000, 4000, 0, 1000],
        'RegistrationYear': [2024, 2023, 2025, 2010],
        'kilowatts': [100, 200, 150, 300],
        'Cylinders': [4, 0, 3, 6],
        'CustomValueEstimate': [10000, 20000, 15000, 30000],
        'CalculatedPremiumPerTerm': [1200, 800, 0, 400],
        'TermFrequency': term_frequency,
    })


def test_add_features_values():
    result = add_features(make_frame(['Monthly', 'quarterly', ' Semi-Annual ', 'ANNUAL']))

    assert result['ClaimRatio'].tolist() == [0.2, 0.5, 0.0, 5.0]
    assert result['VehicleAge'].tolist() == [1, 2, 0, 15]
    assert result['IsNew'].tolist() == [1, 0, 1, 0]
    assert result['PowerPerCylinder'].tolist() == [25.0, 200.0, 50.0, 50.0]
    assert result['IsHighValue'].tolist() == [0, 1, 0, 1]
    assert result['TermFrequency'].tolist() == [12, 4, 2, 1]
    assert result['MonthlyPremium'].tolist() == [100.0, 200.0, 0.0, 400.0]


def test_add_features_does_not_change_input():
    frame = make_frame(['monthly'] * 4)
    before = frame.copy()
    add_features(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_unknown_term_frequency_counts_as_one():
    result = add_features(make_frame(['weekly', None, 12, 'annual']))
    assert result['TermFrequency'].tolist() == [1, 1, 1, 1]


def test_reference_year_is_configurable():
    result = add_features(make_frame(['monthly'] * 4), reference_year=2030)
    assert result['VehicleAge'].tolist() == [6, 7, 5, 20]
