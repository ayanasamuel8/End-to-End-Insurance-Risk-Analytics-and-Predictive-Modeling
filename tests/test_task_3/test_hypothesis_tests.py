import pandas as pd
import numpy as np
from src.task_3.hypothesis_tests import province_risk_test, zip_margin_test, gender_risk_test

def test_province_risk_test():
    df = pd.DataFrame({
        'Province': ['A', 'A', 'B', 'B'],
        'claim_indicator': [1, 0, 1, 1],
        'TotalClaims': [100, 0, 200, 300]
    })
    result = province_risk_test(df, 'claim_frequency')
    assert 'anova_stat' in result
    assert 'p_value' in result

def test_zip_margin_test():
    df = pd.DataFrame({
        'PostalCode': ['X', 'X', 'Y', 'Y'],
        'TotalPremium': [500, 600, 300, 400],
        'TotalClaims': [200, 300, 150, 200]
    })
    result = zip_margin_test(df)
    assert 'p_value' in result
    assert 'results_by_zip' in result

def test_gender_risk_test():
    df = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'claim_indicator': [1, 0, 1, 1],
        'TotalClaims': [1000, 0, 800, 300]
    })
    result = gender_risk_test(df, 'claim_severity')
    assert 'p_value' in result
    assert 'results_by_gender' in result
