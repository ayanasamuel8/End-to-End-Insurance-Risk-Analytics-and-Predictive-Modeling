import pandas as pd
from src.task_3.segmentation_utils import calculate_claim_frequency, calculate_claim_severity, calculate_margin, get_groups_by_feature

def test_claim_frequency():
    df = pd.DataFrame({'claim_indicator': [1, 0, 1, 1]})
    assert calculate_claim_frequency(df) == 0.75

def test_claim_severity():
    df = pd.DataFrame({'claim_indicator': [1, 0, 1], 'TotalClaims': [100, 0, 300]})
    assert calculate_claim_severity(df) == 200

def test_calculate_margin():
    df = pd.DataFrame({'TotalPremium': [1000, 800], 'TotalClaims': [400, 200]})
    assert calculate_margin(df) == 1200

def test_get_groups_by_feature():
    df = pd.DataFrame({'Gender': ['M', 'F', 'F', 'M']})
    a, b = get_groups_by_feature(df, 'Gender', 'M', 'F')
    assert set(a['Gender']) == {'M'}
    assert set(b['Gender']) == {'F'}
