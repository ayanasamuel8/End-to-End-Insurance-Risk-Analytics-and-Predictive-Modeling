import numpy as np
import pandas as pd
import pytest

from src.task_3.hypothesis_tests import gender_risk_test, province_risk_test, zip_margin_test


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


def make_claims():
    return pd.DataFrame({
        'Province': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'],
        'PostalCode': ['1', '1', '2', '2', '3', '3', '4', None],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'claim_indicator': [1, 0, 1, 1, 0, 1, 1, 1],
        'TotalClaims': [100, 0, 150, 300, 0, 500, 120, 250],
        'TotalPremium': [500, 400, 450, 600, 300, 700, 350, 380],
    })


def test_province_risk_test_reports_each_province():
    from src.task_3.hypothesis_tests import province_risk_test

    frequency = province_risk_test(make_claims(), 'claim_frequency')
    assert frequency['results_by_province']['A']['claim_frequency'] == pytest.approx(2 / 3)

    severity = province_risk_test(make_claims(), 'claim_severity')
    assert severity['results_by_province']['B']['claim_severity'] == 400


def test_province_severity_without_claims_is_nan():
    from src.task_3.hypothesis_tests import province_risk_test

    data = make_claims()
    data.loc[data['Province'] == 'C', 'claim_indicator'] = 0
    severity = province_risk_test(data, 'claim_severity')
    assert np.isnan(severity['results_by_province']['C']['claim_severity'])


def test_zip_risk_test_for_both_metrics():
    from src.task_3.hypothesis_tests import zip_risk_test

    frequency = zip_risk_test(make_claims(), 'claim_frequency')
    assert frequency['results_by_zip']['1']['claim_frequency'] == 0.5
    severity = zip_risk_test(make_claims(), 'claim_severity')
    assert severity['results_by_zip']['3']['claim_severity'] == 500


def test_zip_severity_without_claims_is_nan():
    from src.task_3.hypothesis_tests import zip_risk_test

    data = make_claims()
    data.loc[data['PostalCode'] == '4', 'claim_indicator'] = 0
    assert np.isnan(zip_risk_test(data, 'claim_severity')['results_by_zip']['4']['claim_severity'])


def test_gender_risk_test_frequency():
    result = gender_risk_test(make_claims(), 'claim_frequency')
    assert result['results_by_gender']['Male'] == 0.75
    assert result['results_by_gender']['Female'] == 0.75


def test_invalid_risk_metric_raises():
    from src.task_3.hypothesis_tests import zip_risk_test

    with pytest.raises(ValueError):
        gender_risk_test(make_claims(), 'bogus')
    with pytest.raises(ValueError):
        zip_risk_test(make_claims(), 'bogus')
    with pytest.raises(ValueError):
        province_risk_test(make_claims(), 'bogus')
