import numpy as np
import pandas as pd
from src.task_3.stats_helpers import anova_test, t_test_independent, chi_square_test
def province_risk_test(data, risk_metric='claim_frequency'):
    """
    Compare risk across provinces using the specified risk metric.
    
    For claim frequency: Calculate the proportion of policies with at least one claim.
    For claim severity: Consider only policies with at least one claim and compute their average.
    Use ANOVA to compare differences across provinces.
    
    Returns a dictionary with test statistics and p-values.
    """
    results = {}
    provinces = data['Province'].unique()
    groups = []
    
    if risk_metric == 'claim_frequency':
        for province in provinces:
            subset = data[data['Province'] == province]
            # Binary flag: 1 if claim exists, 0 otherwise
            freq = subset['claim_indicator'].mean()
            groups.append(subset['claim_indicator'].values)
            results[province] = {'claim_frequency': freq}
    elif risk_metric == 'claim_severity':
        for province in provinces:
            subset = data[(data['Province'] == province) & (data['claim_indicator'] == 1)]
            severity = subset['TotalClaims'].mean() if not subset.empty else np.nan
            groups.append(subset['TotalClaims'].values if not subset.empty else np.array([0]))
            results[province] = {'claim_severity': severity}
    else:
        raise ValueError("Invalid risk metric specified.")
    
    stat, p_val = anova_test(groups)
    return {'results_by_province': results, 'anova_stat': stat, 'p_value': p_val}


def zip_risk_test(data, risk_metric='claim_frequency'):
    """
    Compares risk across zip codes using the specified risk metric (frequency or severity).
    Performs ANOVA across groups.
    """
    results = {}
    zip_codes = data['PostalCode'].dropna().unique()
    groups = []

    if risk_metric == 'claim_frequency':
        for z in zip_codes:
            subset = data[data['PostalCode'] == z]
            freq = subset['claim_indicator'].mean()
            groups.append(subset['claim_indicator'].values)
            results[z] = {'claim_frequency': freq}
    elif risk_metric == 'claim_severity':
        for z in zip_codes:
            subset = data[(data['PostalCode'] == z) & (data['claim_indicator'] == 1)]
            severity = subset['TotalClaims'].mean() if not subset.empty else np.nan
            groups.append(subset['TotalClaims'].values if not subset.empty else np.array([0]))
            results[z] = {'claim_severity': severity}
    else:
        raise ValueError("Invalid risk metric.")

    stat, p_val = anova_test(groups)
    return {'results_by_zip': results, 'anova_stat': stat, 'p_value': p_val}

def zip_margin_test(data):
    """
    Tests whether margin (TotalPremium - TotalClaims) differs by zip code using ANOVA.
    """
    data = data.copy()
    data['margin'] = data['TotalPremium'] - data['TotalClaims']

    results = {}
    zip_codes = data['PostalCode'].dropna().unique()
    groups = []

    for z in zip_codes:
        subset = data[data['PostalCode'] == z]
        margin_mean = subset['margin'].mean()
        groups.append(subset['margin'].values)
        results[z] = {'average_margin': margin_mean}

    stat, p_val = anova_test(groups)
    return {'results_by_zip': results, 'anova_stat': stat, 'p_value': p_val}

def gender_risk_test(data, risk_metric='claim_severity'):
    """
    Compare risk between genders using a two-sample t-test.
    """
    results = {}

    male_data = data[data['Gender'] == 'Male']
    female_data = data[data['Gender'] == 'Female']

    if risk_metric == 'claim_frequency':
        male_vals = male_data['claim_indicator']
        female_vals = female_data['claim_indicator']
        results['Male'] = male_vals.mean()
        results['Female'] = female_vals.mean()
    elif risk_metric == 'claim_severity':
        male_vals = male_data[male_data['claim_indicator'] == 1]['TotalClaims']
        female_vals = female_data[female_data['claim_indicator'] == 1]['TotalClaims']
        results['Male'] = male_vals.mean()
        results['Female'] = female_vals.mean()
    else:
        raise ValueError("Invalid risk metric.")

    stat, p_val = t_test_independent(male_vals, female_vals)
    return {'results_by_gender': results, 't_stat': stat, 'p_value': p_val}
