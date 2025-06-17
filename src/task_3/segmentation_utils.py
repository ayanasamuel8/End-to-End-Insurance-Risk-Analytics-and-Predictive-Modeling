import numpy as np
import pandas as pd

def calculate_claim_frequency(data):
    """
    Calculate Claim Frequency as the proportion of policies with at least one claim.
    """
    # data['claim_indicator'] = data['TotalClaims'] > 0
    return data['claim_indicator'].mean()

def calculate_claim_severity(data):
    """
    Calculate Claim Severity as the average claim amount for policies with a claim.
    """
    data['claim_indicator'] = data['TotalClaims'] > 0
    claims = data[data['claim_indicator']]
    return claims['TotalClaims'].mean() if not claims.empty else np.nan

def calculate_margin(data):
    """
    Calculate margin as (TotalPremium - TotalClaims) for the data provided.
    """
    return (data['TotalPremium'] - data['TotalClaims']).sum()

def get_groups_by_feature(df, feature: str, value_a, value_b):
    group_a = df[df[feature] == value_a]
    group_b = df[df[feature] == value_b]
    return group_a, group_b
