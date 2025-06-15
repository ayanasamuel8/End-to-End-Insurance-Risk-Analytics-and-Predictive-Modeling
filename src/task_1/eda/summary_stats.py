import pandas as pd
def describe_numerical(df: pd.DataFrame):
    return df.describe()


def calculate_loss_ratio(df: pd.DataFrame):
    df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, pd.NA)
    return df