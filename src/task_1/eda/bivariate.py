import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_matrix(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
    return corr


def scatter_plot(df, x, y):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"{y} vs {x}")
    plt.show()


def group_loss_ratio(df, by):
    return df.groupby(by).agg(
        TotalClaims=("TotalClaims", "sum"),
        TotalPremium=("TotalPremium", "sum")
    ).assign(LossRatio=lambda x: x.TotalClaims / x.TotalPremium.replace(0, pd.NA))


def line_plot(df, date_col, value_col):
    df_sorted = df.sort_values(by=date_col)
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df_sorted[date_col], y=df_sorted[value_col])
    plt.title(f"Trend of {value_col} over time")
    plt.xticks(rotation=45)
    plt.show()

