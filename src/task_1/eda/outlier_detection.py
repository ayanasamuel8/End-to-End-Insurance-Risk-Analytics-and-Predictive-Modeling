import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def detect_outliers_iqr(df: pd.DataFrame, column: str):
    """
    Returns rows where the column has outliers using IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers

def plot_outliers_box(df: pd.DataFrame, column: str, title: str = ""):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.title(title or f"Boxplot for {column}")
    plt.tight_layout()
    plt.show()
