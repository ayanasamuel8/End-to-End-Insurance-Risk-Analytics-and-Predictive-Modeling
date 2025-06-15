import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data_loader import extract_numeric_cols, extract_categorical_cols, extract_date_time_cols

sns.set(style='whitegrid', palette='muted')

def _save_fig(save_fig: bool, fig_path: str, filename: str):
    if save_fig:
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

# ---- Numeric Features ----
def analyze_numeric_features(df: pd.DataFrame, save_fig: bool = False, fig_path: str = "outputs/univariate/numeric/"):
    numeric_cols = extract_numeric_cols(df)
    results = {}

    for col in numeric_cols:
        stats = df[col].describe()
        missing = df[col].isna().sum()
        results[col] = {"stats": stats, "missing": missing}

        # Histogram
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        _save_fig(save_fig, fig_path, f"hist_{col}.png")
        plt.show()

        # Boxplot
        plt.figure(figsize=(6, 2))
        sns.boxplot(x=df[col].dropna(), color='lightcoral')
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        _save_fig(save_fig, fig_path, f"box_{col}.png")
        plt.show()

    return results


# ---- Categorical Features ----
def analyze_categorical_features(df: pd.DataFrame, top_n: int = 10, save_fig: bool = False, fig_path: str = "outputs/univariate/categorical/"):
    cat_cols = extract_categorical_cols(df)
    results = {}

    for col in cat_cols:
        counts = df[col].value_counts(dropna=False)
        results[col] = {"counts": counts}

        plt.figure(figsize=(8, 4))
        top_values = counts[:top_n]
        sns.barplot(x=top_values.values, y=top_values.index, palette="viridis")
        plt.title(f'Top {top_n} Categories in {col}')
        plt.xlabel("Count")
        plt.ylabel(col)
        _save_fig(save_fig, fig_path, f"bar_{col}.png")
        plt.show()

    return results


# ---- Datetime Features ----
def analyze_datetime_features(df: pd.DataFrame, save_fig: bool = False, fig_path: str = "outputs/univariate/datetime/"):
    datetime_cols = extract_date_time_cols(df)
    results = {}

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Ensure datetime conversion
        df_temp = df.copy()
        df_temp["year"] = df[col].dt.year
        df_temp["month"] = df[col].dt.month
        df_temp["weekday"] = df[col].dt.dayofweek
        df_temp["day"] = df[col].dt.day

        results[col] = {
            "min_date": df[col].min(),
            "max_date": df[col].max(),
            "n_missing": df[col].isna().sum()
        }

        # Plot by month
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col].dropna(), bins=30, kde=False, color="slateblue")
        plt.title(f'Date Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel("Count")
        _save_fig(save_fig, fig_path, f"dist_{col}.png")
        plt.show()

    return results
