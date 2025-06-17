# 📦 `src/` — Source Code Overview

This directory contains all core logic and modular components used in data processing, hypothesis testing, modeling, and interpretability for the **Insurance Risk Modeling** project.

---

## 📁 Top-Level Modules

| File | Description |
|------|-------------|
| `__init__.py` | Marks the directory as a Python package. |
| `config.py` | Central location for storing global configuration variables (paths, constants, toggles). |
| `data_loader.py` | Contains utilities for loading raw and processed datasets. |
| `preprocessing.py` | High-level preprocessing functions for cleaning and transforming data. |
| `README.md` | This file – documentation of the codebase structure. |

---

## 🔍 `task_1/` — Exploratory Data Analysis & Visualization

### 📁 `eda/` — EDA Logic
| File | Description |
|------|-------------|
| `bivariate.py` | Analyzes relationships between pairs of variables (e.g., claims vs. gender). |
| `univariate.py` | Summarizes distributions of individual features. |
| `summary_stats.py` | Provides descriptive statistics and summary tables. |
| `outlier_detection.py` | Detects and flags unusual or extreme values using statistical methods. |

### 📁 `viz/` — Visualization Utilities
| File | Description |
|------|-------------|
| `plot_utils.py` | Helper functions for generating consistent plots (bar charts, heatmaps, boxplots, etc.). |

---

## 📈 `task_3/` — Hypothesis Testing & Risk Segmentation

| File | Description |
|------|-------------|
| `hypothesis_tests.py` | Implements z-tests, t-tests, chi-squared tests for hypothesis validation. |
| `data_segmentation.py` | Splits the data by demographic and regional segments for focused analysis. |
| `business_analysis.py` | Converts statistical results into actionable business interpretations. |
| `segmentation_utils.py` | Utilities to group and label segmented datasets. |
| `stats_helpers.py` | Reusable statistical functions (p-value calculators, assumptions checks, etc.). |

---

## 🤖 `task_4/` — Modeling & Interpretability

| File | Description |
|------|-------------|
| `data_processing.py` | Handles encoding, normalization, splitting for modeling. |
| `feature_engineering.py` | Creates new predictive features such as vehicle age, conversion flags, etc. |
| `model_training.py` | Core training logic for regression and classification models (XGBoost, Random Forest, etc.). |
| `interpretability.py` | Generates SHAP & LIME explanations to interpret model decisions. |

---

## ✅ Usage Example

```python
from src.data_loader import load_clean_data
from src.task_1.eda.univariate import summarize_numerics
from src.task_4.model_training import train_xgboost_model
from src.task_4.interpretability import explain_with_shap

# Load and analyze data
df = load_clean_data("data/processed/cleaned_data.csv")
summarize_numerics(df)

# Train and explain model
model, X_test = train_xgboost_model(df)
explain_with_shap(model, X_test)
```
🧩 Design Philosophy
Modular: Code is separated into logically coherent, reusable units.

Interpretable: Business-facing logic (e.g., hypothesis results) is separated from statistical code.

Scalable: Easy to extend for future tasks like time-series modeling or real-time pricing engines.

📝 Note
Ensure all module imports use relative paths (from .module import ...) if running as a package, or adjust PYTHONPATH accordingly for standalone script runs.
