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
| `README.md` | This file, documentation of the codebase structure. |

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
| `feature_engineering.py` | `add_features`: vehicle age, power per cylinder, monthly premium and related features. Returns a copy of its input. |
| `data_processing.py` | Date expansion, integer encoding of categories, removal of identifier and claim-derived columns, and the train/test splits for both tasks. |
| `model_training.py` | Severity candidates, XGBoost randomized search, cross-validated model selection, and the claim probability models. Also holds `train_and_compare_models`, the quick three-model comparison used by the notebook. |
| `evaluation.py` | RMSE, R², classification metrics, RMSE reduction and batch latency measurement. |
| `interpretability.py` | `shap_importance` (mean absolute SHAP per feature) and LIME helpers. |

---

## 🔁 `pipeline/` — DVC Stages

| File | Description |
|------|-------------|
| `params.py` | Reads `params.yaml`, seeds random generators, reads and writes JSON. |
| `prepare.py` | Stage `prepare`: cleaned CSV to train and test Parquet files. |
| `train.py` | Stage `train`: fits and selects the models, writes `models/` and `reports/model_selection.json`. |
| `evaluate.py` | Stage `evaluate`: test metrics, latency and SHAP tables. |
| `check_metrics.py` | Compares the reports with the `gates` in `params.yaml` and exits with an error on a violation. |
| `synthetic.py` | Generates deterministic stand-in data with the same columns, for CI and tests. |

Each stage takes `--params <file>` (default `params.yaml`).

---

## ✅ Usage Example

```python
import pandas as pd

from src.task_4.data_processing import prepare_claim_severity_data
from src.task_4.feature_engineering import add_features
from src.task_4.interpretability import shap_importance
from src.task_4.model_training import fit_severity_models, select_best_model
from src.pipeline.params import load_params

params = load_params("params.yaml")
df = add_features(pd.read_csv("data/cleaned/cleaned_data.csv"))
X_train, X_test, y_train, y_test = prepare_claim_severity_data(df, random_state=params["seed"])

models, cv_rmse, _ = fit_severity_models(X_train, y_train, params["severity"], params["seed"])
best = select_best_model(cv_rmse)
print(shap_importance(models[best], X_test).head())
```

To run the full pipeline instead, use `dvc repro`.

---

## 🧩 Design Philosophy

- **Modular**: code is separated into small, reusable units.
- **Interpretable**: business-facing logic (such as hypothesis results) is separate from statistical code.
- **Parameter driven**: pipeline code takes its values from `params.yaml` and holds no hidden constants.

## 📝 Note

Run commands from the repository root so that `from src... import ...` resolves. In notebooks, add the repository root to `sys.path`.
