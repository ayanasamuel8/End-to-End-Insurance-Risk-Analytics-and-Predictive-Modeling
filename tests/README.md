# 🧪 `tests/` — Unit Testing Suite

This directory holds the tests for the modules in `src/`. They run in CI on every push and pull request to `main`.

```bash
pytest                                     # runs everything and reports coverage
pytest tests/test_task_4/test_model_training.py
```

`pytest` is configured in `pyproject.toml`. It prints a coverage table and fails when total coverage of the measured modules is below 95%. Plotting helpers for the notebooks (`src/task_1/viz/`, `src/task_1/eda/univariate.py`, `src/task_1/eda/bivariate.py`) and `src/config.py` are not measured.

```text
tests/
├── conftest.py                    # Shared fixtures: generated data and CI parameters
├── test_data_loader.py            # Loading, saving and column-type helpers
├── test_preprocessing.py          # Cleaning of the raw data
├── test_task_1_eda.py             # Outlier detection and summary statistics
├── test_pipeline/
│   └── test_pipeline.py           # Pipeline stages, reproducibility, gates, dvc.yaml and params
├── test_task_3/
│   ├── test_business_analysis.py
│   ├── test_data_segmentation.py
│   ├── test_hypothesis_tests.py
│   ├── test_segmentation_utils.py
│   └── test_stats_helpers.py
└── test_task_4/
    ├── test_data_processing.py
    ├── test_evaluation.py
    ├── test_feature_engineering.py
    ├── test_interpretability.py
    └── test_model_training.py
```

## What the tests check

| File | Checks |
|------|--------|
| `test_pipeline/test_pipeline.py` | Runs `clean`, `prepare`, `train` and `evaluate` on generated data. Confirms that claim amounts survive cleaning and that no policy sits on both sides of the split. Confirms that a second run produces byte-identical processed data, models, metrics and SHAP tables. Confirms that each metric gate fires. Confirms that `dvc.yaml` points at real modules and `params.yaml` keys. |
| `test_task_4/test_data_processing.py` | Date expansion, encoding that ignores row order, no claim-derived columns in the claim probability data, reproducible splits. |
| `test_task_4/test_feature_engineering.py` | Exact feature values, case-insensitive `TermFrequency`, input left unchanged. |
| `test_task_4/test_model_training.py` | Candidate models, cross-validated selection, claim probability models. |
| `test_task_4/test_evaluation.py` | Metric values, RMSE reduction, latency reporting. |
| `test_task_4/test_interpretability.py` | SHAP ranking for tree, linear and pipeline models. LIME output. |
| `test_task_3/` | Hypothesis tests, segmentation helpers, statistical helpers and business interpretation. |
| `test_preprocessing.py` | Duplicate removal, imputation, outlier capping, comma-number conversion. |

## Conventions

- Name files `test_<module>.py` and functions `test_<behavior>()`.
- Use fixtures from `conftest.py` instead of reading real data.
- Assert exact values where the result is known.
