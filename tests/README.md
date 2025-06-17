# 🧪 `tests/` — Unit Testing Suite

This directory contains all unit tests for verifying the correctness, robustness, and reliability of the codebase modules in the `src/` directory. The tests are organized by task to maintain modularity and clarity.

> ✅ To run all tests:
```bash
pytest 
```
```bash
tests/
│
├── __init__.py                    # Marks the test suite as a Python package
├── README.md                      # Test documentation (this file)
├── test_data_loader.py           # Tests for general data loading functionality
├── test_example.py               # Sample test (placeholder or CI verification)
│
├── test_task_3/                  # Tests for Task 3 - Hypothesis Testing & Segmentation
│   ├── __init__.py
│   ├── test_business_analysis.py
│   ├── test_data_segmentation.py
│   ├── test_hypothesis_tests.py
│   ├── test_segmentation_utils.py
│   ├── test_stats_helpers.py
│   └── __pycache__/              # Compiled Python files (auto-generated)
│
└── test_task_4/                  # Tests for Task 4 - Modeling & Interpretability
    ├── __init__.py
    ├── test_feature_engineering.py
    ├── test_model_training.py
    └── __pycache__/              # Compiled Python files (auto-generated)
```
## 🧪 Test Descriptions

### 📁 Top-Level Tests

| File                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `test_data_loader.py`  | Validates data loading functions: existence, structure, and content of dataframes. |
| `test_example.py`      | Generic test file — often used to test CI pipelines or placeholder logic.   |

---

### 🔍 `test_task_3/` — Risk Segmentation & Hypothesis Testing

| File                        | Tests                                                                       |
|-----------------------------|-----------------------------------------------------------------------------|
| `test_business_analysis.py` | Business logic functions that interpret test results (e.g., actionable pricing decisions). |
| `test_data_segmentation.py` | Region/segment-based splits and checks for distribution consistency.        |
| `test_hypothesis_tests.py`  | Statistical hypothesis testing: validates p-values, group means, assumptions. |
| `test_segmentation_utils.py`| Utility functions for grouping and label mapping.                           |
| `test_stats_helpers.py`     | Helper statistical computations like t-scores, z-values, variance checks.   |

---

### 🤖 `test_task_4/` — ML Models & Interpretability

| File                         | Tests                                                                       |
|------------------------------|-----------------------------------------------------------------------------|
| `test_feature_engineering.py`| Tests new feature creation (vehicle age, conversion flags, VAT status).     |
| `test_model_training.py`     | Ensures models train without errors, output expected performance metrics, and save artifacts. |

## 🧰 Running Tests

### 🔃 Run All Tests
```bash
pytest
```
🧪 Run a Specific File
```bash
pytest tests/test_task_4/test_model_training.py
```
✅ Best Practices
Follow naming conventions: test_<module_name>.py and test_<function>().

Use fixtures for setup/teardown.

Mock external dependencies or file access where possible.

Keep test coverage above 80% for critical modules (Task 4, Task 3).

🔄 CI Integration
These tests are automatically executed via GitHub Actions to ensure all pushes to main maintain code quality and prevent regressions.