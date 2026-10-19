# 🚗 Insurance Risk Modeling & Dynamic Pricing System

This project develops an explainable, risk-aware pricing model for auto insurance policies. It combines exploratory analysis, statistical testing, machine learning and a versioned data pipeline to predict claim severity and claim probability.

> ✅ Completed as part of the Week 3 Challenge at **10Academy**.

---

## 🧭 Project Goals

- Understand and explore insurance data to uncover actionable insights.
- Keep data, parameters and models reproducible with Git, DVC and pinned dependencies.
- Statistically validate hypotheses related to insurance risk.
- Build predictive models to estimate:
  - 💰 **Claim Severity**: how much a claim is likely to cost.
  - 📈 **Claim Probability**: how likely a policy is to produce a claim.
- Combine both into a risk-based premium:

  `Premium = Pr(claim) × Severity + Expenses + Margin`

---

## 🔧 Technologies & Tools

| Area            | Tools Used                                      |
|-----------------|-------------------------------------------------|
| Programming     | Python 3.13, Jupyter                            |
| Data handling   | Pandas, NumPy, PyArrow, DVC                     |
| Visualization   | Matplotlib, Seaborn                             |
| Modeling        | Scikit-learn, XGBoost                           |
| Interpretation  | SHAP, LIME                                      |
| Quality         | pytest, pytest-cov, Ruff                        |
| CI              | GitHub Actions                                  |
| Environment     | `venv` + pinned `requirements.txt`              |

---

## 📂 Repository Structure

```text
.
├── .dvc/                      # DVC configuration (local remote)
├── .github/workflows/ci.yml   # Lint, tests, pipeline run, metric gates
├── data/
│   └── raw/                   # MachineLearningRating_v3.txt, tracked by DVC (.dvc file)
├── notebooks/                 # EDA, hypothesis testing and modeling notebooks
├── reports/                   # Metrics and SHAP tables written by the pipeline
├── src/
│   ├── pipeline/              # DVC stages: prepare, train, evaluate, check_metrics
│   ├── task_1/                # EDA helpers
│   ├── task_3/                # Hypothesis testing modules
│   ├── task_4/                # Features, encoding, models, evaluation, SHAP and LIME
│   ├── data_loader.py
│   └── preprocessing.py       # Cleaning of the raw data
├── tests/                     # Unit tests and pipeline tests
├── dvc.yaml                   # Pipeline definition
├── params.yaml                # Every tunable value used by the pipeline
├── params.ci.yaml             # Small configuration for CI runs on generated data
├── pyproject.toml             # pytest, coverage and Ruff settings
├── requirements.txt           # Pinned dependencies
└── requirements-notebooks.txt # Extra packages for the notebooks
```

`models/` is created by the pipeline and is tracked by DVC, not by Git.

---

## 📦 Setup

```bash
git clone https://github.com/ayanasamuel8/End-to-End-Insurance-Risk-Analytics-and-Predictive-Modeling.git
cd End-to-End-Insurance-Risk-Analytics-and-Predictive-Modeling
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To run the notebooks, also install `requirements-notebooks.txt`.

---

## 🔁 Reproducing the Results

The pipeline starts from the raw file `data/raw/MachineLearningRating_v3.txt` (about 1,000,000 rows, 52 columns, pipe separated), which DVC tracks by checksum. The 10 Academy challenge supplies it as `MachineLearningRating_v3.zip`.

```bash
dvc pull      # fetch the raw data from the configured remote
dvc repro     # run only the stages whose inputs, code or parameters changed
dvc metrics show
```

If the remote has no copy, unzip the challenge file into `data/raw/` and run `dvc repro`. DVC checks the file against the checksum in `data/raw/MachineLearningRating_v3.txt.dvc`.

The remote in `.dvc/config` is a folder on the author's machine (`~/dvc_storage`). To use another location, run `dvc remote modify localstorage url <path-or-url>`.

### Pipeline stages

| Stage      | Command                           | Reads                             | Writes                                                      |
|------------|-----------------------------------|-----------------------------------|-------------------------------------------------------------|
| `clean`    | `python -m src.pipeline.clean`    | raw data                          | `data/cleaned/cleaned_data.csv`                             |
| `prepare`  | `python -m src.pipeline.prepare`  | cleaned data, `params.yaml`       | `data/processed/` (train and test Parquet files)            |
| `train`    | `python -m src.pipeline.train`    | `data/processed/`                 | `models/`, `reports/model_selection.json`                   |
| `evaluate` | `python -m src.pipeline.evaluate` | `data/processed/`, `models/`      | `reports/metrics.json`, `reports/latency.json`, SHAP tables |

`dvc.lock` is written by the first `dvc repro` and records the checksum of every input and output. Commit it so that `dvc repro` can tell what changed.

### What keeps runs identical

- Dependency versions are pinned in `requirements.txt` and the Python version is in `.python-version`.
- Every random choice (data split, cross-validation folds, hyperparameter search, model seeds) uses `seed` from `params.yaml`.
- Every value that affects a result is declared in `params.yaml`, so DVC reruns exactly the stages that depend on a changed value.
- Cleaning uses no random numbers. Claim amounts are never altered: a row has a claim only if the raw file says so.
- Categorical codes depend on the sorted category values, not on row order.
- A test runs the three stages twice and requires byte-identical processed data, models, metrics and SHAP tables.

`reports/latency.json` is the only output that varies between runs, because it is a timing measurement.

### Models

| Task                | Candidates                                                                       | Selection                                       |
|---------------------|----------------------------------------------------------------------------------|-------------------------------------------------|
| Claim severity      | Mean baseline, linear regression, random forest, default XGBoost, tuned XGBoost  | Lowest cross-validated RMSE on the training set |
| Claim probability   | Logistic regression, XGBoost                                                     | XGBoost                                         |

The tuned XGBoost model comes from a randomized search over the space in `params.yaml` (`severity.search`), which includes the squared-error, gamma and Tweedie objectives. The mean baseline is reported for comparison and is never selected.

Rows are grouped by `PolicyID`. All rows of a policy fall on the same side of the train/test split and inside the same cross-validation fold, so a model is never scored on a policy it has seen.

### Reading the results

`reports/metrics.json` holds, for the held-out test rows:

- RMSE and R² for every severity model, plus the cross-validated RMSE used for selection.
- `xgboost_tuned_rmse_reduction_pct`: the percentage drop in RMSE of the tuned model against each other severity model.
- Accuracy, precision, recall, F1, ROC AUC and average precision for both claim probability models. Only about 0.28% of rows have a claim, so accuracy is near 100% for any model. Read ROC AUC and average precision instead (a model with no skill scores an average precision equal to the claim rate).

`reports/latency.json` holds the median and 95th percentile time to score one batch (`evaluate.latency.batch_size` rows) and the median time per row. `reports/shap_severity.csv` and `reports/shap_claim_probability.csv` list the mean absolute SHAP value of each feature.

Run `dvc metrics show` for a table, or `dvc metrics diff` to compare against an earlier commit.

### Regression gates

`python -m src.pipeline.check_metrics` exits with an error when a limit under `gates` in `params.yaml` is broken:

| Gate                                        | Meaning                                                      |
|---------------------------------------------|--------------------------------------------------------------|
| `min_rmse_reduction_vs_xgboost_default_pct` | The tuned model must not be worse than default XGBoost       |
| `min_claim_probability_roc_auc`             | The classifier must rank claims better than chance           |
| `max_batch_latency_ms`                      | Median batch scoring time must stay within budget            |

---

## ✅ Quality Checks and CI

The workflow in `.github/workflows/ci.yml` runs on every push and pull request to `main`:

1. `ruff check .` for linting.
2. `pytest`, which fails when coverage of the measured modules drops below 95%.
3. `dvc dag` to confirm `dvc.yaml` is valid.
4. The three pipeline stages on generated data with `params.ci.yaml`, then the metric gates.
5. Upload of the generated reports as a build artifact.

Coverage covers `src/`, except the notebook plotting helpers (`src/task_1/viz/`, `src/task_1/eda/univariate.py`, `src/task_1/eda/bivariate.py`) and `src/config.py`. See `pyproject.toml`.

Run the same checks locally:

```bash
ruff check .
pytest
python -m src.pipeline.synthetic --out data/synthetic/raw_data.txt
python -m src.pipeline.clean --params params.ci.yaml
python -m src.pipeline.prepare --params params.ci.yaml
python -m src.pipeline.train --params params.ci.yaml
python -m src.pipeline.evaluate --params params.ci.yaml
python -m src.pipeline.check_metrics --params params.ci.yaml
```

The generated data in `src/pipeline/synthetic.py` only exercises the code. Its metrics say nothing about real insurance data.

---

## 🧮 Latest Results

From the committed run (`dvc.lock`, `reports/`), on the held-out policies. Rerun `dvc repro` to refresh them.

**Data:** 1,000,098 rows, 2,788 of them with a claim (1,408 policies).

**Claim severity** (568 test rows, so treat the RMSE as noisy):

| Model             | Cross-validated RMSE | Test RMSE |
|-------------------|----------------------|-----------|
| Mean baseline     | 38,761               | 38,135    |
| Linear regression | 32,834               | 34,743    |
| Random forest     | 32,691               | 32,920    |
| XGBoost, default  | 35,239               | 36,521    |
| XGBoost, tuned    | 32,515               | 33,239    |

Tuned XGBoost cuts test RMSE by 9.0% against default XGBoost, 4.3% against linear regression and 12.8% against the mean baseline. Random forest is 1.0% better than tuned XGBoost on the test rows, though tuned XGBoost has the lower cross-validated RMSE and is the selected model.

**Claim probability** (200,019 test rows): XGBoost reaches ROC AUC 0.934 and average precision 0.027, against 0.872 and 0.021 for logistic regression. At the default 0.5 cut-off neither model predicts a claim, so precision, recall and F1 are 0. Use the predicted probability, not the label.

**Batch scoring time**, median over 30 runs on the author's machine: 5.1 ms for 568 severity rows and 9.7 ms for 1,000 claim probability rows.

---

## 📊 Task Breakdown

### 🔍 Task 1: EDA & Git Setup

- Configured Git and GitHub, created the `task-1` branch.
- Performed EDA on claims, premiums and customer demographics.
- Visualized insights across provinces, genders and vehicle types.
- Identified key drivers of loss ratio and risk.

### 💾 Task 2: Data Version Control (DVC)

- Initialized DVC and tracked the data files.
- Set up a local remote and pushed the data.
- Added the pipeline in `dvc.yaml` with parameters in `params.yaml`.

### 📊 Task 3: Hypothesis Testing

- Tested whether risk varies across **provinces** and **zip codes**.
- Compared **claim frequency** and **severity** between genders.
- Tested whether profit margins vary by region.
- Used t-tests, ANOVA and chi-squared tests, with a business interpretation of each result.

### 🧠 Task 4: Predictive Modeling

- Severity regression with linear regression, random forest and XGBoost (default and tuned).
- Claim probability classification with logistic regression and XGBoost.
- Interpretation with **SHAP** (pipeline) and **LIME** (notebook).
- Features and encoding in `src/task_4/`, stages in `src/pipeline/`.

#### Data handling rules

- The raw claim amounts are used as they are. Only 2,788 rows have a claim, and the severity models train on those rows alone.
- The claim probability model never sees `TotalClaims` or `ClaimRatio`, because both are computed from the outcome being predicted.
- Identifier columns (`UnderwrittenCoverID`, `PolicyID`) are dropped from both models.
- Date text columns become year and month features.
- `TermFrequency` text is matched without regard to case, so values such as `monthly` are converted to their multiplier.

---

## 📈 Key Insights & Recommendations

| Insight                                | Impact on Pricing Strategy                 |
|----------------------------------------|--------------------------------------------|
| ≤4 cylinder vehicles → ↑ severity risk | Apply loading to small-engine vehicles     |
| Non-VAT registered → ↑ risk            | Raise base rate for unregistered customers |
| Converted/modified vehicles → ↑ risk   | Apply higher risk surcharge                |
| Alarm/immobiliser → ↓ risk             | Provide discount for security features     |
| New vehicles → ↓ risk                  | Discount for newer vehicles                |

These come from the linear-model coefficients and LIME explanations in `notebooks/task_4/`. Re-check them against `reports/shap_severity.csv` after each `dvc repro`.

---

## 📓 Notebooks

The notebooks in `notebooks/` are exploratory. Notebook `05_hypothesis_testing.ipynb` no longer replaces zero claims with the median claim amount, but its saved outputs were produced with that replacement and are out of date. `notebooks/task_4/` uses `train_and_compare_models`, a quick comparison of three untuned models, and does not use the pipeline. Its saved outputs are an earlier run and do not match `reports/metrics.json`. See `notebooks/README.md`.

---

## 👥 Contributors

👤 **Ayana Samuel**: full data science workflow (EDA, DVC, statistical testing, machine learning, CI).
GitHub: <https://github.com/ayanasamuel8/End-to-End-Insurance-Risk-Analytics-and-Predictive-Modeling>

## 📜 License

This project is licensed for academic and demonstration use. Contact the author for commercial usage rights.
