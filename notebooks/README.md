# 📒 Notebooks Overview — Insurance Pricing Project

This folder contains Jupyter Notebooks used to explore, analyze, and model insurance claim data as part of the **10Academy Week 3 Challenge**.

Each notebook is modular and corresponds to a specific task in the data science workflow — from EDA and hypothesis testing to modeling and interpretation.

---

## 📁 Folder Purpose

The `notebooks/` directory serves as the primary space for:

- Experimenting with data pipelines
- Visualizing insights
- Testing hypotheses
- Training and evaluating models
- Documenting key results

---

## 🧭 Execution Order

| Notebook Path                                               | Description                                                                          |
|-------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `task_1/01_Data_understanding.ipynb`                        | 🔍 **Initial Data Exploration** — Overview of dataset structure, basic distributions, and types of variables. |
| `task_1/02.eda_univariate.ipynb`                            | 📊 **Univariate Analysis** — Examines single-variable distributions and statistics, including missing data handling. |
| `task_1/03_eda_bivariate.ipynb`                             | 🔗 **Bivariate Analysis** — Explores relationships between key variables (e.g., claims vs. gender, province). |
| `task_1/04_visualization.ipynb`                             | 📈 **Visual Summary** — Aggregated plots and advanced visuals to communicate key trends and risk factors. |
| `task_3/05_hypthesis_testing.ipynb`                         | 📐 **Statistical Hypothesis Testing** — Validates assumptions across provinces, genders, and customer segments. |
| `task_4/06_model_training_and_interpretability.ipynb`       | 🧠 **Modeling & Interpretability** — Trains severity and claim probability models; interprets them using SHAP/LIME. |


> **Note:** Run these notebooks in order for best results. Dependencies between notebooks are minimal but intentional (e.g., modeling uses cleaned data from Notebook 5).

---

## 🔍 Highlights by Notebook

### 📁 `task_1/01_Data_understanding.ipynb`
- Overview of dataset structure and key variables
- Initial univariate statistics and class distributions
- Identified outliers and null-value patterns

### 📁 `task_1/02.eda_univariate.ipynb`
- Dealt with missing values and variable types
- Performed univariate analysis: claims, premiums, risk flags
- Created new features: vehicle age, risk class, etc.

### 📁 `task_1/03_eda_bivariate.ipynb`
- Bivariate relationships: claim vs gender, province, zip, cylinders
- Used cross-tabulations and grouped summaries
- Inferred possible risk drivers from visual trends

### 📁 `task_1/04_visualization.ipynb`
- Visual storytelling using bar plots, heatmaps, and boxplots
- Focused on loss ratio patterns by region and vehicle attributes
- Illustrated skewness, imbalance, and outliers effectively

### 📁 `task_3/05_hypthesis_testing.ipynb`
- Statistically tested hypotheses on claim risk factors
- Methods: z-test, t-test, chi-squared
- Provided business insights on regional and demographic effects

### 📁 `task_4/06_model_training_and_interpretability.ipynb`
- Modeled both claim severity (regression) and probability (classification)
- Trained Linear, RF, XGBoost models with performance metrics
- Used SHAP and LIME to interpret model decisions and explain pricing


---

## 📦 How to Use

To run the notebooks:

```bash
cd notebooks/
jupyter notebook
