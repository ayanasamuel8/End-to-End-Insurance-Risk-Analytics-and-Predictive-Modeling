# Insurance Risk Analytics Project

This project analyzes insurance data to extract actionable insights on risk, profitability, and optimal pricing strategies.

## 📂 Structure
- `data/`: Raw and processed data tracked via DVC
- `notebooks/`: EDA and modeling notebooks
- `src/`: Reusable Python scripts
- `tests/`: Unit tests for core logic
- `.github/workflows/`: CI/CD pipelines

## 📦 Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## ▶️ Run Tests
```bash
pytest
```
## 📊 Tools
- Python, DVC, GitHub Actions, SHAP, XGBoost