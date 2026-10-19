import os

import pandas as pd
import pytest

from src.task_4.data_processing import prepare_claim_severity_data
from src.task_4.model_training import train_and_compare_models

# Create a small dummy CSV for testing
dummy_csv_path = 'test_dummy_data.csv'

@pytest.fixture(scope="module", autouse=True)
def setup_dummy_data():
    # Create a dummy dataset for claim severity with minimal rows and columns
    data = {
        'UnderwrittenCoverID': [1, 2, 3, 4],
        'PolicyID': [101, 102, 103, 104],
        'TotalClaims': [100, 200, 150, 300],
        'TotalPremium': [1000, 2000, 1500, 3000],
        'RegistrationYear': [2020, 2019, 2021, 2018],
        'kilowatts': [100, 150, 120, 130],
        'Cylinders': [4, 6, 4, 4],
        'CustomValueEstimate': [5000, 7000, 6000, 8000],
        'CalculatedPremiumPerTerm': [100, 150, 120, 130],
        'TermFrequency': [12, 12, 12, 12],
        # Add categorical columns as strings (with small number of categories)
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'claim_indicator': [1, 0, 1, 0]  # Just to mimic full data format
    }
    df = pd.DataFrame(data)
    df.to_csv(dummy_csv_path, index=False)
    yield
    os.remove(dummy_csv_path)

def test_train_and_compare_models_runs():
    # Read the dummy data
    df = pd.read_csv(dummy_csv_path)
    # Assume 'TotalClaims' is the target
    # X = df.drop(columns=['TotalClaims'])
    # y = df['TotalClaims']

    # Simple train-test split (2 train, 2 test)
    X_train, X_test, y_train, y_test = prepare_claim_severity_data(df)

    best_model, best_name, results = train_and_compare_models(X_train, y_train, X_test, y_test)

    # Check return types
    assert best_name in results
    assert hasattr(best_model, 'predict')  # model should have predict method

    # Check results contain expected keys
    assert 'LinearRegression' in results
    assert 'RandomForest' in results
    assert 'XGBoost' in results

    # Check RMSE and R2 are floats
    for key, (rmse, r2) in results.items():
        assert isinstance(rmse, float)
        assert isinstance(r2, float)


def test_fit_severity_models_selects_by_cross_validation(ci_params, synthetic_df):
    from src.task_4.data_processing import prepare_claim_severity_data
    from src.task_4.feature_engineering import add_features
    from src.task_4.model_training import fit_severity_models, select_best_model

    X_train, X_test, y_train, _ = prepare_claim_severity_data(add_features(synthetic_df))
    models, cv_rmse, best_params = fit_severity_models(X_train, y_train, ci_params['severity'], seed=42)

    assert set(models) == {'mean_baseline', 'linear_regression', 'random_forest', 'xgboost_default', 'xgboost_tuned'}
    assert set(cv_rmse) == set(models)
    assert set(best_params) == set(ci_params['severity']['search']['space'])
    assert select_best_model(cv_rmse) == min(
        (name for name in cv_rmse if name != 'mean_baseline'), key=cv_rmse.get
    )
    assert all(model.predict(X_test).shape == (len(X_test),) for model in models.values())


def test_select_best_model_breaks_ties_by_name():
    from src.task_4.model_training import select_best_model

    assert select_best_model({'b': 1.0, 'a': 1.0, 'c': 2.0}) == 'a'


def test_select_best_model_never_picks_the_mean_baseline():
    from src.task_4.model_training import select_best_model

    assert select_best_model({'mean_baseline': 1.0, 'xgboost_tuned': 2.0}) == 'xgboost_tuned'


def test_fit_claim_probability_models(ci_params, synthetic_df):
    from src.task_4.data_processing import prepare_claim_probability_data
    from src.task_4.feature_engineering import add_features
    from src.task_4.model_training import fit_claim_probability_models

    X_train, X_test, y_train, _ = prepare_claim_probability_data(add_features(synthetic_df))
    models = fit_claim_probability_models(X_train, y_train, ci_params['claim_probability'], seed=42)

    assert set(models) == {'logistic_regression', 'xgboost'}
    for model in models.values():
        probabilities = model.predict_proba(X_test)[:, 1]
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
