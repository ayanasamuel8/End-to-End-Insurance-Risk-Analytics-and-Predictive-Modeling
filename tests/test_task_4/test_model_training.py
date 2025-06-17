import os
import pandas as pd
import pytest
from joblib import load
from src.task_4.model_training import train_and_compare_models

# Create a small dummy CSV for testing
dummy_csv_path = 'test_dummy_data.csv'

@pytest.fixture(scope="module", autouse=True)
def setup_dummy_data():
    # Create a dummy dataset for claim severity with minimal rows and columns
    data = {
        'TotalClaims': [100, 200, 150, 300],
        'TotalPremium': [1000, 2000, 1500, 3000],
        'RegistrationYear': [2020, 2019, 2021, 2018],
        'kilowatts': [100, 150, 120, 130],
        'Cylinders': [4, 6, 4, 4],
        'CustomValueEstimate': [5000, 7000, 6000, 8000],
        'CalculatedPremiumPerTerm': [100, 150, 120, 130],
        'TermFrequency': [12, 12, 12, 12],
        # Add categorical columns as strings (with small number of categories)
        'Gender': ['male', 'female', 'male', 'female'],
        'claim_indicator': [1, 0, 1, 0]  # Just to mimic full data format
    }
    df = pd.DataFrame(data)
    df.to_csv(dummy_csv_path, index=False)
    yield
    os.remove(dummy_csv_path)

def test_train_and_compare_models_runs():
    best_model, best_name, results = train_and_compare_models(dummy_csv_path)

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
