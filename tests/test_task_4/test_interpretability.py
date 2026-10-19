import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.task_4.interpretability import explain_model_with_lime, shap_importance, show_lime_explanation


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({'signal': rng.normal(size=200), 'noise': rng.normal(size=200)})
    y = 5 * X['signal'] + rng.normal(scale=0.1, size=200)
    return X, y


def test_shap_importance_ranks_the_informative_feature_first_for_trees(regression_data):
    X, y = regression_data
    model = XGBRegressor(n_estimators=30, random_state=0).fit(X, y)
    importance = shap_importance(model, X, sample_size=100, seed=0)
    assert list(importance.columns) == ['feature', 'mean_abs_shap']
    assert importance.loc[0, 'feature'] == 'signal'


def test_shap_importance_supports_linear_models(regression_data):
    X, y = regression_data
    importance = shap_importance(LinearRegression().fit(X, y), X, sample_size=50, seed=0)
    assert importance.loc[0, 'feature'] == 'signal'


def test_shap_importance_supports_scaled_logistic_pipeline(regression_data):
    X, y = regression_data
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, (y > 0).astype(int))
    importance = shap_importance(model, X, sample_size=50, seed=0)
    assert importance.loc[0, 'feature'] == 'signal'


def test_shap_importance_is_repeatable(regression_data):
    X, y = regression_data
    model = XGBRegressor(n_estimators=20, random_state=0).fit(X, y)
    pd.testing.assert_frame_equal(shap_importance(model, X, 80, 1), shap_importance(model, X, 80, 1))


@pytest.mark.parametrize('as_frame', [True, False])
def test_lime_explanation_lists_ten_or_fewer_features(regression_data, capsys, as_frame):
    X, y = regression_data
    model = LinearRegression().fit(X.to_numpy(), y)
    train, test = (X, X) if as_frame else (X.to_numpy(), X.to_numpy())
    explanation = explain_model_with_lime(model, train, test, list(X.columns), instance_idx=0)
    show_lime_explanation(explanation)
    printed = capsys.readouterr().out
    assert 'Feature contributions to prediction:' in printed
    assert len(explanation.as_list()) <= 10
