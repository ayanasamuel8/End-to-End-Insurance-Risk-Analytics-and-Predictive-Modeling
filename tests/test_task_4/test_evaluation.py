import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.task_4.evaluation import (
    classification_metrics,
    measure_batch_latency,
    regression_metrics,
    rmse_reduction_pct,
)


def test_regression_metrics_for_constant_prediction():
    X = pd.DataFrame({'a': [0, 1, 2, 3]})
    y = pd.Series([2.0, 2.0, 4.0, 4.0])
    metrics = regression_metrics(DummyRegressor(strategy='constant', constant=3.0).fit(X, y), X, y)
    assert metrics['rmse'] == 1.0
    assert metrics['r2'] == 0.0


def test_classification_metrics_for_perfect_scores():
    X = pd.DataFrame({'a': [0, 1, 2, 3]})
    y = pd.Series([0, 0, 1, 1])

    class Perfect:
        def predict(self, data):
            return (data['a'] >= 2).astype(int).to_numpy()

        def predict_proba(self, data):
            positive = (data['a'] >= 2).astype(float).to_numpy()
            return np.column_stack([1 - positive, positive])

    metrics = classification_metrics(Perfect(), X, y)
    assert metrics == {
        'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0, 'average_precision': 1.0
    }


def test_classification_metrics_with_no_positive_predictions():
    X = pd.DataFrame({'a': [0, 1, 2, 3]})
    y = pd.Series([0, 1, 0, 1])
    model = DummyClassifier(strategy='constant', constant=0).fit(X, y)
    metrics = classification_metrics(model, X, y)
    assert metrics['precision'] == 0.0
    assert metrics['recall'] == 0.0
    assert metrics['roc_auc'] == 0.5
    assert metrics['average_precision'] == 0.5


def test_rmse_reduction_pct():
    assert rmse_reduction_pct(100.0, 72.0) == pytest.approx(28.0)
    assert rmse_reduction_pct(100.0, 110.0) == pytest.approx(-10.0)


def test_measure_batch_latency_reports_batch_and_row_timing():
    X = pd.DataFrame({'a': range(50)})
    model = DummyRegressor().fit(X, np.zeros(50))
    timing = measure_batch_latency(model, X, batch_size=20, repeats=5)
    assert timing['batch_size'] == 20
    assert timing['repeats'] == 5
    assert 0 <= timing['batch_ms_median'] <= timing['batch_ms_p95'] + 1e-9
    assert np.isclose(timing['per_row_us_median'], timing['batch_ms_median'] * 1000 / 20)
