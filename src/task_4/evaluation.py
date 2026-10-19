import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(model, X, y) -> dict:
    predictions = model.predict(X)
    return {
        'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
        'r2': float(r2_score(y, predictions)),
    }


def classification_metrics(model, X, y) -> dict:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        'accuracy': float(accuracy_score(y, predictions)),
        'precision': float(precision_score(y, predictions, zero_division=0)),
        'recall': float(recall_score(y, predictions, zero_division=0)),
        'f1': float(f1_score(y, predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, probabilities)),
        'average_precision': float(average_precision_score(y, probabilities)),
    }


def rmse_reduction_pct(reference_rmse: float, model_rmse: float) -> float:
    """Percent drop in RMSE relative to the reference. Negative means the model is worse."""
    return float((reference_rmse - model_rmse) / reference_rmse * 100.0)


def measure_batch_latency(model, X, batch_size: int, repeats: int) -> dict:
    """
    Time ``model.predict`` on one batch, repeated ``repeats`` times after a warm-up call.

    Timings depend on the machine, so they are reported apart from the
    deterministic metrics.
    """
    batch = X.iloc[:batch_size]
    model.predict(batch)
    timings_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(batch)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    median_ms = float(np.median(timings_ms))
    return {
        'batch_size': int(len(batch)),
        'repeats': int(repeats),
        'batch_ms_median': median_ms,
        'batch_ms_p95': float(np.percentile(timings_ms, 95)),
        'per_row_us_median': median_ms * 1000.0 / len(batch),
    }
