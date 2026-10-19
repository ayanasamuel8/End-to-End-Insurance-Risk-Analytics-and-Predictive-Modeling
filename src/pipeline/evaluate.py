"""Stage 3: score the models on the held-out data and explain them."""
from pathlib import Path

import joblib

from src.pipeline.params import ensure_dir, load_params, parse_args, read_json, write_json
from src.pipeline.train import load_split
from src.task_4.data_processing import PROBABILITY_TARGET, SEVERITY_TARGET
from src.task_4.evaluation import (
    classification_metrics,
    measure_batch_latency,
    regression_metrics,
    rmse_reduction_pct,
)
from src.task_4.interpretability import shap_importance


def evaluate(params: dict) -> None:
    processed_dir = params['paths']['processed_dir']
    models_dir = Path(params['paths']['models_dir'])
    reports_dir = ensure_dir(params['paths']['reports_dir'])
    shap_rows = params['evaluate']['shap_sample_size']
    latency = params['evaluate']['latency']

    bundle = joblib.load(models_dir / 'severity.joblib')
    _, _, X_test, y_test = load_split(processed_dir, 'severity', SEVERITY_TARGET)
    X_test = X_test[bundle['features']]
    severity_scores = {name: regression_metrics(model, X_test, y_test) for name, model in bundle['models'].items()}
    tuned_rmse = severity_scores['xgboost_tuned']['rmse']
    selection = read_json(reports_dir / 'model_selection.json')['severity']
    best_model = bundle['models'][bundle['best']]

    classifier_bundle = joblib.load(models_dir / 'claim_probability.joblib')
    _, _, Xc_test, yc_test = load_split(processed_dir, 'claim_probability', PROBABILITY_TARGET)
    Xc_test = Xc_test[classifier_bundle['features']]
    classifier_scores = {
        name: classification_metrics(model, Xc_test, yc_test) for name, model in classifier_bundle['models'].items()
    }
    best_classifier = classifier_bundle['models'][classifier_bundle['best']]

    write_json(
        reports_dir / 'metrics.json',
        {
            'severity': {
                'selected_model': bundle['best'],
                'cv_rmse': selection['cv_rmse'],
                'test': severity_scores,
                'xgboost_tuned_rmse_reduction_pct': {
                    reference: rmse_reduction_pct(severity_scores[reference]['rmse'], tuned_rmse)
                    for reference in ('mean_baseline', 'linear_regression', 'random_forest', 'xgboost_default')
                },
                'test_rows': len(X_test),
            },
            'claim_probability': {
                'test': classifier_scores,
                'test_rows': len(Xc_test),
            },
        },
    )
    write_json(
        reports_dir / 'latency.json',
        {
            'severity_xgboost_tuned': measure_batch_latency(
                bundle['models']['xgboost_tuned'], X_test, latency['batch_size'], latency['repeats']
            ),
            'claim_probability_xgboost': measure_batch_latency(
                best_classifier, Xc_test, latency['batch_size'], latency['repeats']
            ),
        },
    )

    seed = params['seed']
    shap_importance(best_model, X_test, shap_rows, seed).to_csv(reports_dir / 'shap_severity.csv', index=False)
    shap_importance(best_classifier, Xc_test, shap_rows, seed).to_csv(
        reports_dir / 'shap_claim_probability.csv', index=False
    )


def main(argv=None) -> None:
    evaluate(load_params(parse_args('Evaluate models.', argv).params))


if __name__ == '__main__':
    main()
