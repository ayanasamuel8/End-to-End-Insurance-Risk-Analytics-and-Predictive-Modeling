"""Stage 2: fit the severity and claim probability models."""
from pathlib import Path

import joblib
import pandas as pd

from src.pipeline.params import ensure_dir, load_params, parse_args, set_seed, write_json
from src.task_4.data_processing import PROBABILITY_TARGET, SEVERITY_TARGET
from src.task_4.model_training import fit_claim_probability_models, fit_severity_models, select_best_model


def load_split(processed_dir, name: str, target: str):
    train = pd.read_parquet(Path(processed_dir) / f'{name}_train.parquet')
    test = pd.read_parquet(Path(processed_dir) / f'{name}_test.parquet')
    features = [c for c in train.columns if c != target]
    return train[features], train[target], test[features], test[target]


def train(params: dict) -> None:
    seed = params['seed']
    set_seed(seed)
    processed_dir = params['paths']['processed_dir']
    models_dir = ensure_dir(params['paths']['models_dir'])
    reports_dir = params['paths']['reports_dir']

    X_train, y_train, _, _ = load_split(processed_dir, 'severity', SEVERITY_TARGET)
    models, cv_rmse, best_params = fit_severity_models(X_train, y_train, params['severity'], seed)
    best_name = select_best_model(cv_rmse)
    joblib.dump(
        {'features': list(X_train.columns), 'models': models, 'best': best_name},
        Path(models_dir) / 'severity.joblib',
    )
    write_json(
        Path(reports_dir) / 'model_selection.json',
        {
            'severity': {
                'cv_rmse': cv_rmse,
                'selected_model': best_name,
                'xgboost_tuned_params': best_params,
                'train_rows': len(X_train),
            }
        },
    )

    X_train, y_train, _, _ = load_split(processed_dir, 'claim_probability', PROBABILITY_TARGET)
    classifiers = fit_claim_probability_models(X_train, y_train, params['claim_probability'], seed)
    joblib.dump(
        {'features': list(X_train.columns), 'models': classifiers, 'best': 'xgboost'},
        Path(models_dir) / 'claim_probability.joblib',
    )


def main(argv=None) -> None:
    train(load_params(parse_args('Train models.', argv).params))


if __name__ == '__main__':
    main()
