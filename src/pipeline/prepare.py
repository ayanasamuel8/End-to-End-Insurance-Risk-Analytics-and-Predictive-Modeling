"""Stage 1: turn the cleaned claims data into model-ready train and test files."""
from pathlib import Path

import pandas as pd

from src.pipeline.params import ensure_dir, load_params, parse_args
from src.task_4.data_processing import (
    PROBABILITY_TARGET,
    SEVERITY_TARGET,
    prepare_claim_probability_data,
    prepare_claim_severity_data,
)
from src.task_4.feature_engineering import add_features


def prepare(params: dict) -> None:
    out_dir = ensure_dir(params['paths']['processed_dir'])
    settings = params['prepare']

    cleaned = pd.read_csv(params['paths']['cleaned_data'], low_memory=False)
    featured = add_features(cleaned, reference_year=settings['reference_year'])

    splits = {
        'severity': (prepare_claim_severity_data, SEVERITY_TARGET),
        'claim_probability': (prepare_claim_probability_data, PROBABILITY_TARGET),
    }
    for name, (builder, target) in splits.items():
        X_train, X_test, y_train, y_test = builder(
            featured, test_size=settings['test_size'], random_state=params['seed']
        )
        X_train.assign(**{target: y_train}).reset_index(drop=True).to_parquet(Path(out_dir) / f'{name}_train.parquet')
        X_test.assign(**{target: y_test}).reset_index(drop=True).to_parquet(Path(out_dir) / f'{name}_test.parquet')


def main(argv=None) -> None:
    prepare(load_params(parse_args('Prepare model inputs.', argv).params))


if __name__ == '__main__':
    main()
