"""Stage 1: turn the cleaned claims data into model-ready train and test files."""
from pathlib import Path

import pandas as pd

from src.pipeline.params import ensure_dir, load_params, parse_args
from src.task_4.data_processing import (
    GROUP_COLUMN,
    PROBABILITY_TARGET,
    SEVERITY_TARGET,
    split_claim_probability_data,
    split_claim_severity_data,
)
from src.task_4.feature_engineering import add_features


def prepare(params: dict) -> None:
    out_dir = ensure_dir(params['paths']['processed_dir'])
    settings = params['prepare']

    cleaned = pd.read_csv(params['paths']['cleaned_data'], low_memory=False)
    featured = add_features(cleaned, reference_year=settings['reference_year'])

    splits = {
        'severity': (split_claim_severity_data, SEVERITY_TARGET),
        'claim_probability': (split_claim_probability_data, PROBABILITY_TARGET),
    }
    for name, (builder, target) in splits.items():
        X_train, X_test, y_train, y_test, g_train, g_test = builder(
            featured, test_size=settings['test_size'], random_state=params['seed']
        )
        for part, X, y, groups in (('train', X_train, y_train, g_train), ('test', X_test, y_test, g_test)):
            X.assign(**{target: y, GROUP_COLUMN: groups}).reset_index(drop=True).to_parquet(
                Path(out_dir) / f'{name}_{part}.parquet'
            )


def main(argv=None) -> None:
    prepare(load_params(parse_args('Prepare model inputs.', argv).params))


if __name__ == '__main__':
    main()
