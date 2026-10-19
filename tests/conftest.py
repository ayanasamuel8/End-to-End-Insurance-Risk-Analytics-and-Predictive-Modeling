from pathlib import Path

import pytest
import yaml

from src.pipeline.synthetic import make_cleaned_data

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope='session')
def synthetic_df():
    return make_cleaned_data(n_rows=2500, seed=7)


@pytest.fixture
def ci_params(tmp_path, synthetic_df):
    """Pipeline params from params.ci.yaml, pointed at a temporary folder."""
    with open(ROOT / 'params.ci.yaml', encoding='utf-8') as handle:
        params = yaml.safe_load(handle)
    params['paths'] = {
        'raw_data': str(tmp_path / 'raw_data.txt'),
        'cleaned_data': str(tmp_path / 'cleaned_data.csv'),
        'processed_dir': str(tmp_path / 'processed'),
        'models_dir': str(tmp_path / 'models'),
        'reports_dir': str(tmp_path / 'reports'),
    }
    synthetic_df.to_csv(params['paths']['cleaned_data'], index=False)
    return params
