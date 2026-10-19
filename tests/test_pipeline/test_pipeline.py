import hashlib
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

from src.pipeline import check_metrics, evaluate, prepare, synthetic, train
from src.pipeline.params import load_params, parse_args, read_json, set_seed, write_json

ROOT = Path(__file__).resolve().parent.parent.parent


def run_pipeline(params):
    prepare.prepare(params)
    train.train(params)
    evaluate.evaluate(params)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def finished_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('run')
    params = load_params(str(ROOT / 'params.ci.yaml'))
    params['paths'] = {
        'cleaned_data': str(tmp_path / 'cleaned_data.csv'),
        'processed_dir': str(tmp_path / 'processed'),
        'models_dir': str(tmp_path / 'models'),
        'reports_dir': str(tmp_path / 'reports'),
    }
    synthetic.make_cleaned_data(2500, seed=11).to_csv(params['paths']['cleaned_data'], index=False)
    run_pipeline(params)
    return params


def test_pipeline_writes_every_artifact(finished_run):
    paths = finished_run['paths']
    for name in ('severity_train', 'severity_test', 'claim_probability_train', 'claim_probability_test'):
        assert (Path(paths['processed_dir']) / f'{name}.parquet').exists()
    for name in ('severity.joblib', 'claim_probability.joblib'):
        assert (Path(paths['models_dir']) / name).exists()
    for name in ('metrics.json', 'latency.json', 'model_selection.json'):
        assert (Path(paths['reports_dir']) / name).exists()
    for name in ('shap_severity.csv', 'shap_claim_probability.csv'):
        assert (Path(paths['reports_dir']) / name).exists()


def test_metrics_report_every_candidate(finished_run):
    metrics = read_json(Path(finished_run['paths']['reports_dir']) / 'metrics.json')
    assert set(metrics['severity']['test']) == {
        'mean_baseline', 'linear_regression', 'random_forest', 'xgboost_default', 'xgboost_tuned'
    }
    assert set(metrics['severity']['xgboost_tuned_rmse_reduction_pct']) == {
        'mean_baseline', 'linear_regression', 'random_forest', 'xgboost_default'
    }
    assert set(metrics['claim_probability']['test']) == {'logistic_regression', 'xgboost'}


def test_claim_probability_does_not_use_the_claim_amount(finished_run):
    bundle = joblib.load(Path(finished_run['paths']['models_dir']) / 'claim_probability.joblib')
    assert 'TotalClaims' not in bundle['features']
    metrics = read_json(Path(finished_run['paths']['reports_dir']) / 'metrics.json')
    assert metrics['claim_probability']['test']['xgboost']['accuracy'] < 0.999


def test_train_and_test_rows_do_not_overlap(finished_run):
    folder = Path(finished_run['paths']['processed_dir'])
    train_rows = pd.read_parquet(folder / 'severity_train.parquet')
    test_rows = pd.read_parquet(folder / 'severity_test.parquet')
    selection = read_json(Path(finished_run['paths']['reports_dir']) / 'model_selection.json')
    assert selection['severity']['train_rows'] == len(train_rows)
    assert len(test_rows) == pytest.approx(0.25 * len(train_rows), rel=0.02)


def test_rerun_gives_identical_outputs(finished_run, tmp_path):
    """Running the stages again from the same input must reproduce every file byte for byte."""
    second = {**finished_run, 'paths': {
        **finished_run['paths'],
        'processed_dir': str(tmp_path / 'processed'),
        'models_dir': str(tmp_path / 'models'),
        'reports_dir': str(tmp_path / 'reports'),
    }}
    run_pipeline(second)

    compared = [
        ('processed_dir', 'severity_train.parquet'),
        ('processed_dir', 'severity_test.parquet'),
        ('processed_dir', 'claim_probability_train.parquet'),
        ('processed_dir', 'claim_probability_test.parquet'),
        ('models_dir', 'severity.joblib'),
        ('models_dir', 'claim_probability.joblib'),
        ('reports_dir', 'metrics.json'),
        ('reports_dir', 'model_selection.json'),
        ('reports_dir', 'shap_severity.csv'),
        ('reports_dir', 'shap_claim_probability.csv'),
    ]
    for folder, name in compared:
        assert digest(Path(finished_run['paths'][folder]) / name) == digest(Path(second['paths'][folder]) / name), name


def test_latency_is_reported_for_both_models(finished_run):
    latency = read_json(Path(finished_run['paths']['reports_dir']) / 'latency.json')
    assert set(latency) == {'severity_xgboost_tuned', 'claim_probability_xgboost'}
    configured = finished_run['evaluate']['latency']['batch_size']
    assert 0 < latency['severity_xgboost_tuned']['batch_size'] <= configured
    assert 0 < latency['claim_probability_xgboost']['batch_size'] <= configured


def test_gates_pass_for_a_healthy_run(finished_run, capsys):
    reports = Path(finished_run['paths']['reports_dir'])
    gates = finished_run['gates']
    relaxed = {**gates, 'max_batch_latency_ms': 10_000}
    metrics = read_json(reports / 'metrics.json')
    latency = read_json(reports / 'latency.json')
    assert check_metrics.find_violations(metrics, latency, relaxed) == []


def test_gates_flag_every_kind_of_regression(finished_run):
    reports = Path(finished_run['paths']['reports_dir'])
    metrics = read_json(reports / 'metrics.json')
    latency = read_json(reports / 'latency.json')
    strict = {
        'max_batch_latency_ms': 0.0,
        'min_rmse_reduction_vs_xgboost_default_pct': 100.0,
        'min_claim_probability_roc_auc': 1.1,
    }
    violations = check_metrics.find_violations(metrics, latency, strict)
    assert len(violations) == 4
    assert any('RMSE' in v for v in violations)
    assert any('ROC AUC' in v for v in violations)
    assert sum('latency' in v for v in violations) == 2


def test_check_metrics_exit_codes(finished_run, tmp_path):
    params_path = tmp_path / 'params.yaml'
    relaxed = {**finished_run, 'gates': {**finished_run['gates'], 'max_batch_latency_ms': 10_000}}
    params_path.write_text(yaml.safe_dump(relaxed))
    assert check_metrics.main(['--params', str(params_path)]) == 0

    failing = {**relaxed, 'gates': {**relaxed['gates'], 'min_claim_probability_roc_auc': 1.1}}
    params_path.write_text(yaml.safe_dump(failing))
    assert check_metrics.main(['--params', str(params_path)]) == 1


def test_stage_entry_points_read_the_params_file(tmp_path):
    params = load_params(str(ROOT / 'params.ci.yaml'))
    params['paths'] = {
        'cleaned_data': str(tmp_path / 'in' / 'cleaned_data.csv'),
        'processed_dir': str(tmp_path / 'processed'),
        'models_dir': str(tmp_path / 'models'),
        'reports_dir': str(tmp_path / 'reports'),
    }
    params['severity']['search']['n_iter'] = 2
    params_path = tmp_path / 'params.yaml'
    params_path.write_text(yaml.safe_dump(params))

    synthetic.main(['--out', params['paths']['cleaned_data'], '--rows', '1200', '--seed', '5'])
    prepare.main(['--params', str(params_path)])
    train.main(['--params', str(params_path)])
    evaluate.main(['--params', str(params_path)])
    assert (tmp_path / 'reports' / 'metrics.json').exists()


def test_synthetic_data_is_deterministic():
    pd.testing.assert_frame_equal(synthetic.make_cleaned_data(300, seed=1), synthetic.make_cleaned_data(300, seed=1))
    assert not synthetic.make_cleaned_data(300, seed=1).equals(synthetic.make_cleaned_data(300, seed=2))


def test_params_helpers(tmp_path):
    assert parse_args('x', []).params == 'params.yaml'
    assert parse_args('x', ['--params', 'other.yaml']).params == 'other.yaml'
    write_json(tmp_path / 'a' / 'b.json', {'z': 1, 'a': 2})
    assert (tmp_path / 'a' / 'b.json').read_text() == '{\n  "a": 2,\n  "z": 1\n}\n'
    assert read_json(tmp_path / 'a' / 'b.json') == {'a': 2, 'z': 1}
    set_seed(3)


def test_dvc_stages_point_at_real_modules_and_params():
    pipeline = yaml.safe_load((ROOT / 'dvc.yaml').read_text())
    params = load_params(str(ROOT / 'params.yaml'))
    assert list(pipeline['stages']) == ['prepare', 'train', 'evaluate']
    for stage in pipeline['stages'].values():
        module = stage['cmd'].split()[-1]
        assert (ROOT / (module.replace('.', '/') + '.py')).exists()
        for dep in stage['deps']:
            if dep.endswith('.py') or dep.endswith('.csv'):
                assert dep.startswith('data/') or (ROOT / dep).exists(), dep
        for key in stage['params']:
            node = params
            for part in key.split('.'):
                node = node[part]


def test_ci_params_match_the_real_params_shape():
    def shape(node):
        return {k: shape(v) for k, v in node.items()} if isinstance(node, dict) else None

    real = load_params(str(ROOT / 'params.yaml'))
    small = load_params(str(ROOT / 'params.ci.yaml'))
    assert shape(real) == shape(small)
