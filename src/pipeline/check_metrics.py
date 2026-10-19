"""Fail when the latest metrics break a limit set under ``gates`` in the params file."""
import sys
from pathlib import Path

from src.pipeline.params import load_params, parse_args, read_json


def find_violations(metrics: dict, latency: dict, gates: dict) -> list:
    violations = []

    reduction = metrics['severity']['xgboost_tuned_rmse_reduction_pct']['xgboost_default']
    if reduction < gates['min_rmse_reduction_vs_xgboost_default_pct']:
        violations.append(
            f'Tuned XGBoost RMSE reduction vs default is {reduction:.2f}%, '
            f"below {gates['min_rmse_reduction_vs_xgboost_default_pct']}%."
        )

    roc_auc = metrics['claim_probability']['test']['xgboost']['roc_auc']
    if roc_auc < gates['min_claim_probability_roc_auc']:
        violations.append(
            f"Claim probability ROC AUC is {roc_auc:.3f}, below {gates['min_claim_probability_roc_auc']}."
        )

    for name, timing in latency.items():
        if timing['batch_ms_median'] > gates['max_batch_latency_ms']:
            violations.append(
                f"{name} median batch latency is {timing['batch_ms_median']:.2f} ms, "
                f"above {gates['max_batch_latency_ms']} ms."
            )
    return violations


def main(argv=None) -> int:
    params = load_params(parse_args('Check metric gates.', argv).params)
    reports_dir = Path(params['paths']['reports_dir'])
    violations = find_violations(
        read_json(reports_dir / 'metrics.json'), read_json(reports_dir / 'latency.json'), params['gates']
    )
    for violation in violations:
        print(f'FAIL: {violation}')
    if not violations:
        print('All metric gates passed.')
    return 1 if violations else 0


if __name__ == '__main__':
    sys.exit(main())
