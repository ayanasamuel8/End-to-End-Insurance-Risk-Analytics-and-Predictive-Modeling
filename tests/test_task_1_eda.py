import matplotlib

matplotlib.use('Agg')

import pandas as pd  # noqa: E402

from src.task_1.eda.outlier_detection import detect_outliers_iqr, plot_outliers_box  # noqa: E402
from src.task_1.eda.summary_stats import calculate_loss_ratio, describe_numerical  # noqa: E402


def test_detect_outliers_iqr_flags_only_extreme_rows():
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 100]})
    assert detect_outliers_iqr(df, 'x')['x'].tolist() == [100]


def test_plot_outliers_box_runs_headless():
    plot_outliers_box(pd.DataFrame({'x': [1, 2, 3]}), 'x')


def test_describe_numerical_has_standard_rows():
    summary = describe_numerical(pd.DataFrame({'x': [1.0, 2.0, 3.0]}))
    assert summary.loc['mean', 'x'] == 2.0


def test_loss_ratio_treats_zero_premium_as_missing():
    result = calculate_loss_ratio(pd.DataFrame({'TotalClaims': [50.0, 10.0], 'TotalPremium': [100.0, 0.0]}))
    assert result['LossRatio'].iloc[0] == 0.5
    assert pd.isna(result['LossRatio'].iloc[1])
