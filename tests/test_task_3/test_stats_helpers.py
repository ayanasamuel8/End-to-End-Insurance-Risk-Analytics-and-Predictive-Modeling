import pandas as pd
import pytest

from src.task_3.stats_helpers import anova_test, chi_square_test, t_test_independent


def test_t_test_ignores_missing_values_and_detects_a_gap():
    low = pd.Series([1.0, 2.0, 3.0, None, 2.0])
    high = pd.Series([10.0, 11.0, 12.0, 11.0, None])
    result = t_test_independent(low, high)
    assert result.statistic < 0
    assert result.pvalue < 0.001


def test_anova_detects_different_group_means():
    result = anova_test([[1, 2, 3, 2], [10, 11, 12, 11], [20, 21, 19, 20]])
    assert result.pvalue < 0.001


def test_anova_for_identical_groups_is_not_significant():
    result = anova_test([[1, 2, 3, 4], [1, 2, 3, 4]])
    assert result.pvalue == pytest.approx(1.0)


def test_chi_square_on_associated_columns():
    df = pd.DataFrame({'a': ['x'] * 20 + ['y'] * 20, 'b': ['p'] * 18 + ['q'] * 2 + ['p'] * 2 + ['q'] * 18})
    chi2, p_value, dof, expected = chi_square_test(df, 'a', 'b')
    assert p_value < 0.001
    assert dof == 1
    assert expected.shape == (2, 2)
