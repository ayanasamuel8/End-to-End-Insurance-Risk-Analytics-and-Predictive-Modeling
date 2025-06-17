from scipy.stats import ttest_ind, chi2_contingency, f_oneway
import pandas as pd

def t_test_independent(group_a, group_b):
    return ttest_ind(group_a.dropna(), group_b.dropna(), equal_var=False)

def chi_square_test(df, col1, col2):
    contingency = pd.crosstab(df[col1], df[col2])
    return chi2_contingency(contingency)

def anova_test(groups):
    """
    Perform one-way ANOVA.
    `groups` should be a list/tuple of arrays (or lists) containing your samples.
    Returns the ANOVA statistic and p-value.
    """
    return f_oneway(*groups)