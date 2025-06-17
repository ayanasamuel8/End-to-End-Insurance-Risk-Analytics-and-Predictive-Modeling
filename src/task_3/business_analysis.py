def interpret_test_result(test_name, p_value, additional_info=""):
    """
    Returns a clear business interpretation of the test result.
    If p_value < 0.05, we reject the null hypothesis, and we can add further explanation.
    """
    if p_value < 0.05:
        interpretation = f"We reject the null hypothesis for {test_name} (p = {p_value:.3f}). {additional_info}"
    else:
        interpretation = f"We fail to reject the null hypothesis for {test_name} (p = {p_value:.3f}). No significant difference observed."
    return interpretation