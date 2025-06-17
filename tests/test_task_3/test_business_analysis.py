import pytest
from src.task_3.business_analysis import interpret_test_result

def test_interpret_test_result_reject():
    msg = interpret_test_result("Province Risk", 0.01, "Higher risk observed in Gauteng.")
    assert "reject the null hypothesis" in msg
    assert "Higher risk observed" in msg

def test_interpret_test_result_fail():
    msg = interpret_test_result("Gender Risk", 0.2)
    assert "fail to reject the null hypothesis" in msg
    assert "No significant difference" in msg
