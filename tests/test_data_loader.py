import pandas as pd
from src.data_loader import check_missing

def test_missing():
    df = pd.DataFrame({"A": [1, None, 2]})
    assert check_missing(df).A == 1