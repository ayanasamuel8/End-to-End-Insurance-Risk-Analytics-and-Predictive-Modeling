import pandas as pd
from src.data_loader import load_data
from src.config import RAW_DATA_PATH


def test_data_not_empty():
    df = load_data(RAW_DATA_PATH)
    assert not df.empty, "Data is empty!"

def test_missing_values():
    df = load_data()
    assert df.isnull().sum().sum() < 0.05 * df.size, "Too many missing values!"

def test_numeric_ranges():
    df = load_data()
    assert (df["TotalPremium"] >= 0).all(), "Negative premiums found!"
    assert (df["TotalClaimAmount"] >= 0).all(), "Negative claims found!"
