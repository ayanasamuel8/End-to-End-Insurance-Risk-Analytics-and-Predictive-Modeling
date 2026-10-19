import numpy as np
import pandas as pd

from src.preprocessing import clean_data, clean_numeric_strings, save_cleaned_data


def test_clean_numeric_strings_converts_comma_numbers():
    df = pd.DataFrame({'amount': ['1,000', '2,500.5', 'bad'], 'other': ['x', 'y', 'z']})
    result = clean_numeric_strings(df, ['amount'])
    assert result['amount'].iloc[:2].tolist() == [1000.0, 2500.5]
    assert np.isnan(result['amount'].iloc[2])
    assert result['other'].tolist() == ['x', 'y', 'z']


def test_clean_numeric_strings_leaves_numeric_columns_alone():
    df = pd.DataFrame({'amount': [1.5, 2.5]})
    assert clean_numeric_strings(df, ['amount'])['amount'].tolist() == [1.5, 2.5]


def make_raw():
    return pd.DataFrame({
        'TotalClaims': [0.0, 100.0, 100.0, np.nan, 5000.0, 0.0],
        'Premium': [10.0, 20.0, 20.0, np.nan, 40.0, 30.0],
        'City': [' Cape Town', 'DURBAN', 'DURBAN', None, 'Durban', 'durban'],
        'Empty': [np.nan] * 6,
    })


def test_clean_data_removes_duplicates_and_fills_gaps():
    cleaned = clean_data(make_raw())
    assert len(cleaned) == 5
    assert cleaned.isna().sum().sum() == 0
    assert 'Empty' not in cleaned.columns
    assert set(cleaned['City']) == {'cape town', 'durban'}


def test_clean_data_caps_numeric_outliers_but_not_the_claim_amount():
    cleaned = clean_data(make_raw())
    assert cleaned['TotalClaims'].max() == 5000.0
    assert cleaned['Premium'].max() < 40.0


def test_clean_data_converts_comma_formatted_text_columns():
    df = pd.DataFrame({'TotalClaims': [0.0, 1.0, 2.0], 'Value': ['1,000', '2,000', '3,000']})
    cleaned = clean_data(df)
    assert pd.api.types.is_numeric_dtype(cleaned['Value'])


def test_clean_data_keeps_all_missing_categorical_columns_out():
    df = pd.DataFrame({'TotalClaims': [1.0, 2.0], 'Notes': [None, None]})
    assert 'Notes' not in clean_data(df).columns


def test_save_cleaned_data_creates_missing_folders(tmp_path):
    target = tmp_path / 'nested' / 'cleaned.csv'
    save_cleaned_data(pd.DataFrame({'a': [1, 2]}), str(target))
    assert pd.read_csv(target)['a'].tolist() == [1, 2]
