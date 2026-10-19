import pandas as pd

from src.data_loader import (
    check_missing,
    check_structure,
    extract_categorical_cols,
    extract_date_time_cols,
    extract_numeric_cols,
    load_data,
    load_raw_data,
    save_raw_data,
)


def test_missing():
    df = pd.DataFrame({"A": [1, None, 2]})
    assert check_missing(df).A == 1


def test_load_data_reads_pipe_separated_text(tmp_path):
    path = tmp_path / 'raw.txt'
    path.write_text('a|b\n1|x\n2|y\n')
    df = load_data(str(path))
    assert df.columns.tolist() == ['a', 'b']
    assert df['a'].tolist() == [1, 2]


def test_save_and_load_raw_data_round_trip(tmp_path):
    path = tmp_path / 'raw.csv'
    save_raw_data(pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}), str(path))
    assert load_raw_data(str(path)).to_dict('list') == {'a': [1, 2], 'b': ['x', 'y']}


def test_check_structure_returns_info_and_dtypes():
    info, dtypes = check_structure(pd.DataFrame({'a': [1]}))
    assert info is None
    assert dtypes['a'] == 'int64'


def test_column_type_extractors():
    df = pd.DataFrame({
        'n': [1, 2],
        'f': [0.5, 1.5],
        'c': ['x', 'y'],
        'd': pd.to_datetime(['2015-01-01', '2015-02-01']),
    })
    assert extract_numeric_cols(df) == ['n', 'f']
    assert extract_categorical_cols(df) == ['c']
    assert extract_date_time_cols(df) == ['d']
