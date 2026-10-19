import numpy as np
import pandas as pd

from src.task_4.data_processing import (
    build_probability_frame,
    build_severity_frame,
    encode_categoricals,
    expand_date_columns,
    is_date_column,
    prepare_claim_probability_data,
    prepare_claim_severity_data,
    split_claim_probability_data,
    split_claim_severity_data,
)
from src.task_4.feature_engineering import add_features


def test_encode_categoricals_is_independent_of_row_order():
    df = pd.DataFrame({'city': ['b', 'a', None, 'c'], 'flag': [True, False, True, False], 'n': [1, 2, 3, 4]})
    encoded = encode_categoricals(df)
    shuffled = encode_categoricals(df.iloc[::-1])

    assert encoded['city'].tolist() == [2, 1, 0, 3]
    assert encoded['flag'].tolist() == [1, 0, 1, 0]
    assert encoded['n'].tolist() == [1, 2, 3, 4]
    assert shuffled['city'].tolist() == encoded['city'].tolist()[::-1]


def test_is_date_column():
    assert is_date_column(pd.Series(['2015-03-01 00:00:00'] * 10))
    assert not is_date_column(pd.Series(['gauteng'] * 10))
    assert not is_date_column(pd.Series([1, 2, 3]))
    assert not is_date_column(pd.Series([None, None], dtype=object))


def test_expand_date_columns_replaces_text_with_year_and_month():
    df = pd.DataFrame({'when': ['2015-03-01 00:00:00', '2016-11-01 00:00:00'], 'x': [1, 2]})
    result = expand_date_columns(df)
    assert list(result.columns) == ['x', 'when_year', 'when_month']
    assert result['when_year'].tolist() == [2015, 2016]
    assert result['when_month'].tolist() == [3, 11]
    assert 'when' in df.columns


def test_severity_frame_keeps_only_positive_claims_and_drops_identifiers(synthetic_df):
    frame = build_severity_frame(add_features(synthetic_df))
    assert (frame['TotalClaims'] > 0).all()
    assert len(frame) == (synthetic_df['TotalClaims'] > 0).sum()
    for column in ('UnderwrittenCoverID', 'claim_indicator', 'ClaimRatio', 'Citizenship'):
        assert column not in frame.columns
    assert 'TransactionMonth_year' in frame.columns
    assert all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes)


def test_probability_frame_has_no_claim_amount_columns(synthetic_df):
    frame = build_probability_frame(add_features(synthetic_df))
    for column in ('TotalClaims', 'ClaimRatio', 'UnderwrittenCoverID'):
        assert column not in frame.columns
    assert set(frame['claim_indicator'].unique()) == {0, 1}
    assert len(frame) == len(synthetic_df)


def test_probability_frame_derives_target_when_missing(synthetic_df):
    frame = build_probability_frame(synthetic_df.drop(columns=['claim_indicator']))
    assert frame['claim_indicator'].sum() == (synthetic_df['TotalClaims'] > 0).sum()


def test_prepare_splits_are_reproducible_and_disjoint(synthetic_df):
    featured = add_features(synthetic_df)
    first = prepare_claim_severity_data(featured, test_size=0.25, random_state=3)
    second = prepare_claim_severity_data(featured, test_size=0.25, random_state=3)
    for left, right in zip(first, second):
        pd.testing.assert_index_equal(left.index, right.index)
    X_train, X_test, y_train, y_test = first
    assert set(X_train.index).isdisjoint(X_test.index)
    assert 0.1 < len(X_test) / (len(X_train) + len(X_test)) < 0.35
    assert 'PolicyID' not in X_train.columns
    assert 'TotalClaims' not in X_train.columns
    assert np.isfinite(y_train).all()


def test_probability_split_keeps_class_balance(synthetic_df):
    _, _, y_train, y_test = prepare_claim_probability_data(add_features(synthetic_df))
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_splits_keep_each_policy_on_one_side(synthetic_df):
    featured = add_features(synthetic_df)
    for splitter in (split_claim_severity_data, split_claim_probability_data):
        X_train, X_test, y_train, y_test, groups_train, groups_test = splitter(featured)
        assert set(groups_train).isdisjoint(groups_test)
        assert len(groups_train) == len(X_train) == len(y_train)
        assert len(groups_test) == len(X_test) == len(y_test)


def test_split_without_policy_ids_falls_back_to_row_split(synthetic_df):
    featured = add_features(synthetic_df).drop(columns=['PolicyID'])
    severity = split_claim_severity_data(featured)
    probability = split_claim_probability_data(featured)
    assert severity[4] is None and probability[5] is None
    assert abs(probability[2].mean() - probability[3].mean()) < 0.01
