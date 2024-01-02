import pytest
import pandas as pd
import datetime
from vizdataquality.datasets import get_dataset

# Test for 'simple' dataset
def test_get_dataset_simple():
    num_rows, df = get_dataset('simple')
    assert num_rows == 10
    assert list(df.columns) == ['String', 'Integer', 'Float', 'Date']
    assert all(df['String'][:6].notnull())
    assert df['Float'].isnull().sum() == 4
    assert isinstance(df['Date'][0], pd.Timestamp)

def test_get_dataset_missing_1():
    num_rows, df = get_dataset('missing 1')
    assert num_rows == 10000
    assert all(df['A'].notnull())
    assert df['B'].isnull().sum() == 1
    assert df['C'].isnull().sum() == 5000
    assert df['D'].notnull().sum() == 1
    assert all(df['E'].isnull())

def test_get_dataset_numeric_1():
    num_rows, df = get_dataset('numeric 1')
    assert num_rows == 10
    for col in df.columns:
        assert all(df[col].apply(lambda x: isinstance(x, int)))

def test_get_dataset_date_1():
    num_rows, df = get_dataset('date 1')
    assert num_rows == 10
    for col in df.columns:
        assert all(df[col].apply(lambda x: isinstance(x, pd.Timestamp)))

def test_get_dataset_time_1():
    num_rows, df = get_dataset('time 1')
    assert num_rows == 24
    for col in df.columns:
        assert all(df[col].apply(lambda x: isinstance(x, datetime.datetime)))

def test_get_dataset_value_counts_1():
    num_rows, df = get_dataset('value counts 1')
    assert num_rows == 100
    assert df['Categorical'].value_counts()['a'] == 6
    assert df['Categorical (ints)'].dtype == 'object'
    assert df['Date'].dtype == 'datetime64[ns]'

# Test for invalid input
def test_get_dataset_invalid():
    num_rows, __ = get_dataset('foo')
    assert num_rows == 0

def test_get_dataset_datetime_1():
    num_rows, df = get_dataset('datetime 1')
    assert num_rows == 2628000
    assert all(df['Date'].apply(lambda x: isinstance(x, pd.Timestamp)))
