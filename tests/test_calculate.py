import pytest
import pandas as pd
from vizdataquality.calculate import calc

def test_calc_with_dataframe_default_options():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    result = calc(df)

    # Assert the structure of the result
    expected_columns = ['Data type', 'Example value', 'Number of values',
                        'Number of zero values', 'Number of missing values',
                        'Number of unique values', 'Value lengths',
                        'Character patterns', 'Numeric percentiles',
                        'Datetime percentiles', 'Uniqueness', 'Data type conflict']
    assert list(result.columns) == expected_columns
    assert all(result.index == df.columns)

    # Assert specific data type and value counts
    assert result.loc['A', 'Data type'] == 'int64'
    assert result.loc['B', 'Data type'] == 'object'
    assert result.loc['A', 'Number of values'] == 3
    assert result.loc['B', 'Number of unique values'] == 3

