import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from vizdataquality.plot import datetime_counts
import os

data = pd.Series(pd.date_range('2020-01-01', periods=100, freq='D'))
data7d = pd.Series(pd.date_range('2020-01-01', periods=100, freq='7D'))  # Create gaps

# Test with basic datetime data
def test_datetime_counts_basic():
    datetime_counts(data)
    plt.close('all')  # Close plot to avoid memory issues

# Test with specific datetime component (e.g., 'month')
def test_datetime_counts_component():
    datetime_counts(data, component='month')
    plt.close('all')

# # Test with gap_threshold parameter
# def test_datetime_counts_gap_threshold():
#     datetime_counts(data7d, component='day', gap_threshold=10)
#     plt.close('all')

# # Test the show_gaps functionality
# def test_datetime_counts_show_gaps():
#     datetime_counts(data7d, component='day', show_gaps=False)
#     plt.close('all')

# Test with custom axis input
def test_datetime_counts_custom_axis():
    fig, ax = plt.subplots()
    datetime_counts(data, ax_input=ax)
    plt.close('all')

# Test with label rotation
def test_datetime_counts_label_rotation():
    datetime_counts(data, xlabels_rotate=45)
    plt.close('all')

# Test saving plot to a file
def test_datetime_counts_save_file():
    test_filename = 'test_plot.png'
    datetime_counts(data, filename=test_filename)
    assert os.path.exists(test_filename)
    os.remove(test_filename)

# Test overwrite functionality
def test_datetime_counts_overwrite_file():
    test_filename = 'test_plot.png'
    with open(test_filename, 'w') as f:
        f.write('test')  # Create a dummy file
    datetime_counts(data, filename=test_filename, overwrite=True)
    assert os.path.exists(test_filename)
    os.remove(test_filename)

# Test with empty data
# def test_datetime_counts_empty_data():
#     data = pd.Series([])
#     datetime_counts(data)
#     plt.close('all')

# # Test with invalid component
# def test_datetime_counts_invalid_component():
#     with pytest.raises(ValueError):
#         datetime_counts(data, component='invalid_component')
#     plt.close('all')
