# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 04:43:43 2023

   Copyright 2023 Roy Ruddle

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
"""

import os
import pandas as pd
import numpy as np
import numbers
import sys

import re
import datetime

import logging
import traceback


# =============================================================================
# High-level convenience functions
# =============================================================================
def step1_datafile_stats(encoding_results=None, filename=None, df=None):
    """
    Get general statistics anf information about a datafile.

    Parameters
    ----------
    encoding_results : dict
        Dictionary containing a text file's 'encoding' and 'confidence' (e.g., from utils.detect_file_encoding()).
    filename : str
        Full pathname of datafile (default is None). Used to determine text file encoding.
    df : DataFrame
        The data (default is None). Used for the other statistics.

    Returns
    -------
    DataFrame
        DataFrame with the columns ['Statistic', 'Value']

    """
    data = []
    
    if encoding_results is not None:
        data.append(['Encoding', encoding_results['encoding'] + ' (confidence=' + str(encoding_results['confidence']) + ')'])
        
    if filename is not None:
        try:
            file_info = os.stat(filename)
            data.append(['Size (bytes)', file_info.st_size])
        except:
            raise
    
    if df is not None:
        data.append(['Number of rows', df.shape[0]])
        data.append(['Number of columns', df.shape[1]])
        column_stats = calc(df, options={'Missing values': True})
        data.append(['Number of blank values', column_stats['Number of missing values'].sum()])
    
    return pd.DataFrame(data, columns=['Statistic', 'Value'])


def step1_issues(df):
    """
    Get Step 1 data quality issues. If none are detected then an empty DataFrame is returned.

    Parameters
    ----------
    df : DataFrame
        The data.

    Returns
    -------
    DataFrame
        DataFrame with the columns ['Data quality issue', 'Value or description'].

    """
    data = []
    
    missing_col_names = get_missing_column_names(df)
    if len(missing_col_names) > 0:
        data.append(['Number of missing column names', str(len(missing_col_names))])
    
    col_names_to_trim = get_column_names_to_trim(df)
    if len(col_names_to_trim) > 0:
        data.append(['Number of column names with leading/trailing spaces', str(len(col_names_to_trim))])
    
    duplicate_header = check_for_duplicate_header(df)
    if duplicate_header:
        data.append(['Duplicate header', 'Some rows are the same as the column names'])
        
    num_empty_rows = get_num_empty_rows(df)
    if num_empty_rows > 0:
        data.append(['Number of empty rows', str(num_empty_rows)])
    
    num_empty_cols = get_num_empty_cols(df)
    if num_empty_cols > 0:
        data.append(['Number of empty columns', str(num_empty_cols)])
    
    extra_col = check_for_extra_column(df)
    if extra_col:
        data.append(['Extra column', "Occurs because every row ends with the separator (e.g., ',')"])
    
    
    return pd.DataFrame(data, columns=['Data quality issue', 'Value or description'])


# =============================================================================
# Utility functions
# =============================================================================
def check_for_duplicate_header(df):
    """
    Check whether a dataframe contains any rows that are the same as the header, ignoring any empty columns.

    Parameters
    ----------
    df : DataFrame
        The data.

    Returns
    -------
    bool
        True (the header row is duplicated) or False

    """
    # Count the number of non-null values in each column
    count_values = df.notnull().sum()
    # Omit any empty columns
    usecols = count_values[count_values > 0].index.tolist()

    df2 = pd.concat([df[usecols], pd.DataFrame([df.columns], columns=df.columns)[usecols]], ignore_index=True)

    if df2.shape[1] > 0:
        first_col = df2.columns[0]
        df3 = df2[df2[first_col] == first_col]
    
        if len(df3) > 0:
            df4 = df3.duplicated()
            ret = df4.iloc[-1]
        else:
            # No values in the first column match the header
            ret = False
    else:
        # All of the input dataframe's columns are empty
        ret = False
    
    return ret


def check_for_extra_column(df):
    """
    Check whether a dataframe ends with an extra (superfluous) column.

    Parameters
    ----------
    df : DataFrame
        The data.

    Returns
    -------
    bool
        True (there is an extra column) or False

    """
    num_values = df[df.columns[-1]].notnull().sum()
    
    ret = (num_values == 0) and (df.columns[-1] == 'Unnamed: ' + str(df.shape[1] - 1))
    
    return ret
        
        
def get_column_names_to_trim(df):
    """
    Return a list of any column names that have leading and/or trailing spaces.

    Parameters
    ----------
    df : DataFrame
        The data.

    Returns
    -------
    list
        The names of any missing columns that should be trimmed.

    """
    col_names_to_trim = []
    
    for col in df.columns:
        if len(col.strip()) < len(col):
            col_names_to_trim.append(col)
    
    return col_names_to_trim
        
        
def get_missing_column_names(df):
    """
    Return a list of the names Pandas has created for columns that had no name.

    Parameters
    ----------
    df : DataFrame
        The data.

    Returns
    -------
    list
        The names for any missing columns (an empty list if no column names are missing).

    """
    missing_col_names = []
    
    for l1 in range(len(df.columns)):
        if df.columns[l1] == 'Unnamed: ' + str(l1):
            missing_col_names.append(df.columns[l1])
    
    return missing_col_names
        
        
def get_non_numeric_values(df, convert_numbers=False):
    """
    Return a list of the unique, non-numeric values in a dataframe.

    Parameters
    ----------
    df : DataFrame
        The data.
    convert_numbers : bool, optional
        Whether to exclude numbers stored as strings. The default is False.

    Returns
    -------
    list
        The unique, non-numeric values.

    """
    # Create a set containing the non-numeric values
    non_numeric_values = set()

    for col in df.columns:
        for item in df[col].dropna().unique().tolist():
            if pd.api.types.is_numeric_dtype(item) == False:
                
                if convert_numbers:
                    try:
                        num = float(item)
                    except:
                        non_numeric_values.add(item)
                        pass
                else:
                    non_numeric_values.add(item)

    # Convert the set to a sorted list
    nnv_list = list(non_numeric_values)
    nnv_list.sort()
    
    return nnv_list


def get_num_empty_cols(df):
    """
    Calculate the number of columns that do not contain any values.

    Parameters
    ----------
    df : DataFrame
        The data.

    Returns
    -------
    int
        The number of columns that do not contain any values.

    """
    nvals = df.count()
    
    return len(nvals[nvals == 0])


def get_num_empty_rows(df):
    """
    Calculate the number of rows that do not contain any values.

    Parameters
    ----------
    df : DataFrame
        The data.

    Returns
    -------
    int
        The number of rows that do not contain any values.

    """
    return len(df[df.isna().all(axis=1)])


def get_df_extra_values(df1, df2, convert_numbers=False):
    """
    Return the values that are not in both data frames. Optionally, numbers may be converted so that 1 and 1.0 are considered to be the same as '1', and 1.1 is the same as '1.1', etc.

    Parameters
    ----------
    df1 : DataFrame
        A data frames.
    df2 : DataFrame
        A data frames.
    convert_numbers : bool, optional
        Whether to convert numbers to strings. The default is False.

    Returns
    -------
    extra_values : set
        A set of the values that are not in corresponding columns of both data frames. None is returned if the data frames do not have the same column names.

    """
    # Check the dataframes have the same columns
    if df1.columns.tolist() == df2.columns.tolist():
        extra_values = set()
        
        for col in df1.columns:
            extra_values.update(get_series_extra_values(df1[col], df2[col], convert_numbers))
            
    else:
        extra_values = None
        
    return extra_values


def get_series_extra_values(series1, series2, convert_numbers=False):
    """
    Return the values that are not in both series. Optionally, numbers may be converted so that 1 and 1.0 are considered to be the same as '1', and 1.1 is the same as '1.1', etc.

    Parameters
    ----------
    series1 : series
        A series.
    series2 : series
        A series.
    convert_numbers : bool, optional
        Whether to convert numbers to strings. The default is False.

    Returns
    -------
    extra_values : set
        A set of the values that are not in both series.

    """
    slist = [series1, series2]

    if convert_numbers is False or slist[0].dtype == slist[1].dtype:
        # The series have the same data type
        unique_col_values = [set(ss.dropna().unique().tolist()) for ss in slist]
    else:
        # The series have different data types and numbers are to be converted before comparing the series
        unique_col_values = []

        # Create a list of the converted unique values in each list
        for ss in slist:
            uu = ss.dropna().unique().tolist()

            for l1 in range(len(uu)):
                
                if isinstance(uu[l1], numbers.Number):
                    # Whole numbers (1, 1.0, etc) are stored as strings of integers (e.g., '1').
                    # Other numbers are stored as strings of floats (e.g., '1.2')
                    iu = int(uu[l1])
                    uu[l1] = str(iu) if abs(iu - uu[l1]) <= 0.0 else str(uu[l1])
                else:
                    # Non-numeric values are not converted
                    uu[l1] = uu[l1]

            # Create a set of the converted values in this series
            unique_col_values.append(set(uu))

    # Find the extra values
    extra_values = unique_col_values[0].symmetric_difference(unique_col_values[1])

    return extra_values


def get_value_lengths_examples(df):
    """
    Get examples of the shortest, median and longest values of each column in a dataframe.

    Parameters
    ----------
    df : DataFrame
        The data.


    Returns
    -------
    DataFrame
        A dataframe containing the examples. The first column ('Examples') specifies what each row contains (e.g., Shortest value).

    """
    data = {'Examples': ['Shortest value', 'Median value', 'Longest value']}
    
    for cc in df.columns:
        # Get the unique values in this column
        uu = pd.DataFrame(df[cc].dropna().unique()).rename(columns={0: cc})
        # Calculate the value lengths
        uu['Value length'] = uu[cc].apply(lambda x: len(str(x)))
        uu.sort_values(by=['Value length'], inplace=True)

        data[cc] = [uu.iloc[i][cc] for i in [0, int(len(uu)/2), len(uu)-1]]
            
    df_examples = pd.DataFrame.from_dict(data)
    
    return df_examples


# =============================================================================
# Main functions
# =============================================================================
def calc(df, options=None):
    """
    Profile a data frame or series to calculate aspects of data quality and descriptive statistics.

    Parameters
    ----------
    df : DataFrame or Series
        The data.
    options : dict, optional
        The descriptive statistics to output (default is None; output everything)

    Returns
    -------
    DataFrame
        The descriptive statistics (seperate row for each variable; variable names are the index; columns are different descriptive statistics).

    """
    function_name = 'calc()'
    logger = logging.getLogger('QCprofiling')
    logger.info('%s,dataframe: number of rows =,%d,number of columns,%d' %(function_name, df.shape[0], df.shape[1]))
    #
    # Define the output dataframe's columns
    #
    cols = ['_column']

    if options is None or options.get('Data type', False):
        cols.append('Data type')
        
    if options is None or options.get('Example value', False):
        cols.append('Example value')

    if options is None or options.get('Number of values', False):
        cols.append('Number of values')

    if options is None or options.get('Zero values', False):
        cols.append('Number of zero values')

    if options is None or options.get('Missing values', False):
        cols.append('Number of missing values')

    if options is None or options.get('Unique values', False):
        cols.append('Number of unique values')
    
    if options is None or options.get('Value lengths', False):
        cols.append('Value lengths')
    
    if options is None or options.get('Character pattern', False):
        # Omitted because _get_column_patterns() does not handle some symbols correctly
        cols.append('Character patterns')
    
    if options is None or options.get('Numeric percentiles', False):
        cols.append('Numeric percentiles')
    
    if options is None or options.get('Datetime percentiles', False):
        cols.append('Datetime percentiles')
    #
    # Calculate each column's descriptive statistics
    #
    st = datetime.datetime.now()
    data = []
    
    if isinstance(df,pd.Series):
        # Calculate descriptive statistics for a series
        data.append(_profile_column(df, options))
    else:
        # Calculate descriptive statistics for each column of a dataframe
        for col in df.columns:
            try:
                data.append(_profile_column(df[col], options))
            except: 
                # A message should already have been printed
                #break
                raise
    #
    # Create the output dataframe
    #
    df_output = pd.DataFrame(data, columns=cols)
    # Make the variable names the index
    df_output.set_index('_column', inplace=True)
    #
    # Derive aspects of data quality from the calculations that have just been done.
    # First, derive the type of uniqueness.
    #
    try:
        df_output['Uniqueness'] = _derive_uniqueness(df_output)
    except:
        pass

    # Derive the data type conflict
    try:
        df_output['Data type conflict'] = _derive_data_type_conflict(df_output, df)
    except:
        pass

    et = datetime.datetime.now()
    logger.info("%s,time (s),%s" %(function_name, str((et-st).total_seconds())))
    
    return df_output


# =============================================================================
# Internal functions
# =============================================================================
def _apply_uniqueness(row):
    """
    DataFrame.apply() function to derive the type of uniqueness from the number of values, missing values and unique values.

    Parameters
    ----------
    row : dataframe columns
        ['Number of values', 'Number of missing values', 'Number of unique values'].

    Returns
    -------
    value : string
        The uniqueness (e.g., 'All unique').

    """
    
    nval = row['Number of values']
    nmiss = row['Number of missing values']
    nunique = row['Number of unique values']
    
    if nval == 0 or nval == nmiss:
        value = 'All missing'
    elif nval == nunique:
        value = 'All unique'
    elif nval - nmiss == nunique:
        value = 'Unique/Missing'
    elif nunique == 1:
        value = 'One value' + ('' if nmiss == 0 else '/Missing')
    elif nunique == 2:
        value = 'Binary' + ('' if nmiss == 0 else '/Missing')
    else:
        value = 'Other'
            
        
    return value


def _derive_uniqueness(df_output):
    """
    Derive the type of uniqueness from the number of values, missing values and unique values.

    Parameters
    ----------
    df_output : dataframe
        Dataframe containing the columns ['Number of values', 'Number of missing values', 'Number of unique values'].

    Returns
    -------
    A series containing the uniqueness of each record, or None if it cannot be calculated.

    """
    cols = ['Number of values', 'Number of missing values', 'Number of unique values']
    
    if sum([0 if c in df_output.columns else 1 for c in cols]) == 0:
        # df_output contains all the necessary columns
        return df_output[cols].apply(lambda row: _apply_uniqueness(row), axis=1)
    else:
        return None


def _derive_data_type_conflict(df_output, df):
    """
    Derive the type of data type conflict from the number of values, missing values and unique values.

    Parameters
    ----------
    df_output : dataframe
        Dataframe containing the columns ['Data type', 'Example value', 'Number of missing values'].
    df : dataframe
        The data.

    Returns
    -------
    A series containing the data type conflict of each record, or None if it cannot be calculated.

    """
    # Derive the data type conflict
    #
    cols = ['Data type', 'Example value', 'Number of missing values']
    
    if sum([0 if c in df_output.columns else 1 for c in cols]) == 0:
        # df_output contains all the necessary columns
        values = [None] * len(df_output)
        l1 = 0
        
        for index, row in df_output[cols].iterrows():
            vv = 'OK'
            
            if pd.api.types.is_number(row['Example value']):
                
                if row['Data type'] == 'object':
                    vv = 'Mixed types'
                elif row['Data type'] == 'float64' and row['Number of missing values'] > 0:
                    if df[index].sum() == df[index].dropna().apply(lambda x: int(x)).sum():
                        vv = 'Integer/Missing'
                
            values[l1] = vv
            l1 += 1
    else:
        values = None
        
    return values


def _profile_column(series, options=None):
    """
    Internal function that profiles a data frame column or a series.
    

    Parameters
    ----------
    series : TYPE
        DESCRIPTION.
    options : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    function_name = '_profile_column()'
    logger = logging.getLogger('vizdataquality')
    logger.debug("%s,column,%s" %(function_name, series.name))
    num_missing = None
    # NB: series.unique() includes NaT as a value, but series.value_counts() does not
    value_counts = None

    output = []
    # Output the name of the column
    output.append(series.name)

    if options is None or options.get('Data type', False):
        output.append(str(series.dtype))
        
    if options is None or options.get('Example value', False):

        if value_counts is None:
            value_counts = series.value_counts()
        
        # The example is the first value or an empty string if all of the values are missing
        output.append('' if len(value_counts) == 0 else value_counts.index[0])
    
    if options is None or options.get('Number of values', False):
        # The number of values that are present
        num_missing = series.isnull().sum()
        output.append(len(series) - num_missing)
    
    if options is None or options.get('Zero values', False):
        # The number of values that equal zero
        if pd.api.types.is_string_dtype(series):
            num_zero = len(series[series == '0'])
        else:
            num_zero = series.eq(0).sum()
            
        output.append(num_zero)
    
    if options is None or options.get('Missing values', False):
        
        if num_missing is None:
            # Avoid calculating this again
            num_missing = series.isnull().sum()

        output.append(num_missing)

    
    if options is None or options.get('Unique values', False):

        if value_counts is None:
            value_counts = series.value_counts()

        output.append(len(value_counts))
    
    if options is None or options.get('Value lengths', False):

        if value_counts is None:
            value_counts = series.value_counts()

        if len(value_counts) > 0:
            # Output the MIN/MAX value lengths    
            # This is more memory and CPU efficient than using the pandas apply() function
            ##col_series = series.dropna()
            ##min_length = len(str(col_series.values[0]))
            min_length = len(str(value_counts.index[0]))
            max_length = min_length

            for ind, val in value_counts.items():
                value_length = len(str(ind))
                min_length = min(min_length, value_length)
                max_length = max(max_length, value_length)
    
            output.append(','.join([str(x) for x in [min_length, max_length]]))
        else:
            output.append('')
    
    if options is None or options.get('Character pattern', False):
        # Omitted because _get_column_patterns() does not handle some symbols correctly

        if value_counts is None:
            value_counts = series.value_counts()

        # Only output character pattern if the column is of type 'object' and has at least one value
        if series.dtype == 'object' and len(value_counts) > 0:
            try:
                # Output the patterns that are in the series. \d is used instead of [0-9]
                regex_pat_list = [r' ', re.compile(r'[a-z]', flags=re.IGNORECASE), re.compile(r'\d'), re.compile(r'\.'), r'\-', r'/']
                pattern_keys = [' ', 'a', '0', '.', '-', '/']
                # Replace a-z0-9 by \w
                regex_pat_other = re.compile(r'[^ \w\.\-/]', flags=re.IGNORECASE)
                patterns = _get_column_patterns(value_counts, regex_pat_list, pattern_keys, regex_pat_other)
                output.append(''.join(patterns))
            except (SystemExit, KeyboardInterrupt):
                logger.warning("%s KeyboardInterrupt" %(function_name))
                raise
            except Exception as e:
                logger.exception('%s %s. Unexpected exception: %s' %(function_name, type(e).__name__, getattr(e, 'message', str(e))))
                traceback.print_exc(file=sys.stdout)
                raise
        
        else:
            output.append('')

    if options is None or options.get('Numeric percentiles', False):
        
        if np.issubdtype(series.dtype, np.number) and not np.issubdtype(series.dtype, np.datetime64):
            qq = series.dropna().quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values.tolist()
            output.append(','.join([str(x) for x in qq]))
        else:
            output.append('')
    
    if options is None or options.get('Datetime percentiles', False):
        
        if np.issubdtype(series.dtype, np.datetime64):
            # Get quantiles and convert them to strings
            qq = np.datetime_as_string(series.dropna().quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values)
            # Convert the quantiles into comma-separated strings
            output.append(','.join([str(x) for x in qq]))
        else:
            #output = output + ['', '', '', '', '']
            output.append('')

    return output


def _get_column_patterns(value_counts, regex_pat_list, pattern_keys, regex_pat_other):
    """
    Finds whether each of the supplied regex patterns occurs anywhere in a series
    
    :param value_counts: Value counts stored as a series (the index contains the values)
    :param regex_pat_list: A list of compiled regular expressions (e.g., [r'[a-z]', r['0-9']])
    :param pattern_keys: A list of single-character strings that are used to indicate the corresponding items in regex_pat_list  (e.g., ['a', '0']). The regex_pat_list and pattern_keys lists must be the same length.
    :param regex_pat_other: A regular expression that tests for the occurrence of anything not covered by any entry in the regex_pat_list (e.g., r'[^a-z0-9]')
    
    :return A list of the patterns that occur in the series, starting with the pattern_keys for patterns that occur in the series, and then followed by any other characters that matched regex_pat_other.
    """
    function_name = '_get_column_patterns()'
    logger = logging.getLogger('vizdataquality')
    logger.debug("%s" %(function_name))

    does_contain = [False for l1 in range(len(regex_pat_list))]
    pattern_dict = {}
        
    for ind, val in value_counts.items():
        # Check for each of the patterns in regex_pat_list
        for l2 in range(len(does_contain)):
            # Convert the value to a string because datasets loaded from Excel can have mixed data types in a column (e.g., string and currency)
            if re.search(regex_pat_list[l2], str(ind)):
                does_contain[l2] = True
            
        # Check for the regex_pat_other patterns
        other_list = re.findall(regex_pat_other, str(ind))
        # Add each character that is in other_list to pattern_dict
        for l2 in range(len(other_list)):
            
            if other_list[l2] not in pattern_dict:
                pattern_dict[other_list[l2]] = True

    # Add the patterns that occurred to the list that will be returned
    pattern_list = []
    
    for l1 in range(len(does_contain)):
        
        if does_contain[l1] == True:
            pattern_list.append(pattern_keys[l1])
            
    for item in pattern_dict:
        pattern_list.append(item)
            
    return pattern_list
