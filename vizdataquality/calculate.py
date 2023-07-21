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

import pandas as pd
import numpy as np
import sys

import re
import datetime

import logging
import traceback


##############################################################################    
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
    df_output : DataFrame
        The descriptive statistics (rows are the supplied dataframe's variables; columsn are different descriptive statistics).

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

    if options is None or options.get('Missing values', False):
        cols.append('Number of missing values')

    if options is None or options.get('Unique values', False):
        cols.append('Number of unique values')
    
    if options is None or options.get('Value lengths', False):
        for item in ['(min)', '(max)']:
            cols.append('Value lengths ' + item)
    
    if False and (options is None or options.get('Character pattern', False)):
        # Omitted because _get_column_patterns() does not handle some symbols correctly
        cols.append('Patterns')
    
    if options is None or options.get('Numeric percentiles', False):
        for item in ['(min)', '(25th)', '(median)', '(75th)', '(max)']:
            cols.append('Numeric ' + item)
    
    if options is None or options.get('Datetime percentiles', False):
        for item in ['(min)', '(25th)', '(median)', '(75th)', '(max)']:
            cols.append('Datetime ' + item)
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

    et = datetime.datetime.now()
    logger.info("%s,time (s),%s" %(function_name, str((et-st).total_seconds())))
    
    return df_output


##############################################################################
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
        
        # The example is the first value
        output.append(value_counts.index[0])
    
    if options is None or options.get('Missing values', False):
        output.append(series.isnull().sum())

    
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
            ##for l1 in range(1, len(col_series)):
            for ind, val in value_counts.items():
                ##value_length = len(str(col_series.values[l1]))
                value_length = len(str(ind))
                min_length = min(min_length, value_length)
                max_length = max(max_length, value_length)
    
            output = output + [min_length, max_length]
        else:
            output = output + ['', '']
    
    if False and (options is None or options.get('Character pattern', False)):
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
            output = output + series.dropna().quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values.tolist()
        else:
            output = output + ['', '', '', '', '']
    
    if options is None or options.get('Datetime percentiles', False):
        
        if np.issubdtype(series.dtype, np.datetime64):
            quantiles = series.dropna().quantile([0.0, 0.25, 0.5, 0.75, 1.0])#.values.tolist()
                
            for ind, val in quantiles.items():
                output.append(str(val))
        else:
            output = output + ['', '', '', '', '']

    return output


##############################################################################    
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
