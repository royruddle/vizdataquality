# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:54:11 2022

Functions for analysing missing data. Internal functions are prefixed by '_'.

Some of the functions have been derived from setvis (https://pypi.org/project/setvis/),
which was developed by the same authors as vizdataquality. The derived
functions are included here to avoid issues with dependencies.
vizdataquality and setvis use the same Apache License 2.0 (see LICENCE notice).

@author: Roy Ruddle, University of Leeds
"""

import csv
import numpy as np
import pandas as pd

from vizdataquality import missing_data_utils as mdu

# =============================================================================
# Functions for calculating the purity of missingness patterns
# =============================================================================
def _get_missiness_pattern_two_columns(intersection_columns, intersection_cardinality, patterns, threshold):
    """
    Determine which patterns of missing data occur and their purity, for a pair of variables.

    Parameters
    ----------
    intersection_columns : dataframe
        Only containing the two columns, and containing the combinations of missing values that occur in the dataset.
        Columns are True (missing value) or False (value present).
        Index is the 'intersection_id'
    intersection_cardinality : series
        The number of times each intersection occurs.
    patterns : list
        The patterns to look for or None (all those in mdu.PATTERNS).
    threshold : float
        A purity threshold, below which a pattern will not be returned.

    Returns
    -------
    DataFrame
        One row for each type of pattern/pair of columns, storing ['Pattern', 'Purity', 'Column1', 'Column2'].

    """
    # Storing intersections for each pair of columns could require a lot of memory
    store_intersections = False
    
    # Create dataframe of set intersections: True (False) indicates a variable is missing (present) for that intersection.
    column1 = intersection_columns.columns[0]
    column2 = intersection_columns.columns[1]
    # Drop rows where both variables are False (i.e., no missing values)
    df_check2 = intersection_columns[intersection_columns.any(axis=1)]
    # List to store dataframe of intersections where [both columns missing, only column1 missing, only column2 missing]
    df_combinations = [None] * 3
    # Select intersections where both variables are missing
    df_combinations[0] = df_check2[df_check2.all(axis=1)]
    # Select the intersections where only one of the variables is missing
    df_check3 = df_check2[~df_check2.all(axis=1)]
    # Select intersections where only column1 is missing
    df_combinations[1] = df_check3[df_check3[column1]]
    # Select intersections where only column2 is missing
    df_combinations[2] = df_check3[df_check3[column2]]
    
    # Calculate the number of times [both columns missing, only column1 missing, only column2 missing]
    cardinality = [0] * len(df_combinations)
    
    for l1 in range(len(df_combinations)):
        try:
            cardinality[l1] = intersection_cardinality.loc[df_combinations[l1].index].sum()
        except:
            pass
        
    data = []
    
    if patterns is None:
        structs = mdu.PATTERNS
    elif type(patterns) is list:
        structs = patterns
    else:
        structs = [patterns]
        
    # min, mean or overall
    PURITY_METHOD = 'overall'
        
    for ss in structs:
        purity = None
        # The columns missing together in at least one intersection
        if ss == mdu.DISJOINT:
            if cardinality[0] == 0:
                # The columns are never missing together
                purity = 1.0
            elif (cardinality[1] + cardinality[2]) == 0:
                # If one column is missing then the other is also missing
                purity = 0.0
            else:
                # The columns are sometimes missing separately
                if PURITY_METHOD == 'overall':
                    purity = float(cardinality[1] + cardinality[2]) / float(sum(cardinality))
                else:
                    pp = [None, None]
                    
                    for l1 in range(2):
                        pp[l1] = float(cardinality[l1 + 1]) / float(cardinality[0] + cardinality[l1 + 1])
                    
                    if PURITY_METHOD == 'min':
                        purity = min(pp)
                    elif PURITY_METHOD == 'mean':
                        purity = 0.5 * sum(pp)
                
        elif ss == mdu.BLOCK:
            if cardinality[0] == 0:
                # The columns are never missing together
                purity = 0.0
            elif (cardinality[1] + cardinality[2]) == 0:
                # If one column is missing then the other is also missing
                purity = 1.0
            else:
                # The columns are sometimes missing separately
                if PURITY_METHOD == 'overall':
                    purity = float(cardinality[0]) / float(sum(cardinality))
                else:
                    pp = [None, None]
                    
                    for l1 in range(2):
                        pp[l1] = float(cardinality[0]) / float(cardinality[0] + cardinality[l1 + 1])
                    
                    if PURITY_METHOD == 'min':
                        purity = min(pp)
                    elif PURITY_METHOD == 'mean':
                        purity = 0.5 * sum(pp)
        elif ss == mdu.MONOTONE:
            if cardinality[0] == 0:
                # The columns are never missing together
                purity = 0.0
            elif (cardinality[1] + cardinality[2]) == 0:
                # The columns are never missing separately (i.e., a block pattern)
                purity = float(cardinality[0] - 1) / float(sum(cardinality))
            elif min(cardinality[1], cardinality[2]) == 0 and max(cardinality[1], cardinality[2]) > 0:
                # Only one column is sometimes missing separately
                purity = 1.0
            else:
                # The columns are sometimes missing together and each column is sometimes missing separately
                if PURITY_METHOD == 'overall':
                    purity = float(cardinality[0] + max(cardinality[1], cardinality[2])) / float(sum(cardinality))
                else:
                    pp = [None, None]
                    pp[0] = float(cardinality[0]) / float(cardinality[0] + min(cardinality[1], cardinality[2]))
                    pp[1] = float(cardinality[0] + max(cardinality[1], cardinality[2])) / float(sum(cardinality))
                    
                    if PURITY_METHOD == 'min':
                        purity = min(pp)
                    elif PURITY_METHOD == 'mean':
                        purity = 0.5 * sum(pp)
            
        try:
            if purity >= threshold:
                # Store the pattern
                if cardinality[1] <= cardinality[2]:
                    col1 = column1
                    col2 = column2
                else:
                    col1 = column2
                    col2 = column1
                    
                if store_intersections:
                    data.append([ss, purity, col1, col2, ' '.join(map(str, df_check2.index.tolist()))])
                else:
                    data.append([ss, purity, col1, col2])
                    
        except:
            # purity was None
            pass

    if store_intersections:
        dfcols = ['Pattern', 'Purity', 'Column1', 'Column2', 'Intersection_list']
    else:
        dfcols = ['Pattern', 'Purity', 'Column1', 'Column2']
        
    if len(data) > 0:
        df_patterns = pd.DataFrame(columns=dfcols, data=data)
    else:
        df_patterns = pd.DataFrame(columns=dfcols)
    
    return df_patterns


def get_missiness_pattern(num_missing, intersection_columns, intersection_cardinality, patterns=None, threshold=0.0):
    """
    Determine which patterns of missing data occur and their purity, for columns that have missing values.

    Parameters
    ----------
    num_missing : series
        Series containing the number of missing values in each column.
    intersection_columns : dataframe
        The combinations of missing values that occur in the dataset.
        Columns are True (missing value) or False (value present).
        Index is the 'intersection_id'
    intersection_cardinality : series
        The number of times each intersection occurs.
    patterns : list, optional
        The patterns to look for. The default is None (look for all the mdu.PATTERNS).
    threshold : float, optional
        A purity threshold, below which a pattern will not be returned.The default is 0.0.

    Returns
    -------
    DataFrame
        One row for each type of pattern/pair of columns, storing ['Pattern', 'Purity', 'Column1', 'Column2', 'Intersection_list'].

    """
    df_patterns = None
    
    # Loop over ever pair of columns
    for l1 in range(intersection_columns.shape[1]-1):
        col1 = intersection_columns.columns[l1]
        
        if num_missing.loc[col1] > 0:
            # At least one value is missing from this column
            for l2 in range(l1+1, intersection_columns.shape[1]):
                col2 = intersection_columns.columns[l2]
        
                if num_missing.loc[col2] > 0:
                    # At least one value is missing from this column
                    df_column_pair_pattern = _get_missiness_pattern_two_columns(intersection_columns[[col1, col2]], intersection_cardinality, patterns, threshold)
                    
                    try:
                        df_patterns = pd.concat([df_patterns, df_column_pair_pattern], ignore_index=True)
                    except:
                        pass
                        df_patterns = df_column_pair_pattern
                
    return df_patterns


def select_missiness_patterns(df_patterns, columns=None, patterns=None, threshold=0.0):
    """
    Return a DataFrame of purity patterns that contains rows that satisfy the selection criteria.

    Parameters
    ----------
    df_patterns : DataFrame
        DataFrame from get_missiness_pattern(). Columns are ['Pattern', 'Purity', 'Column1', 'Column2', 'Intersection_list'].
    columns : list, optional
        List of columns. Rows are selected if both 'Column1' and 'Column2' are in the list. Default is None (select all columns).
    patterns : list, optional
        The patterns to look for. The default is None (select all patterns in df_patterns).
    threshold : float, optional
        A purity threshold, below which a pattern will not be returned.The default is 0.0.

    Returns
    -------
    DataFrame
        One row for each type of pattern/pair of columns, storing ['Pattern', 'Purity', 'Column1', 'Column2', 'Intersection_list'].

    """
    # By default select all the columns
    if columns is None:
        cols = list(set(df_patterns['Column1'].unique().tolist() + df_patterns['Column2'].unique().tolist()))
    else:
        cols = columns
        
    # By default select all patterns
    if patterns is None:
        structs = df_patterns['Pattern'].unique().tolist()
    elif type(patterns) is list:
        structs = patterns
    else:
        structs = [patterns]

    df_struct2 = df_patterns[df_patterns['Purity'] >= threshold].copy()
    include = [False] * len(df_struct2)
    l1 = 0

    for index, row in df_struct2.iterrows():
        if row['Column1'] in cols and row['Column2'] in cols and row['Pattern'] in structs:
            include[l1] = True
            
        l1 += 1

    df_struct2['include'] = include
    
    return df_struct2[df_struct2['include']].sort_values(['Column1', 'Column2'])


# =============================================================================
# The following functions have been derived from setvis code.
# The functions calculate set intersections (combinations of missing values)
# and cardinalities (number of records with each intersection)
# =============================================================================
def _get_intersections(columns, dict_displayedMissingCombs, record_intersection_id):
    """
    Convert a dictionary (set intersections) and NumPy array (intersection cardinalities) to a dataframe and series, respectively.

    Parameters
    ----------
    columns : list
        Names of the dataset's columns.
    dict_displayedMissingCombs : dict
        The combinations of missing values that occur in the dataset.
    record_intersection_id : NumPy array
        The intersection ID of each record in the dataset.

    Returns
    -------
    intersection_id_to_columns : dataframe
        Stores the combinations of missing values that occur in the dataset.
        Columns are True (missing value) or False (value present).
        Index is the 'intersection_id'
    intersection_id_to_records : series
        The 'intersection_id' (Index) of each record.

    """
    # Create a dataframe with the same columns as the original dataset
    intersection_id_to_columns = pd.DataFrame(columns=columns)
    
    # Add each intersection as a row, with the ID as the location
    for key, value in dict_displayedMissingCombs.items():
        intersection_id_to_columns.loc[value] = list(key)
    
    # Replace t/f with True/False (column is missing/present in that intersection)
    intersection_id_to_columns.replace({'f': False, 't': True}, inplace=True)
    intersection_id_to_columns.rename_axis('intersection_id', inplace=True)
    
    intersection_id_to_records = pd.Series(record_intersection_id).rename_axis("_record_id").reset_index().set_index(0).rename_axis('intersection_id')

    return intersection_id_to_columns, intersection_id_to_records


def get_intersections_from_dataframe(df):
    """
    Get the combinations of missing values that occur in a dataset that is stored as a dataframe.

    Parameters
    ----------
    df : dataframe
        The dataset.

    Returns
    -------
    num_missing : series
        The number of missing values in each column
    intersection_id_to_columns : dataframe
        The combinations of missing values that occur in the dataset.
        Columns are True (missing value) or False (value present).
        Index is the 'intersection_id'
    intersection_id_to_records : series
        The 'intersection_id' (Index) of each record.

    """
    columns, num_missing, dict_displayedMissingCombs, record_intersection_id = _compute_missingness_from_dataframe(df)
    intersection_id_to_columns, intersection_id_to_records = _get_intersections(columns, dict_displayedMissingCombs, record_intersection_id)
    
    return num_missing, intersection_id_to_columns, intersection_id_to_records


def get_intersections_from_file(dataset_filename, verbose=False, **kwargs):
    """
    Get the combinations of missing values that occur in a dataset that is stored in a text data file.
    If kwargs specifies the chunksize then the file is read iteratively, to handle large datafiles in a memory-efficient manner.
    If nrows is also specified then the chunked reading is more efficient because the record_intersection_id array is created the correct size rather than having to be resized during chunking.

    Parameters
    ----------
    dataset_filename : string
        Name of the datafile.
    verbose : boolean
        True (print information after each chunk of the file is read) or False (silent). The default is False.
    kwargs : dictionary
        Keyword arguments for Pandas read_csv()

    Returns
    -------
    num_missing : series
        The number of missing values in each column
    intersection_id_to_columns : dataframe
        The combinations of missing values that occur in the dataset.
        Columns are True (missing value) or False (value present).
        Index is the 'intersection_id'
    intersection_id_to_records : series
        The 'intersection_id' (Index) of each record.

    """
    chunksize = kwargs.get('chunksize', None)

    # Create empty dictionary to store the combinations of missing values
    dict_displayedMissingCombs = {}
    row_num = 0
    
    if chunksize is None:
        df = pd.read_csv(dataset_filename, **kwargs)
        columns, num_missing, dict_displayedMissingCombs, record_intersection_id = _compute_missingness_from_dataframe(df)
    else:
        df_nmissing = None
        record_intersection_id= None
        nrows = kwargs.get('nrows', None)
        l1 = 0
        
        for chunk in pd.read_csv(dataset_filename, **kwargs):

            if record_intersection_id is None:
                # The first chunk. Array size is nrows (if that is specified) or chunksize
                size = chunksize if nrows is None else nrows
                record_intersection_id = np.empty(size, np.int8)
                columns = list(chunk.columns)
            elif nrows is None and record_intersection_id.size < row_num + chunksize:
                # Subsequent chunks. Increase array so there's space for each record in this chunk
                record_intersection_id.resize(row_num + chunksize)
                
            nmissing = _compute_missingness(chunk, dict_displayedMissingCombs, record_intersection_id)
            
            try:
                df_nmissing[(str(l1))] = nmissing
            except:
                pass
                df_nmissing = pd.DataFrame(nmissing)
                
            l1 = l1 + 1
            row_num = row_num + len(chunk)
            
            if verbose:
                print('missing_data_functions.py', 'get_intersections_from_file()')
                print('   chunk =', l1, 'number of intersections =', len(dict_displayedMissingCombs), 'record array size =', len(record_intersection_id))
                print('record_intersection_id', len(record_intersection_id))
            
        # Make sure this array is the correct size (if the last chunk was smaller than chunksize then the array will be too long)
        record_intersection_id.resize(row_num)

        if verbose:
            print('missing_data_functions.py', 'get_intersections_from_file()')
            print('number of intersections =', len(dict_displayedMissingCombs), 'number of records =', len(record_intersection_id))
            
        # Calculate the total number of values that were missing in each variable
        num_missing = df_nmissing.sum(axis=1)
    #
    # Create membership dataframes
    #
    intersection_id_to_columns, intersection_id_to_records = _get_intersections(columns, dict_displayedMissingCombs, record_intersection_id)

    return num_missing, intersection_id_to_columns, intersection_id_to_records


def _compute_missingness(df, dict_displayedMissingCombs, record_intersection_id):
    """
    Get infomation about missing values in a dataset.
    Information is the number of missing values in each column, combinations of missing values, and combination:record mapping)

    Parameters
    ----------
    df : dataframe
        The dataset.
    dict_displayedMissingCombs : dict
        The combinations of missing values that occur in the dataset. Keys are combinations of the letters 'f' and 't'
        (one letter for each column, indicating that values are missing ('t') or present ('f') for that column).
        Updated on exit.
    record_intersection_id : NumPy array
        The intersection ID of each record in the dataset. Updated on exit.

    Returns
    -------
    series
        The number of missing values in each column

    """
    num_missing = df.isnull().sum()
    num_intersections = len(dict_displayedMissingCombs)
    
    for index, row in df.iterrows():

        if False:
            # Calculate the number of missing values in the row
            num_missing_values = row.isnull().sum()
            print(num_missing_values)
            # [False, True, True]
            print(row.isnull().tolist())
            # ['f', 't', 't']
            print(row.isnull().replace({False: 'f', True: 't'}).tolist())
            # 'ftt'
            print(''.join(row.isnull().replace({False: 'f', True: 't'}).tolist()))

        # Get a compact representation of the data that are missing in this row.
        # There is one letter per dataframe column, containing 't' (missing value) or f' (value present)
        row_tf = ''.join(row.isnull().replace({False: 'f', True: 't'}).tolist())

        try:
            # Get the ID of this intersection
            intersection_id = dict_displayedMissingCombs[row_tf]
        except:
            # Add this new intersection
            intersection_id = num_intersections
            dict_displayedMissingCombs[row_tf] = intersection_id
            num_intersections = num_intersections + 1

        record_intersection_id[index] = intersection_id
    
    return num_missing
    
    
def _compute_missingness_from_dataframe(df):
    """
    Get infomation about missing values in a dataset that is stored in a dataframe.
    Information is the number of missing values in each column, combinations of missing values, and combination:record mapping)

    Parameters
    ----------
    df : dataframe
        The dataset.

    Returns
    -------
    df.columns list
        Names of the dataset's columns.
    dict_displayedMissingCombs : dict
        The combinations of missing values that occur in the dataset. Keys are combinations of the letters 'f' and 't'
        (one letter for each column, indicating that values are missing ('t') or present ('f') for that column).
    record_intersection_id : NumPy array
        The intersection ID of each record in the dataset.


    Returns
    -------
    num_missing : series
        The number of missing values in each column
    intersection_id_to_columns : dataframe
        The combinations of missing values that occur in the dataset.
        Columns are True (missing value) or False (value present).
        Index is the 'intersection_id'
    intersection_id_to_records : series
        The 'intersection_id' (Index) of each record.
    """
    # Create empty dictionary to store the combinations of missing values
    dict_displayedMissingCombs = {}
    # The intersection ID of each record (NumPy array uses less memory than Python list)
    #record_intersection_id = [-1] * len(df)
    record_intersection_id = np.empty(len(df), np.int8)
    num_missing = _compute_missingness(df, dict_displayedMissingCombs, record_intersection_id)
    
    return df.columns, num_missing, dict_displayedMissingCombs, record_intersection_id


# =============================================================================
# Utility functions for intersections
# =============================================================================
def get_intersection_degree(intersection_id_to_columns):
    """
    Get the degree of each intersection (i.e., number of variables that are missing for that intersection).

    Parameters
    ----------
    intersection_id_to_columns : dataframe
        The combinations of missing values that occur in the dataset.
        Columns are True (missing value) or False (value present).
        Index is the 'intersection_id'

    Returns
    -------
    series
        The degree of each intersection. Index is the 'intersection_id'.

    """
    return intersection_id_to_columns.sum(axis=1)


def get_intersection_cardinality(intersection_id_to_records):
    """
    Get the cardinality of each intersection (i.e., number of records which are missing that combination of variables).

    Parameters
    ----------
    intersection_id_to_records : series
        The 'intersection_id' (Index) of each record.

    Returns
    -------
    series
        The cardinality of each intersection. Index is the 'intersection_id'.

    """
    return intersection_id_to_records.index.value_counts().sort_index()


# =============================================================================
# Functions to get data to plot an intersection heatmap
# =============================================================================
def get_intersection_heatmap_data(intersection_id_to_columns, intersection_cardinality, remove_complete_variables=True, remove_complete_records=True):
    """
    Get dataframe for plotting an intersection heatmap.

    Parameters
    ----------
    intersection_id_to_columns : dataframe
        The combinations of missing values that occur in the dataset.
        Columns are True (missing value) or False (value present).
        Index is the 'intersection_id'
    intersection_cardinality : series
        The number of times each intersection occurs. Index is the 'intersection_id'.
    remove_complete_variables : boolean
        True (remove variables that are complete from the intersection data) or False. The default is True.
    remove_complete_records : boolean
        True (remove recrds that are complete from the intersection data) or False. The default is True.


    Returns
    -------
    dataframe
        Heat map data (one row per intersection; value is cardinality of the intersection or np.nan (variable is not part of the intersection))
    """
    # This join adds a column containing the cardinality of each intersection
    count_intersections = intersection_id_to_columns.join(intersection_cardinality)
    count_intersections.rename(columns={count_intersections.columns[-1]: '_count'}, inplace=True)
    
    # From setvis intersection_heatmap_data()
    data = count_intersections.astype(int).mul(count_intersections['_count'], axis=0)
    #data = data.sort_values(by="_count", ascending=True)
    data = data.drop('_count', axis=1).replace(0,np.nan)
    
    if remove_complete_variables:
        # Drop columns for variables that are complete
        data.dropna(axis='columns', how='all', inplace=True)

    if remove_complete_records:
        # Drop the row that is the intersection for records that are complete
        data.dropna(axis=0, how='all', inplace=True)
    
    
    return data
