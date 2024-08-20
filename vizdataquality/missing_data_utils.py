# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 12:15:42 2024

Utility functions for analysing missing data.

Internal functions are prefixed by '_'. The external functions extract information from setvis data patterns.

@author: Roy Ruddle, University of Leeds
"""

# =============================================================================
# Constants
# =============================================================================
DISJOINT = 'disjoint'
BLOCK = 'block'
MONOTONE = 'monotone'
PATTERNS = [DISJOINT, BLOCK, MONOTONE]

def _get_pattern_colournum(struct, purity):
    """
    Get the colour number for a pattern and its purity: 0 (impure DISJOINT), 1 (pure DISJOINT), etc.

    Parameters
    ----------
    struct : str
        One of the following constants: DISJOINT, BLOCK, MONOTONE.

    purity : float
        Purity of the pattern (0.0 - 1.0), where 1.0 indicates the pattern is completely pure.

    Returns
    -------
    int
        The colour number (0 - 5) or None if struct was invalid.

    """
    colournum = None
    
    if struct == DISJOINT:
        colournum = 0
    elif struct == BLOCK:
        colournum = 2
    elif struct == MONOTONE:
        colournum = 4
        
    try:
        colournum += 0.0 if purity < 1.0 else 1.0
    except:
        pass
    
    return colournum


def _get_pattern_legend():
    """
    Return the pattern/purity colour legend for a purity heatmap

    Returns
    -------
    list
        List of the items in the pattern/purity colour legend.

    """
    legend = []
    
    for ss in [DISJOINT, BLOCK, MONOTONE]:
        for pp in [' - not pure', ' - pure']:
            legend.append(ss.capitalize() + pp)
            
    return legend


# =============================================================================
# Functions for extracting information from the set and intersection data structures
# =============================================================================
def get_degree(intersection_degree, degree, intersection_id_to_columns):
    """
    Get the column names and intersection IDs for all intersections with the specified degree.

    Parameters
    ----------
    intersection_degree : series
        The number of columns in each intersection (i.e., its degree).
    degree : int
        The degree.
    intersection_id_to_columns : dataframe
        The combinations of missing values that occur in the dataset.
        Columns are True (missing value) or False (value present).
        Index is the 'intersection_id'

    Returns
    -------
    list, list
        Lists of the column names and intersection IDs

    """
    int_list = intersection_degree[intersection_degree == degree].index.tolist()
    clist = intersection_id_to_columns.loc[int_list].any()
    col_list = clist[clist == True].index.tolist()  
    
    return col_list, int_list


def get_pattern_intersections(intersection_id_to_columns, pattern, pure, columns_A, columns_B=None, num_missing=None):
    """
    Return the intersections that have a pure (True) or impure (False) pattern in the specified columns.

    Parameters
    ----------
    intersection_id_to_columns : dataframe
        The combinations of missing values that occur in the dataset.
        Columns are True (missing value) or False (value present).
        Index is the 'intersection_id'
    pattern : str
        'monotone' or 'disjoint'.
    pure : boolean
        True or False.
    columns_A : str or list
        One column or a list of columns. For 'monotone' it must be a list (i.e., 2+ columns).
    columns_B : str or list, optional
        One column or a list of columns. Mandatory for 'planned' but not used for 'monotone'. The default is None.
    num_missing : series, optional
        Series containing the number of missing values in each column. Mandatory for 'monotone' but not used for 'planned'. The default is None.

    Returns
    -------
    int_list : list
        A list of intersections IDs.

    """
    int_list = []
    # Either a single column or a list of columns may be supplied
    if isinstance(columns_A, list):
        ca = columns_A
    else:
        ca = [columns_A]
    
    if isinstance(columns_B, list):
        cb = columns_B
    else:
        cb = [columns_B]
        
    if pattern == DISJOINT:
        # Get the intersections that have missing data for any of the A and B columns, respectively
        intA = intersection_id_to_columns[ca].any(axis=1)
        setA = set(intA[intA == True].index.tolist())
        intB = intersection_id_to_columns[cb].any(axis=1)
        setB = set(intB[intB == True].index.tolist())
        # Get the intersections that are in A and B
        A_and_B = setA.intersection(setB)
        
        if pure:
            int_list = list(setA.union(setB).difference(A_and_B))
        else:
            int_list = list(A_and_B)
        
    elif pattern == MONOTONE:
        # Order the list of columns in ascending number of missing values
        ca2 = num_missing.loc[ca].sort_values().index.tolist()
        
        # A monotone pattern must involve at least two columns
        if len(ca2) > 1:
            # Store a list of the intersections that include any of the columns
            ss = intersection_id_to_columns[ca2].any(axis=1)
            all_ints = ss[ss == True].index.tolist()
            #print(pattern, 'all_ints', len(all_ints))
            
            for l1 in range(len(ca2)-1):
                # Determine the intersections that: (a) are still in the all_ints list, and (b) where columns l1 onward are all missing
                ss = intersection_id_to_columns[ca2[l1:]].iloc[all_ints].all(axis=1)
                iall = ss[ss == True].index.tolist()
                # Get the intersections of column l1 that are still in the all_ints list
                itmp = intersection_id_to_columns.iloc[all_ints]
                setA = set(itmp[itmp[ca2[l1]] == True].index.tolist())
                
                if pure:
                    # Add the intersections where columns l1 onward are all missing
                    int_list += iall
                else:
                    # Add the intersections where some subsequent columns are not included in the column l1 intersections
                    int_list += list(setA.difference(set(iall)))
            
                # Remove the column l1 intersections from the list
                all_ints = list(set(all_ints).difference(setA))
            
            if pure:
                # Add all_ints (the remaining intersections, which are only in the last column) to the list
                int_list += all_ints
                
    return int_list
