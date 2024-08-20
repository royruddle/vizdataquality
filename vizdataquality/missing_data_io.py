# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:47:41 2024

Functions for reading/writing data structures used by the missing data functionality

@author: Roy Ruddle, University of Leeds
"""
import os
import pandas as pd

from vizdataquality import explanation_graph as eg

# =============================================================================
# Functions to read/write the set and intersection data structures
# =============================================================================
def write_set_data(folder='', stem='', overwrite=False, variables=None, intersections=None, degree=None, cardinality=None, records=None, **kwargs):
    """
    A convenience function to write the set and intersection data structures to files. 

    Parameters
    ----------
    folder : string, optional
        Folder to write the files to. The default is ''.
    stem : string, optional
        A stem for each filename. The default is ''.
    overwrite : boolean, optional
        False (do not overwrite the files) or True (overwrite files if they exist). The default is False.
    variables : series, optional
        The number of missing values in each column. The default is None.
    intersections : dataframe, optional
        The combinations of missing values that occur in the dataset.
        Columns are True (missing value) or False (value present).
        Index is the 'intersection_id'
        The default is None.
    degree : series, optional
        The degree of each intersection. The default is None.
    cardinality : series, optional
        The cardinality of each intersection. The default is None.
    records : dataframe, optional
        The 'intersection_id' (Index) of each record. The default is None.
    kwargs :
        Keyword arguments for series/dataframe.to_csv()

    Returns
    -------
    None.

    """
    names = ['_variables', '_intersections', '_degree', '_cardinality', '_records']
    ext = '.csv'
    outputs = [variables, intersections, degree, cardinality, records]
    error = False
    
    if not overwrite:
        # Check that none of the files already exist
        for l1 in range(len(names)):
            
            if outputs[l1] is not None:
                filename = os.path.join(folder, stem + names[l1] + ext)
    
                if os.path.isfile(filename):
                    print('*** ERROR *** missing_data_utils.write_set_data(). File already exists:')
                    print(filename)
                    error = True

    # Write the files
    if not error:
        for l1 in range(len(names)):
            
            if outputs[l1] is not None:
                filename = os.path.join(folder, stem + names[l1] + ext)
                outputs[l1].to_csv(filename, **kwargs)


def read_set_data(folder='', stem='', variables=True, intersections=True, degree=True, cardinality=True, records=True, **kwargs):
    """
    A convenience function to read the set and intersection data structures from files. 

    Parameters
    ----------
    folder : string, optional
        Folder to write the files to. The default is ''.
    stem : string, optional
        A stem for each filename. The default is ''.
    variables : boolean, optional
        Return the number of missing values in each column as a series. The default is True.
    intersections : boolean, optional
        Return the combinations of missing values that occur in the dataset as a dataframe. The default is True.
    degree : boolean, optional
        Return the degree of each intersection as a series. The default is True.
    cardinality : boolean, optional
        Return the cardinality of each intersection as a series. The default is True.
    records : boolean, optional
        Return the 'intersection_id' (Index) of each record as a dataframe. The default is True.
    kwargs :
        Keyword arguments for series/dataframe.read_csv()

    Returns
    -------
    list
        The items are the series/dataframes that were requested, in this order: variables, intersections, degree, cardinality, records

    """
    names = ['_variables', '_intersections', '_degree', '_cardinality', '_records']
    ext = '.csv'
    inputs = [variables, intersections, degree, cardinality, records]
    data = []

    # Read the files
    for l1 in range(len(names)):
        
        if inputs[l1]:
            filename = os.path.join(folder, stem + names[l1] + ext)
            df = pd.read_csv(filename, index_col=[0], **kwargs)
            
            if l1 == 1 or l1 == 4:
                # Return a dataframe
                data.append(df)
            else:
                # Return a series
                data.append(df[df.columns[0]])

    return data


# =============================================================================
# Functions to read/write the purity data
# =============================================================================
def write_purity_data(data, folder='', filename='purities.csv', overwrite=False, **kwargs):
    """
    A convenience function to write the purities dataframe to a file. 

    Parameters
    ----------
    data : dataframe
        The purities for pairs of variables, output by missing_data_functions.get_missiness_structure()
    folder : string, optional
        Folder to write the file to. The default is ''.
    filename : string, optional
        The output filename. The default is 'purities.csv'.
    overwrite : boolean, optional
        False (do not overwrite the file) or True (overwrite file if it exists). The default is False.
    kwargs :
        Keyword arguments for series/dataframe.to_csv()

    Returns
    -------
    None.

    """
    error = False
    fname = os.path.join(folder, filename)
    
    if not overwrite:
        # Check that the file does not exist

        if os.path.isfile(fname):
            print('*** ERROR *** missing_data_utils.write_purity_data(). File already exists:')
            print(fname)
            error = True

    # Write the files
    if not error:
        data.to_csv(fname, **kwargs)


def read_purity_data(folder='', filename='purities.csv', **kwargs):
    """
    A convenience function to read the purities dataframe from a file. 

    Parameters
    ----------
    folder : string, optional
        Folder to write the files to. The default is ''.
    filename : string, optional
        The output filename. The default is 'purities.csv'.
    kwargs :
        Keyword arguments for series/dataframe.read_csv()

    Returns
    -------
    Dataframe
        The purities, as would be output by missing_data_functions.get_missiness_structure()

    """
    fname = os.path.join(folder, filename)
    data = pd.read_csv(fname, index_col=[0], **kwargs)

    return data


# =============================================================================
# Functions to read/write the explanation graph
# =============================================================================
def write_graph_data(graph, folder='', stem='', overwrite=False, **kwargs):
    """
    A convenience function to write the explanation graph to files. 

    Parameters
    ----------
    graph : Explanation_Graph
        An explanation graph
    folder : string, optional
        Folder to write the files to. The default is ''.
    stem : string, optional
        A stem for each filename. The default is ''.
    overwrite : boolean, optional
        False (do not overwrite the files) or True (overwrite files if they exist). The default is False.
    kwargs :
        Keyword arguments for series/dataframe.to_csv()

    Returns
    -------
    None.

    """
    num_nodes = len(graph._nodelist)
    #
    # Create a series that stores the graph's attributes
    #
    graph_attributes = pd.Series(data=[num_nodes, graph._criteria], index=['num_nodes', '_criteria'])
    #
    # Create a dataframe that stores the nodes
    #
    columns = ['_id', '_attributes', '_caption', '_description', '_columns', '_intersections', '_children', '_calculated_width', '_width',
               '_x', '_y']#, '_rendering']
    data = {}
    
    for col in columns:
        array = [None] * num_nodes
        
        for l1 in range(len(graph._nodelist)):
            node = graph._nodelist[l1]
            
            if col == '_id':
                array[l1] = node._id
            elif col == '_attributes':
                array[l1] = node._attributes
            elif col == '_caption':
                array[l1] = node._caption
            elif col == '_description':
                array[l1] = node._description
            elif col == '_columns' and node._columns is not None:
                array[l1] = list(node._columns)
            elif col == '_intersections' and node._intersections is not None:
                array[l1] = list(node._intersections)
            elif col == '_children':
                array[l1] = [nn._id for nn in node._children]
            elif col == '_calculated_width':
                array[l1] = node._calculated_width
            elif col == '_width':
                array[l1] = node._width
            elif col == '_x':
                array[l1] = node._x
            elif col == '_y':
                array[l1] = node._y
            #elif col == '_rendering':
            #    array[l1] = node._rendering
        
        # Add this column to the data
        data[col] = array
    
    # Create the dataframe
    graph_nodes = pd.DataFrame.from_dict(data)
    
    names = ['_graph_intersections', '_graph_intersection_cardinality', '_graph', '_graph_nodes']
    ext = '.csv'
    outputs = [graph._intersection_id_to_columns, graph._intersection_cardinality, graph_attributes, graph_nodes]
    error = False
    
    if not overwrite:
        # Check that none of the files already exist
        for l1 in range(len(names)):
            filename = os.path.join(folder, stem + names[l1] + ext)

            if os.path.isfile(filename):
                print('*** ERROR *** missing_data_utils.write_set_data(). File already exists:')
                print(filename)
                error = True

    # Write the files
    if not error:
        for l1 in range(len(names)):
            filename = os.path.join(folder, stem + names[l1] + ext)
            outputs[l1].to_csv(filename, **kwargs)


def read_graph_data(folder='', stem='', **kwargs):
    """
    A convenience function to read the explanation graph from files. 

    Parameters
    ----------
    folder : string, optional
        Folder to write the files to. The default is ''.
    stem : string, optional
        A stem for each filename. The default is ''.
    kwargs :
        Keyword arguments for series/dataframe.read_csv()

    Returns
    -------
    Explanation_Graph
        An explanation graph

    """
    #
    # Read the files (graph_intersection_cardinality and graph are series; the other two are dataframes)
    #
    names = ['_graph_intersections', '_graph_intersection_cardinality', '_graph', '_graph_nodes']
    ext = '.csv'
    dfs = []
    
    for l1 in range(len(names)):
        filename = os.path.join(folder, stem + names[l1] + ext)
        df = pd.read_csv(filename, index_col=[0], **kwargs)
        
        if l1 == 0 or l1 == 3:
            # Return a dataframe
            dfs.append(df)
        else:
            # Return a series
            dfs.append(df[df.columns[0]])
            
    #
    # Create the graph
    #
    graph = eg.Explanation_Graph(dfs[0], dfs[1], import_attributes=dfs[2], import_nodes=dfs[3])

    return graph
