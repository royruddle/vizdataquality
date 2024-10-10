# -*- coding: utf-8 -*-
"""
Created on Tue May  2 09:07:30 2023

Functions for the Explanation_Graph and Explanation_Node classes. Internal functions are prefixed by '_'.
The links between nodes in the graph are generated automatically from columns and/or intersections that nodes have in common.
That captures relationships between the individual nodes that a user creates while investigating patterns of missing data.


@author: Roy Ruddle
"""

import pandas as pd
import numpy as np
import ast
import re

# Bitwise constants (check columns and/or intersections to find the parent(s) of a node)
CHECK_COLUMNS = 1
CHECK_INTERSECTIONS = 2

class Explanation_Graph:
    """
    An Explanation_Graph starts with a root node (representing the dataset), and comprises explanations (nodes) and the connections between them (links).
    When a node is added, the whole graph is traversed to find nodes that involve the same dataset columns or missing data intersections as the new node.
    The new node is then connected to each branch of the graph that have columns/intersections in common.
    If there are no such nodes then the new node is connected to the root.
    """
    DEBUG = False
    def __init__(self, intersection_id_to_columns, intersection_cardinality, criteria=CHECK_COLUMNS, import_attributes=None, import_nodes=None):
        """
        Initialise an Explanation_Graph

        Parameters
        ----------
        intersection_id_to_columns : dataframe
            A dataframe with as many rows as there are unique intersections (patterns of missingness), and a column for each column in the original dataframe.
            NB: This is one of primary Membership data structures in the setvis package (available from PyPI).
        intersection_cardinality : series
            The cardinality of each intersection. Index is the 'intersection_id'.
        criteria : int, optional
            Determine an explanation's parents by checking columns (CHECK_COLUMNS) and/or intersections (CHECK_INTERSECTIONS). Default is CHECK_COLUMNS).
        import_attributes : series
            The graph's attributes. Only used when a graph is being imported. Default is None.
        import_nodes : dataframe
            The graph's nodes. Only used when a graph is being imported. Default is None.

        Returns
        -------
        None.

        """
        if import_attributes is None and import_nodes is None:
            # Initialise a new graph.
            self._criteria = criteria
            # Dataframe with one row per intersection (the index) a column for each variable.
            # Values are 0 (column is not part of the intersection), -1 (it is but not yet in the graph) or 1 (in the graph; i.e., 'explained')
            self._intersection_id_to_columns = intersection_id_to_columns.copy().astype('int64').replace(1, -1)
            # The cardinality of each intersection. Index is the 'intersection_id'.
            self._intersection_cardinality = intersection_cardinality.copy()
            # Set the number of nodes to None. The root will become node zero.
            Explanation_Node.num_nodes = None
            # Create the root node
            self._root = Explanation_Node(caption='Root')
            self._nodelist = [self._root]
        else:
            # Import the graph
            self._criteria = import_attributes.loc['_criteria']
            self._intersection_id_to_columns = intersection_id_to_columns
            self._intersection_cardinality = intersection_cardinality

            Explanation_Node.num_nodes = None
            
            for l1 in range(len(import_nodes)):
                # Get this node's variables as a series
                node_variables = import_nodes[import_nodes['_id'] == l1].squeeze(axis=0)
                # Some of the variables are parameters of the node init function
                ndict = {}
                
                params = ['_caption', '_description']
                
                for pp in params:
                    ndict[pp] = None if pd.isnull(node_variables.loc[pp]) else node_variables.loc[pp]
                
                params = ['_columns', '_intersections']
                
                for pp in params:
                    vv = None if pd.isnull(node_variables.loc[pp]) else node_variables.loc[pp]
                    
                    if pp == '_columns':
                        # Columns is a list (e.g., ['col1', 'col2'])
                        ndict[pp] = vv if vv is None else re.split(', ', vv[1:-1].replace("'", ''))
                    elif pp == '_intersections':
                        # Intersections is a list of integers that have been read in as text (e.g., [1, 2])
                        ndict[pp] = vv if vv is None else list(map(int, re.split(', ', vv[1:-1])))
                
                # Get text and bbox attributes
                textattr = None
                bboxattr = None
                pp = '_attributes'
                
                try:
                    textattr = ast.literal_eval(node_variables.loc[pp])
                    
                    try:
                        bboxattr = textattr['bbox']
                        del textattr['bbox']
                    except:
                        pass
                except:
                    pass
                
                if l1 == 0:
                    # Create the root node
                    self._root = Explanation_Node(columns=ndict['_columns'], intersections=ndict['_intersections'], caption=ndict['_caption'], description=ndict['_description'], text_attributes=textattr, bbox_attributes=bboxattr, import_variables=node_variables)
                    self._nodelist = [self._root]
                else:
                    new_explanation = Explanation_Node(columns=ndict['_columns'], intersections=ndict['_intersections'], caption=ndict['_caption'], description=ndict['_description'], text_attributes=textattr, bbox_attributes=bboxattr, import_variables=node_variables)
                    self._nodelist.append(new_explanation)

            #
            # Specify children of each node
            #
            for l1 in range(len(import_nodes)):
                # Get this node's variables as a series
                node_variables = import_nodes[import_nodes['_id'] == l1].squeeze(axis=0)
                vv = node_variables.loc['_children']
                # ids is a list of integers that have been read in as text (e.g., [1, 2])
                ids = [] if vv == '[]' else list(map(int, re.split(', ', vv[1:-1])))
                
                for ii in ids:
                    # Add child to node
                    self._nodelist[l1].add_child(self._nodelist[ii])



    def _get_int_to_columns(self):
        """
        Get a dataframe that identifies the columns with missing data and intersections that involve missing data.

        The values are:
          - np.nan (column is not part of the intersection)
          - False (it is but not yet in the graph)
          - True (in the graph; i.e., 'explained')

        Returns
        -------
        DataFrame
            A similar dataframe to _intersection_id_to_columns, but only contains columns with missing data and intersections that involve missing data.

        """
        return self._intersection_id_to_columns.copy().replace({-1: False, 0: np.nan, 1: True}).dropna(axis='columns', how='all').dropna(axis=0, how='all')


    def _set_node_y(self):
        """
        Calculate the Y coordinate (if drawn left to right) of every node. If drawn vertically this is the X coordinate.
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        # The root is positioned top left
        self._root._set_y(self._root._width, 1)    


    def get_node_attributes(self, num):
        """
        Get the attributes for drawing a node
        
        Parameters
        ----------
        num : int
            A node ID

        Returns
        -------
        attributes : dictionary
            A dictionary of attributes or None or the node

        """
        return self._nodelist[num]._attributes


    def reset_node_attributes(self, num_or_list):
        """
        Reset the attributes for drawing one or more nodes.

        Parameters
        ----------
        num_or_list : int or list
            A node ID or a list of IDs

        Returns
        -------
        None.

        """
        if isinstance(num_or_list, list):
            nodenums = num_or_list
        else:
            nodenums = [num_or_list]
            
        for nn in nodenums:
            self._nodelist[nn].reset_attributes()


    def set_node_text_attributes(self, num_or_list, **kwargs):
        """
        Set attributes for drawing the text of one or more nodes.

        Parameters
        ----------
        num_or_list : int or list
            A node ID or a list of IDs
        **kwargs : dictionary
            Keyword arguments. If none are supplied then the attributes are reset to None.

        Returns
        -------
        None.

        """
        if isinstance(num_or_list, list):
            nodenums = num_or_list
        else:
            nodenums = [num_or_list]
            
        for nn in nodenums:
            self._nodelist[nn].set_text_attributes(**kwargs)


    def set_node_bbox_attributes(self, num_or_list, **kwargs):
        """
        Set attributes for drawing the bbox of one or more nodes.

        Parameters
        ----------
        num_or_list : int or list
            A node ID or a list of IDs
        **kwargs : dictionary
            Keyword arguments. If none are supplied then the attributes are reset to None.

        Returns
        -------
        None.

        """
        if isinstance(num_or_list, list):
            nodenums = num_or_list
        else:
            nodenums = [num_or_list]
            
        for nn in nodenums:
            self._nodelist[nn].set_bbox_attributes(**kwargs)
    
    
    def get_node_list(self):
        """
        Get a list of the graph's nodes, in ID order.
        
        Parameters
        ----------
        None.

        Returns
        -------
        list
            A list of the nodes, in ID order

        """
        return self._nodelist
    
    
    def get_node(self, node_id):
        """
        Get a nodes.
        
        Parameters
        ----------
        node_id : int.
            The node ID (0 is the root node)

        Returns
        -------
        Explanation_Node
            A node.

        """
        try:
            ret = self._nodelist[node_id]
        except:
            raise
            
        return ret

    
    def num_nodes(self):
        """
        Return the number of nodes in the graph.
        
        Parameters
        ----------
        None.

        Returns
        -------
        int
            The number of nodes in the graph.

        """
        return len(self._nodelist)


    def get_max_level(self, nodes):
        """
        Get the maximum level of any node in the graph.
        
        Parameters
        ----------
        nodes : list
            A list of the nodes, in ID order

        Returns
        -------
        int
            A list of the nodes, in ID order

        """
        max_level = 0
        
        for n in nodes:
            max_level = max(max_level, n['Level'])
        
        
        return max_level


    def calc_node_width(self):
        """
        Calculate the width of every node
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        self._root.calc_width()


    def calc_layout(self):
        """
        Calculate the layout of the graph, storing it in each node's attributes.'

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        # Reset certain variables for each node
        for nn in self.get_node_list():
            nn._reset()
            
        # Traverse the graph to store each node in the list and set its level from the longest path to it
        self._root._set_node_level(1)
        
        # Get width of every node
        self.calc_node_width()
        
        # Set the Y coordinate (if drawn left to right) for each node. The root is positioned top left
        self._set_node_y()    


    def get_coords(self, node_id, vert=True):
        """
        Get the coords, label and attributes to draw a node in the ExplanationGraph

        Parameters
        ----------
        node_id : int
            The node ID (0 to n).
        vert: boolean, optional
            True (vertical layout; the default) or False (horizontal). The default is True.

        Returns
        -------
        int, int, str, dict
            The node's X coordinate, Y coordinate, label and attributes

        """
        # Get the node
        nn = self._nodelist[node_id]
        text = str(nn._id) + '. ' + nn._caption
        
        if vert:
            px = self._root._width - nn._y
            
            max_level = self._root._x
            
            for n2 in self._nodelist:
                max_level = max(max_level, n2._x)
                
            py = max_level - nn._x
        else:
            px = nn._x
            py = nn._y - 1

        return px, py, text, self._nodelist[node_id].get_attributes()


    def get_column_combination_intersections(self, columns_A, columns_B, get_AB=True, get_A_notB=True, get_B_notA=True, get_not_AB=True):
        """
        Convenience function that returns a list of sets, which contain the intersections that involve combinations of columns.

        Parameters
        ----------
        columns_A : string or list
            A column or list of columns.
        columns_B : string or list
            A column or list of columns.
        get_AB : boolean, optional
            Get intersections that involve all of columns_A and columns_B. The default is True.
        get_A_notB : boolean, optional
            Get intersections that involve all of columns_A but not all of columns_B. The default is True.
        get_B_notA : boolean, optional
            The opposite of get_A_notB. The default is True.
        get_not_AB : boolean, optional
            Get all of the other intersections. The default is True.

        Returns
        -------
        list of sets.
            A list order is [get_AB, get_A_notB, get_B_notA, get_not_AB] but only includes those that are True. Each set is the intersections that involve the given combinations of columns (get_AB, etc.).

        """
        if isinstance(columns_A, list):
            colsA = columns_A
        else:
            colsA = [columns_A]

        if isinstance(columns_B, list):
            colsB = columns_B
        else:
            colsB = [columns_B]

        list_of_sets = []
        
        if get_not_AB:
            ints_not_AB = set(self._intersection_id_to_columns.index.tolist())
        
        if get_AB:
            list_of_sets.append(self.get_intersections(colsA + colsB, all_cols=True))
            
            if get_not_AB:
                ints_not_AB.difference(list_of_sets[-1])
        
        if get_A_notB:
            list_of_sets.append(self.get_intersections(colsA, all_cols=True, exclude_cols=colsB, exclude_all_cols=True))
            
            if get_not_AB:
                ints_not_AB.difference(list_of_sets[-1])
        
        if get_B_notA:
            list_of_sets.append(self.get_intersections(colsB, all_cols=True, exclude_cols=colsA, exclude_all_cols=True))
            
            if get_not_AB:
                ints_not_AB.difference(list_of_sets[-1])

        if get_not_AB:
            list_of_sets.append(ints_not_AB)
            
        return list_of_sets
    

    def get_intersections(self, columns, all_cols=False, exclude_cols=None, exclude_all_cols=False):
        """
        Return the intersections that involve any or all of the supplied column(s).

        Parameters
        ----------
        columns : string or list
            A column or list of columns.
        all_cols : boolean, optional
            True (must involve all of the columns) or False (any of the columns). The default is False.
        exclude_cols : string or list, optional
            A column or list of columns that must not be in the intersections. The default is None.
        exclude_all_cols : boolean, optional
            True (must exclude all of the columns) or False (any of the columns). The default is None.

        Returns
        -------
        set
            The intersections that involve the column(s).

        """
        # Find the intersections that involve these columns
        intersections = set()
        
        if isinstance(columns, list):
            cols = columns
        else:
            cols = [columns]
            
        if all_cols:
            # Intersections must involve all of the columns (i.e., all are non-zero)
            ss = self._intersection_id_to_columns[cols].all(axis=1)
            intersections = set(ss[ss].index.tolist())
        else:
            # Intersections involve any of the columns (i.e., any are non-zero)
            ss = self._intersection_id_to_columns[cols].any(axis=1)
            intersections = set(ss[ss].index.tolist())
            
        # Exclude columns
        if exclude_cols is not None:
            exclude_ints = set()
            
            if isinstance(exclude_cols, list):
                cols = exclude_cols
            else:
                cols = [exclude_cols]
                
            if exclude_all_cols:
                # Intersections must not involve all of the columns (i.e., all are non-zero)
                ss = self._intersection_id_to_columns[cols].all(axis=1)
                exclude_ints = set(ss[ss].index.tolist())
            else:
                # Intersections not involve any of the columns (i.e., any are non-zero)
                ss = self._intersection_id_to_columns[cols].any(axis=1)
                exclude_ints = set(ss[ss].index.tolist())
                
            intersections.difference(exclude_ints)
            
        return intersections
                
        
    def get_columns(self, category):
        """
        Return a list of the columns that are completely, partly or not explained by this graph.

        Parameters
        ----------
        category : str
            'completely explained', 'partly explained' or 'not explained' (case insensitive).

        Returns
        -------
        list
            A list of the columns that are completely, partly or not explained by this graph.

        """
        cols = None
        #
        # This function is based on get_summary()
        #
        int_to_columns = self._get_int_to_columns()
        
        cat = category.lower()
        # True if all values are 1
        aa = 0
        explain_all = int_to_columns.all(axis=aa)
        completely = explain_all[explain_all].index.tolist()
        
        if cat == 'completely explained':
            cols = completely
        else:
            # True if any values are 1
            explain_any = int_to_columns.any(axis=aa)
            sany = set(explain_any[explain_any].index.tolist())
            
            if cat == 'partly explained':
                cols = list(sany.difference(set(completely)))
            elif cat == 'not explained':
                cols = list(set(int_to_columns.columns.tolist()).difference(sany))
            
        return cols
                
        
    def get_summary(self):
        """
        Return a dataframe containing a summary of the amount of missingness that is explained by this graph.

        Parameters
        ----------
        None.

        Returns
        -------
        DataFrame
            Contains the number of intersections/rows/columns/values that are completely/partly/not explained by the graph's nodes.

        """
        
        data = {'Category': ['Completely explained', 'Partly explained', 'Not explained']}
        # Get a dataframe that identifies the columns with missing data and intersections that involve missing data.
        # Values are:
        #   - np.nan (column is not part of the intersection)
        #   - False (it is but not yet in the graph)
        #   - True (in the graph; i.e., 'explained')
        int_to_columns = self._get_int_to_columns()

        # Rows (intersections) then columns
        for aa in range(2):
            # True if any values are 1
            explain_any = int_to_columns.any(axis=aa)
            # True if all values are 1
            explain_all = int_to_columns.all(axis=aa)
            num = int_to_columns.shape[1 if aa==0 else 0]

            num_completely = len(explain_all[explain_all])
            num_partly = len(explain_any[explain_any]) - num_completely

            num_not = num - num_partly - num_completely
            data['Variables' if aa == 0 else 'Combinations'] = [num_completely, num_partly, num_not]

        # Dataframe with the same shape as int_to_columns, but values are
        #   - np.nan (column is not part of the intersection)
        #   - 0 (intersection is 'not explained')
        #   - intersection cardinality (number of records; an 'explained' intersection)
        df_num_explained = int_to_columns.mul(self._intersection_cardinality, axis=0).dropna(axis=0, how='all')
        # Same as df_num_explained, but values are the intersection cardinality (column is part of the intersection) or np.nan
        df_total_missing = int_to_columns.replace(False, True).mul(self._intersection_cardinality, axis=0).dropna(axis=0, how='all')
        
        num_completely = df_num_explained.sum().sum()
        num_not = df_total_missing.sum().sum() - num_completely
        data['Values'] = [num_completely, 0.0, num_not]
        
        # NB: intersection_cardinality.sum() gives the number of rows in the dataset
        num_completely = df_num_explained.min(axis=1).sum()
        num_partly = df_num_explained.max(axis=1).sum() - num_completely
        num_not = df_total_missing.max(axis=1).sum() - num_completely - num_partly
        data['Records'] = [num_completely, num_partly, num_not]
        
        cols = ['Category', 'Combinations', 'Records', 'Variables', 'Values']
        
        return pd.DataFrame.from_dict(data)[cols].set_index('Category')

        
    def add_node(self, new_explanation):
        """
        Add a new explanation to the graph, connected to the appropriate parents.

        Parameters
        ----------
        new_explanation : Explanation_Node
            An explanation.

        Returns
        -------
        None.

        """
        if Explanation_Graph.DEBUG:
            print('Explanation_Graph.add_node() ID =', new_explanation._id)
            
        # Breadth-first search to get a list of the graph's nodes in ID order
        nodes = self.get_node_list()
        # Create a list with one position for each node
        visited = [False] * len(nodes)
        # Start at node zero (the root)
        queue = [0]
        visited[0] = True
        parents = []
        
        while queue:
            # Get the ID of the next node in the queue
            nid = queue.pop(0)
            # Get the Explanation_Node instance for that ID
            nn = nodes[nid]
        	
            if new_explanation.is_intersect(nn, self._criteria):
                # Node nn is a parent of new_explanation because they have columns/intersections in common
                parents.append(nn)
                # Get a list of this node's descendents (by getting a list that includes this node and then setting its entry to None)
                descendents = [None] * len(nodes)
                nn._add_node_and_descendents_to_list(descendents)
                descendents[nid] = None
                
                for n2 in descendents:
                    if n2 is not None:
                        nid2 = n2._id

                        # Remove descendent from parents (if it is in it)
                        try:
                            parents.remove(nid2)
                        except:
                            pass

                        # Remove descendent from queue (if it is in it)
                        try:
                            queue.remove(nid2)
                        except:
                            pass

                        # Add descendent to visited (so it will not be visited)
                        visited[nid2] = True
            else:
                # Node nn is not a parent. Add its children to the queue if they are not in the visited list
                for child in nn._children:
                    if visited[child._id] == False:
                            queue.append(child._id)

        if Explanation_Graph.DEBUG:
            tmp = []
            if len(parents)>0:
                for s in parents:
                    tmp.append(s._id)

            print('Explanation_Graph.add_node() ID =', new_explanation._id, ' PARENTS are ', tmp)
            new_explanation.print()

        if len(parents) == 0:
            # Add the new node to the root node
            self._root.add_child(new_explanation)
        else:
            # Make the new node a child of the nodes calculated from columns/intersections in common
            for exp in parents:
                exp.add_child(new_explanation)
                
        # Update the columns and intersections that are contained in the graph. Warnings are printed if:
        #  - Any column is not part of every intersection (that will always occur if an explanation involves 2+ intersections)
        #  - Any column/intersection pair is already in the Explanation_Graph (that would only occur if a user adds a second explanation for a given intersectin, involving some of the same columns as before)
        warning = [False, False]
        for intersection in new_explanation.get_intersections():
            
            for col in new_explanation.get_columns():
                
                if self._intersection_id_to_columns.iloc[intersection][col] == -1:
                    self._intersection_id_to_columns.iloc[intersection][col] = 1
                else:
                    warning[self._intersection_id_to_columns.iloc[intersection][col]] = True

        # Add the new node to the list
        self._nodelist.append(new_explanation)
        
        if Explanation_Graph.DEBUG:
            if warning[0]:
                print('** WARNING ** Explanation_Graph.add_node(). Some columns are not part of every intersection.')
            elif warning[1]:
                print('** WARNING ** Explanation_Graph.add_node(). Some columns have already been added to the Explanation_Graph for a given intersection.')

        
    def recalculate(self, criteria=CHECK_COLUMNS | CHECK_INTERSECTIONS):
        """
        Recalculate the explanation graph (i.e., with a different criterion).

        Parameters
        ----------
        criterion : int, optional
            Check columns (CHECK_COLUMNS) and/or intersections (CHECK_INTERSECTIONS). Default is both (ORed).

        Returns
        -------
        None.

        """
        # Get a list of the graph's nodes in ID order
        nodes = self.get_node_list()
        
        # Reset certain variables for each node
        for nn in nodes:
            nn._reset()
            nn._remove_children()
            
        self._nodelist = [self._root]
        self._criteria = criteria
        # Using the supplied criterion, recalculate the graph (skipping the root node)
        for nn in nodes[1:]:
            self.add_node(nn)
                    
        
    def output_table(self, filename, overwrite=False, sep='\t', **kwargs):
        """
        Output a table containing each Explanation_Node in a text format file.
    
        Parameters
        ----------
        filename : str
            The full pathname of file.
        overwrite : boolean, optional
            Do not (False) or do overwrite the file if it exists. The default is False.
        sep : str, optional
            The character to use to separate table columns. The default is a TAB.
        **kwargs : dictionary
            Keyword arguments for open()
    
        Returns
        -------
        None.
    
        """
        # Get a list of the graph's nodes in ID order
        nodes = self.get_node_list()
        
        try:
            with open(filename, 'w', **kwargs) as fout:
                # Output the table header
                fout.write(sep.join(['id', 'Caption', 'Description', 'Number of columns', 'Number of intersections', 'Number of records', 'Children']) + '\n')
                
                # Output the details of each node in a separate row
                for nn in nodes:
                    fout.write(nn._get_table_row(sep, self._intersection_cardinality) + '\n')
        except Exception:
            raise

        
class Explanation_Node:
    """
    An Explanation_Node contains a textual explanation of a pattern of missing data, together with the dataset columns and missing data intersections that are involved.
    When explanations are added to the Explanation_Graph they become children of one or more existing explanations in the graph.
    """
    DEBUG = False#True

    def __init__(self, columns=None, intersections=None, caption=None, description=None, text_attributes=None, bbox_attributes=None, import_variables=None):
        """
        Create an explanation node. 

        Parameters
        ----------
        columns : string or list, optional
            The column(s) that are part of this node. Must be provided if the CHECK_COLUMNS criterion is used. The default is None.
        intersections : list, optional
            The IDs of the intersections that are part of this node. Must be provided if the CHECK_INTERSECTIONS criterion is used. The default is None.
        caption : string, optional
            The node caption. The default is None.
        description : string, optional
            The node's description. The default is None.
        text_attributes : dictionary, optional
            Keyword arguments a Matplotlib Text object. The default is None.
        bbox_attributes : dictionary, optional
            Keyword arguments a Matplotlib FancyBboxPatch object. The default is None.
        import_variables : series
            The node's variables. Only used when a graph is being imported. Default is None.

        Returns
        -------
        None.

        """
        if import_variables is None:
            # A new node
            self._reset()
            self._remove_children()
            
            try:
                Explanation_Node.num_nodes = Explanation_Node.num_nodes + 1
            except:
                # Create the graph's root node
                pass
                Explanation_Node.num_nodes = 0
    
            # Attributes for drawing the node
            self._attributes = None
            
            if text_attributes is not None:
                self.set_text_attributes(**text_attributes)
                
            if bbox_attributes is not None:
                self.set_bbox_attributes(**bbox_attributes)
                
            self._id = Explanation_Node.num_nodes
            self._caption = caption
            self._description = description
        else:
            # Import the node
            self._children = []#None
            self._calculated_width = import_variables.loc['_calculated_width']
            self._width = import_variables.loc['_width']
            self._x = import_variables.loc['_x']
            self._y = import_variables.loc['_y']
            self._rendering = 0#import_variables.loc['_rendering']
            
            
            try:
                Explanation_Node.num_nodes = Explanation_Node.num_nodes + 1
            except:
                # Create the graph's root node
                pass
                Explanation_Node.num_nodes = 0
                
            self._attributes = None
            
            if text_attributes is not None:
                self.set_text_attributes(**text_attributes)
                
            if bbox_attributes is not None:
                self.set_bbox_attributes(**bbox_attributes)

            self._id = import_variables.loc['_id']
            self._caption = caption
            self._description = description

        # Code for new/imported node           
        if columns is None:
            self._columns = columns
        elif isinstance(columns, list):
            self._columns = set(columns)
        else:
            self._columns = set([columns])

        if intersections is None:
            self._intersections = None
        elif isinstance(intersections, set):
            self._intersections = set(intersections)
        elif isinstance(intersections, list):
            self._intersections = set(intersections)
        else:
            self._intersections = set([intersections])
            
        if Explanation_Node.DEBUG:
            self.print()


    def _remove_children(self):
        """
        Remove this node's children.

        Returns
        -------
        None.

        """
        #self._parent = None
        self._children = []#None


    def _reset(self):
        """
        Reset certain variables to allow a graph to be recalculated.

        Returns
        -------
        None.

        """
        # The width required for the children and their descendents
        self._calculated_width = False
        self._width = 0
        # X and Y are 'correct' if the graph is drawn left to right. If it's drawn top to bottom then switch X and Y in your head!
        # In both cases the root is positioned top left
        self._x = None
        self._y = None
        # Set to 1 when the graph is plotted
        self._rendering = 0

    
    def _add_node_and_descendents_to_list(self, nodes):
        """
        Add this node to the list, and then add its descendents by traversing its children.
        
        Parameters
        ----------
        nodes : list
            A list of nodes in the graph, in ID order. The list has a position for each node (initialised to None) and updated on exit with this node and its descendents.

        Returns
        -------
        None.

        """
        if nodes[self._id] is None:
            # This node has not yet been added to the list
            nodes[self._id] = self
            
            # Add the node's descendents            
            for ss in self._children:
                ss._add_node_and_descendents_to_list(nodes)

    
    def _set_node_level(self, level):
        """
        Set the level of this node, add it to the list, and then traverse its children. The root node is at level = 1.
        
        Parameters
        ----------
        level : int
            The traversal level (root node is at level 1)

        Returns
        -------
        None.

        """
        if self._x is None or self._x < level:
            # This ensure that each node's level is specified from the longest branch
            self._x = level
            
            for ss in self._children:
                ss._set_node_level(level+1)


    def _set_x(self, x):
        """
        Set the X coordinate to use when node is plotted as a network. If the graph is drawn vertically then this is the Y coordinate.

        Parameters
        ----------
        x : float
            X coordinate to use when node is plotted as a network.

        Returns
        -------
        None.

        """
        self._x = x
        
    
    def _set_y(self, y, rendering):
        """
        Set the Y coordinate to use when node is plotted as a network. If the graph is drawn vertically then this is the X coordinate.
        
        Parameters
        ----------
        y : float
            Y coordinate to use when node is plotted as a network
            
        rendering : int
            A value for this plotting of the network.

        Returns
        -------
        None.

        """
        if rendering > self._rendering:
            # Re-rendering the graph
            self._y = y
            yc = y

            # Traverse graph to set Y for each descendent
            for child in self._children:
                if child._set_y(yc, rendering):
                    # Adjust for the width of this child because it was rendered
                    yc -= child._width
                
            self._rendering = rendering
            ret = True
        else:
            # This node had already been re-rendered
            ret = False
            
        return ret
                    
        
    def _get_table_row(self, sep, intersection_cardinality):
        """
        Get a row for this node to output in a table listing all the graph's nodes.
    
        Parameters
        ----------
        sep : str
            The character to use to separate table columns.
        intersection_cardinality : series
            The number of times each intersection occurs. Index is the 'intersection_id'.
    
        Returns
        -------
        str
            This node's row.
    
        """
        # Get the node's ID, caption and description
        text = ['' if i is None else i for i in [str(self._id), self._caption, self._description]]
        # Number of columns
        text.append('' if self._columns is None else str(len(self._columns)))
        # Number of intersections
        text.append('' if self._intersections is None else str(len(self._intersections)))
        # Number of records
        text.append('' if self._intersections is None else str(intersection_cardinality.loc[list(self._intersections)].sum()))
        # Get the IDs of the node's children
        child_ids = [str(nn._id) for nn in self._children]
        # Convert the information to a single string, which will be output as a row of the table
        row = sep.join(text + [' '.join(child_ids)])
        
        return row


    def get_id(self):
        """
        Return the node's ID

        Returns
        -------
        int
            The node's ID

        """
        return self._id


    def get_children(self):
        """
        Return a list of the node's children

        Returns
        -------
        list
            A  list of the node's children

        """
        return self._children
    
    
    def get_attributes(self):
        """
        Get the attributes for drawing this node

        Returns
        -------
        attributes : dictionary
            A dictionary of attributes or None

        """
        return self._attributes
    
    
    def reset_attributes(self):
        """
        Reset the attributes for drawing this node

        Returns
        -------
        None.

        """
        self._attributes = None
    
    
    def set_text_attributes(self, **kwargs):
        """
        Set attributes for drawing the text of this node.

        Parameters
        ----------
        **kwargs : dictionary
            Keyword arguments.

        Returns
        -------
        None.

        """
        if len(kwargs) > 0:
            if self._attributes is None:
                self._attributes = {}
            
            for key, value in kwargs.items():
                self._attributes[key] = value
    
    
    def set_bbox_attributes(self, **kwargs):
        """
        Set attributes for drawing the bbox of this node.

        Parameters
        ----------
        **kwargs : dictionary
            Keyword arguments.

        Returns
        -------
        None.

        """
        if len(kwargs) > 0:
            if self._attributes is None:
                self._attributes = {'bbox': {}}
            elif 'bbox' not in self._attributes:
                self._attributes['bbox'] = {}
            
            for key, value in kwargs.items():
                self._attributes['bbox'][key] = value
            
    
    def calc_width(self, reset=False):
        """
        Return the width required for this node, and all its children and their descendents.
        
        Parameters
        ----------
        reset : boolean, optional
            True (recalculate all widths) or False (calculation in progress). The default is False.

        Returns
        -------
        None.

        """
        if reset:
            self._calculated_width = False
            self._width = 0
        
        if self._calculated_width == False:
            # NB: If it is true then the width of this node and its descendents has been included in the width of another parent

            # The width is the sum of the widths for all children
            for child in self._children:
                if child._calculated_width == False:
                    # This child's width has not been included in the width of another parent
                    self._width += child.calc_width()

            # Include the space for this node
            self._width = max(self._width, 1)
            self._calculated_width = True
        
        return self._width


    def get_columns(self):
        """
        Return a set that contains this node's columns
        """
        return self._columns


    def get_intersections(self):
        """
        Return a set that contains this node's intersections
        """
        return self._intersections


    def print(self):
        """
        Print the node (for debugging)
        """
        print('Explanation_Node', self._id, self._caption, 'Columns: ', self._columns,
              'Number of intersections =', 0 if self._intersections is None else len(self._intersections))

        
    def add_child(self, child_explanation):
        """
        Add the provided explanation as a child of this one.
        
        Parameters
        ----------
        child_explanation : Explanation
            The child explanation

        Returns
        -------
        None
        
        """
        self._children.append(child_explanation)
            
        if Explanation_Node.DEBUG:
            tmp = []
            for s in self._children:
                tmp.append(s._id)
            print('NODE add_child(): parent ', self._id, ' child ', child_explanation._id, 'list of children', tmp)


    def is_intersect(self, explanation, criterion):
        """
        Determine whether or not this and another explanation involve any of the same columns or intersections.
        
        Parameters
        ----------
        explanation : Explanation
            Another explanation
        criterion : int
            Check columns (CHECK_COLUMNS) and/or intersections (CHECK_INTERSECTIONS).

        Returns
        -------
        Boolean
            True (they do involve some of the same columns/intersections) or False
        
        """
        ret = False
        
        try:
            if (criterion & CHECK_COLUMNS) and len(self._columns.intersection(explanation._columns)) > 0:
                # The explanation involves some of the same columns as this one
                ret = True
            elif (criterion & CHECK_INTERSECTIONS) and len(self._intersections.intersection(explanation._intersections)) > 0:
                # The explanation involves some of the same intersections as this one
                ret = True
        except:
            # Root node has no columns or intersections
            pass
        
        return ret
    