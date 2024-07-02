# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:01:04 2023

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

   Functions for data quality visualizations.

   Internal functions are prefixed by '_'. The functions are grouped as follows:
       General functions
       Unused functions
       Functions for summary plots
       Functions for purity plots
       Functions to plot sets and intersections
       Explanation graph functions

"""
import os
import pandas as pd
import numpy as np
#import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# Internal functions
# =============================================================================
def _draw_fig(filename, overwrite, **kwargs):
    """
    Internal function, which is called to draw a plot to the screen or save it in a file.

    Parameters
    ----------
    filename : string
        None or a filename for the figure
    overwrite : boolean
        False (do not overwrite file) or True (overwrite file if it exists)
    **kwargs : dictionary
        Keyword arguments for fig.savefig().

    Returns
    -------
    None.

    """
    if filename is not None:
        if overwrite or not os.path.isfile(filename):
            fig=plt.gcf()
            fig.savefig(filename, **kwargs)
        elif overwrite == False:
            print('** WARNING ** vizdataquality, plot.py, _draw_fig(): Figure not output because a file with the supplied name already exists.')
            print(filename)

    plt.show()


def _get_parameter(value, num):
    """
    Get the 'num'th item in a list of values, or value itself

    Parameters
    ----------
    value : list or other data type
        A list of values, or a single value.
    num : integer
        Index for the list.

    Returns
    -------
    The 'num'th item in the list, or the value itself.

    """
    param = value

    if isinstance(value, list):
        try:
            # Get the 'num'th item in the list
            param = value[num]
        except:
            pass

    return param


# =============================================================================
# Utility functions
# =============================================================================
def apply_perceptual_discontinuity_individually(input_data, perceptual_threshold, axis_limits=None):
    """
    Apply perceptual discontinuity threshold.
    If axis_limits is None then values in range 0 < x < perceptual_threshold are set equal to perceptual_threshold.
    Otherwise values in the range 0 < x < max are adjusted so each is distinguishable from 0 and max.

    Parameters
    ----------
    input_data : series or data frame
        The data.
    perceptual_threshold : float
        An absolute value (axis_limits = None) or a percentage (0.0 - 1.0) of the axis limit max.
    axis_limits : None or (min, max) tuple
        The axis limits. The default is None.

    Returns
    -------
    DataFrame
        The adjusted values and, if a series was input, then two extra columns for stacked bar chart plotting of values that have been adjusted and not adjusted, respectively.

    """
    #data = input_data.copy()
    # Distinguish between original values and values adjusted by perceptual discontinuity
    legend_labels = ['Perceptual discontinuity', 'Original number']
    # Finer-grained distinction
    #legend_labels = []

    if axis_limits is None or not isinstance(axis_limits, tuple):
        # The value of perceptual_threshold is used
        val1 = perceptual_threshold

        if type(input_data) is pd.Series:
            #data = input_data.apply(lambda x: x if x <= 0 else max(x, perceptual_threshold)).to_frame()
            #.reset_index().rename(columns={data.name: 'Value', 'count': 'Count'})
            stack = []
            for l1 in range(2):
                stack.append([0] * len(input_data))

            l1 = 0
            for index, value in input_data.items():
                if value > 0.0 and value < val1:
                    stack[0][l1] = val1
                elif value >= val1:
                    stack[1][l1] = value

                l1 += 1

            data = input_data.to_frame()

            if len(legend_labels) == 2:
                data[legend_labels[0]] = stack[0]
                data[legend_labels[1]] = stack[1]
            else:
                data['0 < value < ' + str(val1)] = stack[0]
                data[str(val1) + ' <= value'] = stack[1]
        else:
            data = input_data.apply(lambda x: x if x <= 0 else max(x, val1))

    else:
        # The threshold is perceptual_threshold * axis_limits[1], which assumes that min = 0
        max_value = axis_limits[1]
        val1 = perceptual_threshold*max_value
        val2 = (1.0-perceptual_threshold)*max_value

        if val1.is_integer() and val2.is_integer():
            val1 = int(val1)
            val2 = int(val2)

        if type(input_data) is pd.Series:
            stack = []
            for l1 in range(2 if len(legend_labels) == 2 else 4):
                stack.append([0] * len(input_data))

            if len(legend_labels) == 2:
                l1 = 0
                for index, value in input_data.items():
                    if value > 0.0 and value < val1:
                        # Value adjusted by perceptual discontinuity
                        stack[0][l1] = val1
                    elif value <= val2:
                        # Value is unchanged
                        stack[1][l1] = value
                    elif value < max_value:
                        # Value adjusted by perceptual discontinuity
                        stack[0][l1] = val2
                    elif value >= max_value:
                        # Value is unchanged
                        stack[1][l1] = value

                    l1 += 1

                data = input_data.to_frame()
                data[legend_labels[0]] = stack[0]
                data[legend_labels[1]] = stack[1]
            else:
                l1 = 0
                for index, value in input_data.items():
                    if value > 0.0 and value < val1:
                        stack[0][l1] = val1
                    elif value <= val2:
                        stack[1][l1] = value
                    elif value < max_value:
                        stack[2][l1] = val2
                    elif value >= max_value:
                        stack[3][l1] = value

                    l1 += 1

                data = input_data.to_frame()
                data['0 < value < ' + str(val1)] = stack[0]
                data[str(val1) + ' <= value <= ' + str(val2)] = stack[1]
                data[str(val2) + ' < value < ' + str(max_value)] = stack[2]
                data['value = ' + str(max_value)] = stack[3]
        else:
            data = input_data.apply(lambda x: x if x <= 0 or x >= max_value else (max(x, perceptual_threshold*max_value) if x < max_value/2.0 else min(x, (1.0-perceptual_threshold)*max_value)))

    return data


def apply_perceptual_discontinuity_to_group(input_data, perceptual_threshold, axis_limits=None):
    """
    Apply perceptual discontinuity threshold to a group of values. The values are adjusted so each is >= perceptual_threshold * sum, but the sum is unchanged.
    That is achieved by increasing values that are below the threshold and decreasing values that are above the threshold.

    Parameters
    ----------
    input_data : series or data frame
        The data
    perceptual_threshold : float
        An absolute value (axis_limits = None) or a percentage (0.0 - 1.0) of the axis limit max.
    axis_limits : None or (min, max) tuple
        The axis limits. The default is None.

    Returns
    -------
    Series or DataFrame
        A series or dataframe, with perceptual discontinuity applied to groups of values (e.g. in a stacked bar chart)

    """
    data = input_data.copy()
    
    if axis_limits is None or not isinstance(axis_limits, tuple):
        p = perceptual_threshold
    else:
        # The threshold is perceptual_threshold * axis_limits[1], which assumes that min = 0
        p = perceptual_threshold * axis_limits[1]

    if isinstance(data, pd.Series):
        total = data.sum()
        # First, make sure every value is >= threshold
        data = data.apply(lambda x: 0.0 if x <= 0.0 else max(x, p))

        if data.sum() > total:
            # The sum of the values is > max_value, so calculate the excess
            e = float(data.sum() - total)
            sump = float(data.sum() - p * len(data))
            # Reduce the values proportionately, making sure every value is still >= threshold
            data = data.apply(lambda x: x if x <= p else x - e*(x-p)/sump)
    elif isinstance(data, pd.DataFrame):
        for col in data.columns:
            total = data[col].sum()
            # First, make sure every value is >= threshold
            data[col] = data[col].apply(lambda x: 0.0 if x <= 0.0 else max(x, p))

            if data[col].sum() > total:
                # The sum of the values is > max_value, so calculate the excess
                e = float(data[col].sum() - total)
                sump = float(data[col].sum() - p * (data.shape[0]))
                # Reduce the values proportionately, making sure every value is still >= threshold
                data[col] = data[col].apply(lambda x: x if x <= p else x - e*(x-p)/sump)

    return data


# =============================================================================
# Functions for multiple plots
# =============================================================================
def plotgrid(tasktype, data, num_rows=None, num_cols=None, vert=True, xlabels_rotate=0.0, perceptual_threshold=0.05, legend=True, components='raw data', gap_threshold=None, show_gaps=True, datalabels=False, continuous_value_axis=True, filename=None, overwrite=False, fig_kw={}, ax_kw={}, legend_kw={}, **kwargs):
    """
    Create a grid of plots of a given type.

    Parameters
    ----------
    tasktype : string
        'datetime distribution', 'scalars' or 'value counts'
    data : dataframe or series
        The data.
    num_rows : int, optional
        The number of rows in the grid. The default is None.
    num_cols : TYPE, optional
        The number of columns in the grid. The default is None.
    vert: boolean
        True (vertical bars; the default) or False (horizontal)
    xlabels_rotate : float
        Angle to rotate X axis labels by (only used if vert = True)
    perceptual_threshold : float
        Preceptual discontinuity threshold (0.0 - 1.0) or None
    legend: boolean
        True (add a legend, if a stacked bar chart is plotted) or False (no legend)
    components : list or string
        Only used if tasktype is 'datetime distribution': Component(s) to plot ('year', 'month', 'dayofweek', 'hour', 'minute' or 'second'; case independent) or 'raw data' (default)
    gap_threshold: None, int or datetime
        Only used if tasktype is 'datetime distribution': None (threshold will be based on the component of the data; the default) or value (threshold to use). Only used if component is specified.
    show_gaps: boolean
        Only used if tasktype is 'datetime distribution': True (the default) or False (draw lines across gaps). Only used if component is specified.
    datalabels : boolean
        Label each data point (False (default) or True)
    continuous_value_axis : boolean
        Plot numerical/datetime values on a continuous axis to show any gaps in values. The default is True.
    filename : string
        None or a filename for the figure
    overwrite : boolean
        False (do not overwrite file; the default) or True (overwrite file if it exists)
    fig_kw : dictionary
        Keyword arguments for a Matplotlib Figure object
    ax_kw : dictionary
        Keyword arguments for a Matplotlib Axes object
    **kwargs : dictionary
        Keyword arguments for the plotting function, e.g., scalar_bar().

    Returns
    -------
    None.
    """
    if tasktype == 'datetime distribution' or tasktype == 'scalars' or tasktype == 'value counts':
        axkwargs = ax_kw.copy()
        # Get size of data
        data_nrow = len(data)
        data_ncol = 1 if isinstance(data, pd.Series) else data.shape[1]

        if tasktype == 'datetime distribution':
            # A separate plot is created for each component of each column
            if isinstance(components, list):
                comps = components
            else:
                comps = [components]

            data_ncol = data_ncol * len(comps)

        # Calculate the number of rows/columns of plots
        if num_rows is None and num_cols is None:
            # Default is 2 columns if a dataframe is input
            ncols = 1 if isinstance(data, pd.Series) else 2
            nrows = int((data_ncol + ncols - 1) / ncols)
        elif num_rows is None:
            ncols = num_cols
            nrows = int((data_ncol + ncols - 1) / ncols)
        elif num_cols is None:
            nrows = num_rows
            ncols = int((data_ncol + nrows - 1) / nrows)
        else:
            nrows = num_rows
            ncols = num_cols

        # Create the figure and subplots
        fig, axs = plt.subplots(nrows, ncols)

        # Figure kwargs are applied here
        fig.set(**fig_kw)

        if 'title' in axkwargs and isinstance(axkwargs['title'], str):
            fig.suptitle(ax_kw['title'])
            axkwargs['title'] = None
        #
        # Create the plots
        #
        # Dataframe column number
        cnum = 0
        # Date component number
        dnum = 0
        # Parameter number in list
        pnum = 0

        for l1 in range(nrows):
            for l2 in range(ncols):

                if type(axs) is mpl.axes.Axes:
                    ax = axs
                elif ncols == 1:
                    ax = axs[l1]
                elif nrows == 1:
                    ax = axs[l2]
                else:
                    ax = axs[l1][l2]

                if isinstance(data, pd.Series):
                    plotdata = data if cnum == 0 else None
                else:
                    plotdata = data[data.columns[cnum]] if cnum < data_ncol else None

                #plotdata = data if isinstance(data, pd.Series) else data[data.columns[cnum]] if cnum < data_ncol else None

                if plotdata is not None:
                    # If a list of titles is provided then apply the title for this plot
                    kw = 'title'

                    if kw in ax_kw and isinstance(ax_kw[kw], list):
                        # Need to check whether title is a list, so the suptitle() isn't used
                        axkwargs[kw] = _get_parameter(ax_kw[kw], pnum)

                    # Get rotation for this plot, if a list was provided
                    xlabrot = _get_parameter(xlabels_rotate, pnum)

                    # This check is made because the grid may contain more cells than there are dataframe columns
                    if tasktype == 'scalars':
                        scalar_bar(plotdata, ax_input=ax, vert=vert, xlabels_rotate=xlabrot, perceptual_threshold=perceptual_threshold, legend=legend, filename=filename, overwrite=overwrite, ax_kw=axkwargs, legend_kw={}, **kwargs)

                        cnum += 1
                    elif tasktype == 'datetime distribution':
                        datetime_counts(plotdata, ax_input=ax, xlabels_rotate=xlabrot, component=comps[dnum], gap_threshold=gap_threshold, show_gaps=show_gaps, filename=filename, overwrite=overwrite, ax_kw=axkwargs, **kwargs)

                        if dnum < len(comps) - 1:
                            # Next component
                            dnum += 1
                        else:
                            # First component of next dataframe column
                            cnum += 1
                            dnum = 0
                    elif tasktype == 'value counts':
                        lollipop(plotdata.value_counts(), ax_input=ax, vert=vert, xlabels_rotate=xlabrot, datalabels=datalabels, continuous_value_axis=continuous_value_axis, filename=filename, overwrite=overwrite, ax_kw=axkwargs, **kwargs)
                        cnum += 1

                    # Increment the parameter number
                    pnum += 1
                else:
                    # This cell in the grid is not used
                    ax.axis('off')

        #
        # Draw or output the figure to a file
        #
        _draw_fig(filename, overwrite)
    else:
        print('** WARNING ** vizdataquality, plot.py, plotgrid(): The tasktype is not valid:', tasktype)


def multiplot(plottype, data, perceptual_threshold=0.05, number_of_variables_per_row=None, vert=True, xlabels_rotate=0.0, clist=[], datalabels=False, legend=True, continuous_value_axis=True, filename=None, overwrite=False, plt_kw={}, fig_kw={}, ax_kw={}, legend_kw={}, **kwargs):
    """
    Plot a data quality attribute (e.g., number of missing values in each variable).
    The variables can be plotted on multiple rows of bar charts.
    The length of each bar can be adjusted to ensure that important perceptual differences are visible.

    Parameters
    ----------
    plottype : string
        'bar', 'box', 'dot-or-whisker', 'lollipop', 'stackedbar' or 'violin'
    data : dataframe (stackedbar or violinplot) or series (all plot types except stackedbar)
        'bar', 'box', 'dot-or-whisker': Series containing the variable names (index) and data quality attribute to be plotted (e.g., number of missing values in each variable)
        'lollipop': Series containing value counts.
        'stackedbar': Dataframe where each column is a bar and the index/rows are the stacks
        'violin': The values of one (Series) or more columns (dataframe)
    perceptual_threshold : float
        Preceptual discontinuity threshold (0.0 - 1.0) or None
    number_of_variables_per_row : int
        None (plot all variables in one bar chart) or the number of variables to show in each row
    vert: boolean
        True (vertical bars; the default) or False (horizontal)
    xlabels_rotate : float
        Angle to rotate X axis labels by (only used if vert = True)
    clist : list, optional
        A list of the colours to use (a different one for each stack in stacked bars). The default is an empty list (use the default colours).
    datalabels : boolean
        Label each data point. The default is False.
    legend: boolean
        True (add a legend) or False (no legend). The default is True.
    continuous_value_axis : boolean
        Plot numerical/datetime values on a continuous axis to show any gaps in values. The default is True.
    filename : string
        None or a filename for the figure
    overwrite : boolean
        False (do not overwrite file; the default) or True (overwrite file if it exists)
    plt_kw : dictionary
        Keyword arguments for a Matplotlib pyplot object
    fig_kw : dictionary
        Keyword arguments for a Matplotlib Figure object
    ax_kw : dictionary
        Keyword arguments for a Matplotlib Axes object
    legend_kw : dictionary
        Keyword arguments for a Matplotlib legend. Only used if plottype = 'bar'
    kwargs : dictionary
        Keyword arguments for a Matplotlib Axes.bar (scalarbar), Axes.bxp (boxplot), Axes.scatter and Axes.errorbar (dot_whisker), Axes.scatter and Axes.plot (lollipop) or violinplot object (violin)

    Returns
    -------
    None.

    """
    if isinstance(data, pd.DataFrame) and plottype != 'violin' and plottype != 'stackedbar':
        print('** WARNING ** vizdataquality, plot.py, multiplot(): A dataframe cannot be used with a', plottype, 'plot')
    elif plottype == 'bar' or plottype == 'box' or plottype == 'dot-or-whisker' or plottype == 'lollipop' or plottype == 'stackedbar' or plottype == 'violin':
        axkwargs = ax_kw.copy()
        # Calculate the number of rows/columns of plots
        if plottype == 'violin' or plottype == 'stackedbar':
            if isinstance(data, pd.DataFrame):
                # The dataframe columns are the violins/bars
                num_subplots = int((data.shape[1] + number_of_variables_per_row - 1) / number_of_variables_per_row)
            else:
                num_subplots = 1
        else:
            # The other plot types
            try:
                num_subplots = int((len(data) + number_of_variables_per_row - 1) / number_of_variables_per_row)
            except:
                num_subplots = 1

        if vert:
            # Multiple rows of plots
            fig, axs = plt.subplots(num_subplots, sharey=True)
        else:
            # Multiple columns of plots
            fig, axs = plt.subplots(1, num_subplots, sharex=True)

        # Figure kwargs are applied here and not passed to scalar_bar()
        fig.set(**fig_kw)
        #grid_shape = (num_subplots, num_cols)

        if True:
            # This shouldn't be needed because "When subplots have a shared x-axis along a column, only the x tick labels of the bottom subplot are created". Similarly for the Y axis
            # Use a single label for each axis
            if plottype == 'violin':
                default_axis_label = 'Value'
            elif plottype == 'lollipop' or plottype == 'stackedbar':
                default_axis_label = 'Count'
            else:
                default_axis_label = data.name

            label1 = 'Value' if plottype == 'lollipop' else 'Variable'
            fig.supxlabel(axkwargs.get('xlabel', label1 if vert else default_axis_label))
            axkwargs['xlabel'] = None
            fig.supylabel(axkwargs.get('ylabel', default_axis_label if vert else label1))
            axkwargs['ylabel'] = None
        #
        # Sort the values if plotting value counts
        #
        if plottype == 'lollipop':
            plotdata = data.sort_index()
        else:
            plotdata = None
        #
        # Create the plots in 1+ rows
        #
        for l1 in range(num_subplots):
            #
            # Calculate which variables are plotted in this row
            if num_subplots == 1:
                # Plot all variables in one row
                start = 0
                end = len(data)
            elif l1 == num_subplots - 1:
                # This is the last of multiple rows
                start = l1 * number_of_variables_per_row
                
                if isinstance(data, pd.DataFrame):
                    # stackedbar or violin
                    end = data.shape[1]
                else:
                    end = len(data)
            else:
                # All but the last of multiple rows
                start = l1 * number_of_variables_per_row
                end = start + number_of_variables_per_row

            #ax = plt.subplot2grid(shape=grid_shape, loc=(l1, 0))
            ax = axs if type(axs) is mpl.axes.Axes else axs[l1]

            if plottype == 'bar':
                # Add the legend to the first subplot
                add_legend = legend if l1 == 0 else False
                scalar_bar(data.iloc[start:end], perceptual_threshold=perceptual_threshold, number_of_variables_per_row=number_of_variables_per_row, ax_input=ax, vert=vert, xlabels_rotate=xlabels_rotate, datalabels=datalabels, legend=add_legend, filename=filename, overwrite=overwrite, ax_kw=axkwargs, legend_kw=legend_kw, **kwargs)
            elif plottype == 'box':
                boxplot(data.iloc[start:end], number_of_variables_per_row=number_of_variables_per_row, ax_input=ax, vert=vert, xlabels_rotate=xlabels_rotate, filename=filename, overwrite=overwrite, ax_kw=axkwargs, **kwargs)
            elif plottype == 'dot-or-whisker':
                dot_whisker(data.iloc[start:end], number_of_variables_per_row=number_of_variables_per_row, ax_input=ax, vert=vert, xlabels_rotate=xlabels_rotate, filename=filename, overwrite=overwrite, ax_kw=axkwargs, **kwargs)
            elif plottype == 'lollipop':
                lollipop(plotdata[start:end], number_of_variables_per_row=number_of_variables_per_row, ax_input=ax, vert=vert, xlabels_rotate=xlabels_rotate, datalabels=datalabels, continuous_value_axis=continuous_value_axis, filename=filename, overwrite=overwrite, ax_kw=axkwargs, **kwargs)
            elif plottype == 'stackedbar':
                # Add the legend to the first subplot
                add_legend = legend if l1 == 0 else False
                stacked_bar(data[data.columns[start:end]], perceptual_threshold=perceptual_threshold, number_of_variables_per_row=number_of_variables_per_row, ax_input=ax, vert=vert, xlabels_rotate=xlabels_rotate, clist=clist, datalabels=datalabels, legend=add_legend, filename=filename, overwrite=overwrite, ax_kw=axkwargs, legend_kw=legend_kw, **kwargs)
            elif plottype == 'violin':
                violinplot(data[data.columns[start:end]], number_of_variables_per_row=number_of_variables_per_row, ax_input=ax, vert=vert, xlabels_rotate=xlabels_rotate, filename=filename, overwrite=overwrite, ax_kw=axkwargs, **kwargs)

        #
        # Plot keyword arguments
        #
        plt.xlim(plt_kw.get('xlim', None))
        plt.ylim(plt_kw.get('ylim', None))
        #
        # Draw or output the figure to a file
        #
        _draw_fig(filename, overwrite)
    else:
        print('** WARNING ** vizdataquality, plot.py, multiplot(): The plottype is not valid:', plottype)


# =============================================================================
# Functions for summary plots
# =============================================================================
def scalar_bar(data, perceptual_threshold=0.05, number_of_variables_per_row=None, ax_input=None, vert=True, xlabels_rotate=0.0, datalabels=False, legend=True, filename=None, overwrite=False, fig_kw={}, ax_kw={}, legend_kw={}, **kwargs):
    """
    Create a bar chart showing a data quality attribute (e.g., number of missing values in each variable).
    The length of each bar can be adjusted to ensure that important perceptual differences are visible.

    Parameters
    ----------
    data : series
        Series containing the variable names (index) and data quality attribute to be plotted (e.g., number of missing values in each variable)
    perceptual_threshold : float
        Preceptual discontinuity threshold (0.0 - 1.0) or None. The default is 0.05.
    number_of_variables_per_row : int
        None (plot all variables in one bar chart) or the number of variables to show in each row. The default is None.
    ax_input: axis or None
        Matplotlib axis. The default is None.
    vert: boolean
        True (vertical bars; the default) or False (horizontal). The default is True.
    xlabels_rotate : float
        Angle to rotate X axis labels by (only used if vert = True). The default is 0.0.
    datalabels : boolean
        Label each data point. The default is False.
    legend: boolean
        NOT CURRENTLY USED. True (add a legend, if perceptual discontinuity is used) or False (no legend). The default is True.
    filename : string
        None or a filename for the figure. The default is None.
    overwrite : boolean
        False (do not overwrite file; the default) or True (overwrite file if it exists). The default is False.
    fig_kw : dictionary
        Keyword arguments for a Matplotlib Figure object. The default is an empty dictionary.
    ax_kw : dictionary
        Keyword arguments for a Matplotlib Axes object. The default is an empty dictionary.
    legend_kw : dictionary
        NOT CURRENTLY USED. Keyword arguments for a Matplotlib legend. The default is an empty dictionary.
    kwargs : dictionary
        Keyword arguments for a Matplotlib Axes.bar object

    Returns
    -------
    None.

    """
    axkwargs = ax_kw.copy()
    kw = kwargs.copy()
    #
    # Perceptual discontinuity
    #
    axis_limits = ax_kw.get('ylim' if vert else 'xlim', None)

    if perceptual_threshold is None:
        # Plot the values in the input data frame
        plotdata = data
    else:
        # Apply perceptual discontinuity threshold so that non-zero bars are visible and almost complete variables do not look complete
        plotdata = apply_perceptual_discontinuity_individually(data, perceptual_threshold, axis_limits)
        edgecolour = [0,0,0]
        stack_colours = [[1,1,1], [0.67,0.67,0.67], [0.33,0.33,0.33], [0,0,0]]
    #
    # Plot axis
    #
    if ax_input is None:
        #fig = plt.figure()
        fig, ax = plt.subplots()
        fig.set(**fig_kw)
        #ax = fig.add_subplot()
    else:
        ax = ax_input

    # The index contains the variable names
    xlabels = plotdata.index.to_list()
    # Set this to False to create stacks that use different colours for the original and adjusted values
    percep_disc_onebar = True

    # Store the data quality attribute in an array
    if type(plotdata) is pd.Series:
        name = plotdata.name
        row_values = plotdata.to_numpy()
        stack = None
    else:
        name = plotdata.columns[0]
        row_values = plotdata[name].to_numpy()
        stack = {}
        
        if percep_disc_onebar:
            stack[plotdata.columns[1]] = plotdata[plotdata.columns[1:]].sum(axis=1).to_numpy()
        else:
            for col in plotdata.columns[1:]:
                stack[col] = plotdata[col].to_numpy()

    # By default, draw axis labels  at the end of the bars
    threshold = None

    try:
        if isinstance(axis_limits, tuple):
            # Threshold defines whether an axis label will be drawn at the end of a bar or within it
            threshold = (axis_limits[1] - max(0, axis_limits[0])) * 0.2
    except:
        pass

    try:
        # If necessary, extend the arrays to accommodate any dummy variables at the end of the plot
        num_bars_in_row = max(len(plotdata), number_of_variables_per_row)
        num_append = num_bars_in_row - len(plotdata)
        row_values = np.append(row_values, [0] * num_append)
        xlabels = xlabels + [''] * num_append

        if isinstance(stack, dict):
            for key, value in stack.items():
                stack[key] = np.append(value, [0] * num_append)

        #print('stack', stack)
    except:
        # Just plot the variables
        pass
        num_bars_in_row = len(plotdata)

    # Offset for some data value labels
    label_padding = 10

    #
    # bar() and barh() calls are similar for vertical and horizontal bar chart
    #
    if stack is None:
        if vert:
            p = ax.bar(np.arange(num_bars_in_row), row_values, **kw)
        else:
            p = ax.barh(np.arange(num_bars_in_row), row_values, **kw)

        if datalabels:
            # Draw axis labels  at the end of the bars
            bar_labels = [str(data.values[i]) for i in range(len(data))]

            if num_bars_in_row > len(bar_labels):
                bar_labels += [''] * (num_bars_in_row - len(bar_labels))

            ax.bar_label(p, labels=bar_labels, padding=label_padding)
    else:
        # Stacked bar chart
        bottom = np.zeros(num_bars_in_row)
        l1 = 0

        for key, value in stack.items():
            if value.max() > 0:
                # Check there are values to be plotted
                if vert:
                    if percep_disc_onebar:
                        p = ax.bar(np.arange(num_bars_in_row), value, bottom=bottom, label=key, **kw)
                    else:
                        p = ax.bar(np.arange(num_bars_in_row), value, bottom=bottom, label=key, ec=edgecolour, color=stack_colours[l1], **kw)
                else:
                    if percep_disc_onebar:
                        p = ax.barh(np.arange(num_bars_in_row), value, left=bottom, label=key, **kw)
                    else:
                        p = ax.barh(np.arange(num_bars_in_row), value, left=bottom, label=key, ec=edgecolour, color=stack_colours[l1], **kw)

                if datalabels:
                    # Draw bar labels for non-zero values in this stack
                    if threshold is None:
                        # Draw axis labels  at the end of the bars
                        bar_labels = ['' if row_values[i] == 0 else str(data.values[i]) for i in range(len(data))]

                        if num_bars_in_row > len(bar_labels):
                            bar_labels += [''] * (num_bars_in_row - len(bar_labels))

                        ax.bar_label(p, labels=bar_labels, padding=label_padding)
                    else:
                        # Threshold defines whether an axis label will be drawn at the end of a bar or within it
                        # NB: There should be only one item in the stack for each stacked bar
                        bar_labels = ['' if value[i] < threshold else str(data.values[i]) for i in range(len(data))]

                        if num_bars_in_row > len(bar_labels):
                            bar_labels += [''] * (num_bars_in_row - len(bar_labels))

                        ax.bar_label(p, labels=bar_labels, label_type='center')
                        bar_labels = ['' if value[i] >= threshold or value[i] == 0 else str(data.values[i]) for i in range(len(data))]

                        if num_bars_in_row > len(bar_labels):
                            bar_labels += [''] * (num_bars_in_row - len(bar_labels))

                        ax.bar_label(p, labels=bar_labels, padding=label_padding)

                bottom += value

            # Increment colour number
            l1 += 1

        if percep_disc_onebar == False and legend:
            ax.legend(**legend_kw)
        # Add legend to the figure, so there is only one legend if multiplot() was called
        #fig=plt.gcf()
        #fig.legend(**legend_kw)

    if vert:
        # Set the default axis labels if none have been supplied as ax_kw
        if 'xlabel' not in axkwargs:
            axkwargs['xlabel'] = 'Variable'

        if 'ylabel' not in axkwargs:
            axkwargs['ylabel'] = name

        ax.set(**axkwargs)
        # Create the X ticks (including any dummy variables)
        ax.set_xticks(range(num_bars_in_row))
        ax.set_xticklabels(xlabels, rotation=xlabels_rotate)
        ax.set_xlim(-0.5, num_bars_in_row - 0.5)
        #ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        #ax.ticklabel_format(axis='x', style='plain')
    else:
        # Set the default axis labels if none have been supplied as ax_kw
        if 'xlabel' not in axkwargs:
            axkwargs['xlabel'] = name

        if 'ylabel' not in axkwargs:
            axkwargs['ylabel'] = 'Variable'

        ax.set(**axkwargs)

        if abs(xlabels_rotate) > 0.0:
            ax.tick_params('x', labelrotation=xlabels_rotate)

        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        # Create the Y ticks (including any dummy variables)
        ax.set_yticks(range(num_bars_in_row))
        ax.set_yticklabels(xlabels)
        ax.set_ylim(-0.5, num_bars_in_row - 0.5)

    if ax_input is None:
        _draw_fig(filename, overwrite)


def stacked_bar(data, perceptual_threshold=0.05, number_of_variables_per_row=None, ax_input=None, vert=True, xlabels_rotate=0.0, clist=[], elist=[], datalabels=False, legend=True, filename=None, overwrite=False, fig_kw={}, ax_kw={}, legend_kw={}, **kwargs):
    """
    Create a bar chart showing a data quality attribute (e.g., number of missing values in each variable).
    The length of each bar can be adjusted to ensure that important perceptual differences are visible.

    Parameters
    ----------
    data : series or dataframe
        The data to be plotted (single bar for a series; one bar per column for a dataframe). The index contains the names of the stacks. 
    perceptual_threshold : float
        Preceptual discontinuity threshold (0.0 - 1.0) or None. The default is 0.05.
    number_of_variables_per_row : int
        None (plot all variables in one bar chart) or the number of variables to show in each row. The default is None.
    ax_input: axis or None
        Matplotlib axis. The default is None.
    vert: boolean
        True (vertical bars; the default) or False (horizontal). The default is True.
    xlabels_rotate : float
        Angle to rotate X axis labels by (only used if vert = True). The default is 0.0.
    clist : list, optional
        The fill colours to use (a different one for each stack). The default is an empty list (use the default colours).
    elist : list, optional
        The edge colours to use (a different one for each stack). The default is an empty list (use the default colours).
    datalabels : boolean
        Label each data point. The default is False.
    legend: boolean
        True (add a legend) or False (no legend). The default is True.
    filename : string
        None or a filename for the figure. The default is None.
    overwrite : boolean
        False (do not overwrite file; the default) or True (overwrite file if it exists). The default is False.
    fig_kw : dictionary
        Keyword arguments for a Matplotlib Figure object. The default is an empty dictionary.
    ax_kw : dictionary
        Keyword arguments for a Matplotlib Axes object. The default is an empty dictionary.
    legend_kw : dictionary
        Keyword arguments for a Matplotlib legend. The default is an empty dictionary.
    kwargs : dictionary
        Keyword arguments for a Matplotlib Axes.bar object

    Returns
    -------
    None.

    """
    axkwargs = ax_kw.copy()
    kw = kwargs.copy()
    #
    # Perceptual discontinuity
    #
    axis_limits = ax_kw.get('ylim' if vert else 'xlim', None)

    if perceptual_threshold is None:
        # Plot the values in the input data frame
        plotdata = data
    else:
        # Apply perceptual discontinuity threshold so that non-zero bars are visible and almost complete variables do not look complete
        plotdata = apply_perceptual_discontinuity_to_group(data, perceptual_threshold, axis_limits)
    #
    # Plot axis
    #
    if ax_input is None:
        fig, ax = plt.subplots()
        fig.set(**fig_kw)
    else:
        ax = ax_input

    # Store the data for the bar stacks in a dictionary
    if isinstance(plotdata, pd.Series):
        xlabel = 'Variable'
        name = 'Count'
        xlabels = [plotdata.name]
        stack = {}
        
        for index, value in plotdata.items():
            stack[index] = np.array([value])
    else:
        xlabel= 'Variable'
        name = 'Count'
        # The index contains the variable names
        xlabels = plotdata.columns.to_list()
        stack = {}

        for l1 in range(len(plotdata)):
            stack[plotdata.index[l1]] = plotdata.values[l1]
            
    # By default, draw axis labels  at the end of the bars
    threshold = None
    num_bars = 1 if isinstance(plotdata, pd.Series) else plotdata.shape[1]

    try:
        if isinstance(axis_limits, tuple):
            # Threshold defines whether an axis label will be drawn at the end of a bar or within it
            threshold = (axis_limits[1] - max(0, axis_limits[0])) * 0.2
    except:
        pass

    try:
        # If necessary, extend the arrays to accommodate any dummy variables at the end of the plot
        num_bars_in_row = max(num_bars, number_of_variables_per_row)
        num_append = num_bars_in_row - num_bars
        xlabels = xlabels + [''] * num_append

        if isinstance(stack, dict):
            for key, value in stack.items():
                stack[key] = np.append(value, [0] * num_append)
    except:
        # Just plot the variables
        num_bars_in_row = num_bars
        pass

    # Set the colourmap
    #stack_colours = [[1,1,1], [0.67,0.67,0.67], [0.33,0.33,0.33], [0,0,0]]
    if isinstance(clist, list) and len(clist) == len(stack):
        fillcolours = clist
    else:
        # Greyscale
        #colours = [[i/(len(stack)-1.0), i/(len(stack)-1.0), i/(len(stack)-1.0)] for i in range(len(stack))]
        fillcolours = None
    
    edgecolours = elist if isinstance(elist, list) and len(elist) == len(stack) else None
    #if 'edgecolor' not in kw:
    #    kw['edgecolor'] = [0,0,0]
        
    # Offset for some data value labels
    label_padding = 10

    #
    # bar() and barh() calls are similar for vertical and horizontal bar chart
    #
    # Stacked bar chart
    bottom = np.zeros(num_bars_in_row)
    l1 = 0

    for key, value in stack.items():

        if value.max() > 0:
            skw = kw.copy()
            
            # Set the colour for this bar
            if skw.get('color', None) is None and fillcolours is not None:
                skw['color'] = fillcolours[l1]
        
            if skw.get('edgecolor', None) is None and edgecolours is not None:
                skw['edgecolor'] = edgecolours[l1]
        
            # Check there are values to be plotted
            if vert:
                p = ax.bar(np.arange(num_bars_in_row), value, bottom=bottom, label=key, **skw)
            else:
                p = ax.barh(np.arange(num_bars_in_row), value, left=bottom, label=key, **skw)

            if datalabels:
                # The input data values are used for the labels
                if isinstance(data, pd.Series):
                    # A single value
                    input_values = data.loc[key]
                else:
                    input_values = data.values[l1]
                    
                # Draw bar labels for non-zero values in this stack
                if threshold is None:
                    # Draw axis labels  at the end of the bars
                    if isinstance(data, pd.Series):
                        bar_labels = [input_values]
                    else:
                        bar_labels = ['' if input_values[i] == 0 else str(input_values[i]) for i in range(data.shape[1])]

                    if num_bars_in_row > len(bar_labels):
                        bar_labels += [''] * (num_bars_in_row - len(bar_labels))

                    ax.bar_label(p, labels=bar_labels, padding=label_padding)
                else:
                    # Threshold defines whether an axis label will be drawn at the end of a bar or within it
                    # NB: There should be only one item in the stack for each stacked bar
                    if isinstance(data, pd.Series):
                        bar_labels = [input_values]
                        
                        if num_bars_in_row > len(bar_labels):
                            bar_labels += [''] * (num_bars_in_row - len(bar_labels))

                        if input_values >= threshold:
                            ax.bar_label(p, labels=bar_labels, label_type='center')
                        elif input_values > 0:
                            ax.bar_label(p, labels=bar_labels, padding=label_padding)
                    else:
                        bar_labels = ['' if value[i] < threshold else str(input_values[i]) for i in range(data.shape[1])]

                        if num_bars_in_row > len(bar_labels):
                            bar_labels += [''] * (num_bars_in_row - len(bar_labels))

                        ax.bar_label(p, labels=bar_labels, label_type='center')
                        bar_labels = ['' if value[i] >= threshold or value[i] == 0 else str(input_values[i]) for i in range(data.shape[1])]

                        if num_bars_in_row > len(bar_labels):
                            bar_labels += [''] * (num_bars_in_row - len(bar_labels))

                        ax.bar_label(p, labels=bar_labels, padding=label_padding)

            bottom += value

        # Increment colour number
        l1 += 1

    if legend:
        ax.legend(**legend_kw)

    if vert:
        # Set the default axis labels if none have been supplied as ax_kw
        if 'xlabel' not in axkwargs:
            axkwargs['xlabel'] = xlabel

        if 'ylabel' not in axkwargs:
            axkwargs['ylabel'] = name

        ax.set(**axkwargs)
        # Create the X ticks (including any dummy variables)
        ax.set_xticks(range(num_bars_in_row))
        ax.set_xticklabels(xlabels, rotation=xlabels_rotate)
        ax.set_xlim(-0.5, num_bars_in_row - 0.5)
        #ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        #ax.ticklabel_format(axis='x', style='plain')
    else:
        # Set the default axis labels if none have been supplied as ax_kw
        if 'xlabel' not in axkwargs:
            axkwargs['xlabel'] = name

        if 'ylabel' not in axkwargs:
            axkwargs['ylabel'] = xlabel

        ax.set(**axkwargs)

        if abs(xlabels_rotate) > 0.0:
            ax.tick_params('x', labelrotation=xlabels_rotate)

        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        # Create the Y ticks (including any dummy variables)
        ax.set_yticks(range(num_bars_in_row))
        ax.set_yticklabels(xlabels)
        ax.set_ylim(-0.5, num_bars_in_row - 0.5)

    if ax_input is None:
        _draw_fig(filename, overwrite)


def table(data, ax_input=None, include_index=False, auto_column_width=True, filename=None, overwrite=False, fig_kw={}, ax_kw={}, **kwargs):
    """
    Plot a table

    Parameters
    ----------
    data : series or dataframe
        Series containing the variable names (index) and data quality attribute to be plotted (e.g., number of missing values in each variable)
    ax_input: axis or None
        Matplotlib axis
    include_index: boolean
        Include the index in the table (default is False)
    auto_column_width: boolean
        Automatically set the widths of the table columns (default is True)
    filename : string
        None or a filename for the figure
    overwrite : boolean
        False (do not overwrite file; the default) or True (overwrite file if it exists)
    fig_kw : dictionary
        Keyword arguments for a Matplotlib Figure object
    ax_kw : dictionary
        Keyword arguments for a Matplotlib Axes object
    kwargs : dictionary
        Keyword arguments for a Matplotlib Axes.table object. A useful one is loc='center'

    Returns
    -------
    None.

    """
    axkwargs = ax_kw.copy()
    kw = kwargs.copy()
    #
    # Plot axis
    #
    if ax_input is None:
        fig, ax = plt.subplots()#layout='tight')
        fig.set(**fig_kw)
        fig.patch.set_visible(False)
    else:
        ax = ax_input

    #hide the axes
    ax.axis('off')
    ax.axis('tight')

    ax.set(**axkwargs)

    # Get the table's data
    if isinstance(data, pd.Series):
        if include_index:
            plotdata = pd.DataFrame(data).reset_index()
            plotdata.rename(columns={plotdata.columns[0]: 'Variable'}, inplace=True)
        else:
            plotdata = data
    else:
        # Dataframe
        if include_index:
            plotdata = data.reset_index()
            plotdata.rename(columns={plotdata.columns[0]: 'Variable'}, inplace=True)
        else:
            plotdata = data

    if 'cellText' not in kw:
        kw['cellText'] = plotdata.values

    if 'colLabels' not in kw:
        if isinstance(plotdata, pd.Series):
            kw['colLabels'] = [plotdata.name]
            print('SERIES', plotdata.name)
        else:
            kw['colLabels'] = plotdata.columns.tolist()

    #create table
    table = ax.table(**kw)

    # Options
    if auto_column_width:
        table.auto_set_column_width([i for i in range(len(kw['colLabels']))])

    #print(table.get_clip_box())

    if ax_input is None:
        _draw_fig(filename, overwrite)


def text(plotdata, number_of_variables_per_row=None, ax_input=None, legend=True, xlabels_rotate=0.0, filename=None, overwrite=False, fig_kw={}, ax_kw={}, **kwargs):
    """
    Plot text for each variable

    Parameters
    ----------
    plotdata : series
        Series containing the variable names (index) and data quality attribute to be plotted (e.g., number of missing values in each variable)
    number_of_variables_per_row : int
        None (plot all variables in one bar chart) or the number of variables to show in each row
    ax_input: axis or None
        Matplotlib axis
    legend: boolean
        True (add a legend, if a stacked bar chart is plotted) or False (no legend)
    xlabels_rotate : float
        Angle to rotate X axis labels by (only used if vert = True)
    filename : string
        None or a filename for the figure
    overwrite : boolean
        False (do not overwrite file; the default) or True (overwrite file if it exists)
    fig_kw : dictionary
        Keyword arguments for a Matplotlib Figure object
    ax_kw : dictionary
        Keyword arguments for a Matplotlib Axes object
    kwargs : dictionary
        Keyword arguments for a Matplotlib Axes.bar object

    Returns
    -------
    None.

    """
    axkwargs = ax_kw.copy()
    #
    # Plot axis
    #
    if ax_input is None:
        fig, ax = plt.subplots()
        fig.set(**fig_kw)
    else:
        ax = ax_input

    # The index contains the variable names
    xlabels = plotdata.index.to_list()

    # The index contains the variable names
    xlabels = plotdata.index.to_list()
    num_bars_in_row = len(plotdata)
    # Hide the X ticks and labels
    ax.set_xticklabels([])
    ax.tick_params('x', bottom=False)

    ax.set_yticks(range(num_bars_in_row))
    ax.set_yticklabels(xlabels)
    ax.set_ylim(-0.5, num_bars_in_row - 0.5)

    if 'ylabel' not in axkwargs:
        axkwargs['ylabel'] = 'Variable'

    ax.set(**axkwargs)

    for l1 in range(len(plotdata)):
        ax.annotate(plotdata.iloc[l1], (0.03, l1))

    if ax_input is None:
        _draw_fig(filename, overwrite)


def boxplot(data, number_of_variables_per_row=None, ax_input=None, vert=True, xlabels_rotate=0.0, filename=None, overwrite=False, fig_kw={}, ax_kw={}, **kwargs):
    """
    Create a box plot to show the distribution of numerical data.

    Parameters
    ----------
    data : series
        Series containing the variable names (index) and data quality attribute to be plotted (e.g., number of missing values in each variable)
    number_of_variables_per_row : int
        None (plot all variables in one bar chart) or the number of variables to show in each row
    ax_input: axis or None
        Matplotlib axis
    vert: boolean
        True (vertical bars; the default) or False (horizontal)
    xlabels_rotate : float
        Angle to rotate X axis labels by (only used if vert = True)
    filename : string
        None or a filename for the figure
    overwrite : boolean
        False (do not overwrite file; the default) or True (overwrite file if it exists)
    fig_kw : dictionary
        Keyword arguments for a Matplotlib Figure object
    ax_kw : dictionary
        Keyword arguments for a Matplotlib Axes object
    kwargs : dictionary
        Keyword arguments for a Matplotlib Axes.bxp object

    Returns
    -------
    None.

    """
    axkwargs = ax_kw.copy()
    #
    # Plot axis
    #
    if ax_input is None:
        fig, ax = plt.subplots()
        fig.set(**fig_kw)
    else:
        ax = ax_input
    #
    # The index contains the variable names
    #
    boxes = []
    minmax = [None, None]

    for index, value in data.items():
        # Convert value (a comma-separated string) into a list
        strvals = value.split(',')

        if len(strvals) == 5:
            try:
                # Check whether the first value is numeric
                float(strvals[0])
                numeric_boxplot = True
            except:
                pass
                numeric_boxplot = False

            # The number of values is correct for a box plot
            if numeric_boxplot:
                # Assume the values are numbers, so convert the values to floats
                vals = list(map(float, strvals))
                # Append this box plot's parameters to the list
                boxes.append({'label': index, 'whislo': vals[0], 'q1': vals[1], 'med': vals[2], 'q3': vals[3], 'whishi': vals[4], 'fliers': []})
            else:
                # Assume the values are dates or times
                # Convert the values to a datetime series
                vals = pd.to_datetime(pd.Series(strvals)).values
                # Append this box plot's parameters to the list
                boxes.append({'label': index, 'whislo': vals[0], 'q1': vals[1], 'med': vals[2], 'q3': vals[3], 'whishi': vals[4], 'fliers': []})

                # Min and max date/time
                if minmax[0] is None:
                    minmax[0] = vals[0]
                    minmax[1] = vals[4]
                else:
                    minmax[0] = min(minmax[0], vals[0])
                    minmax[1] = min(minmax[1], vals[4])

    # Should the values be plotted as times?
    try:
        # Check if max - min < 24 hours
        time_axis = True if (minmax[1] - minmax[0]) / np.timedelta64(1,'h') <= 24 else False
    except:
        pass
        time_axis = False
    #
    # Plot the data
    #
    try:
        num_bars_in_row = max(len(boxes), number_of_variables_per_row)
    except:
        pass
        num_bars_in_row = len(boxes)

    if len(boxes) == 0:
        print('** WARNING ** vizdataquality, plot.py, boxplot(): No variables to be plotted.')
    elif vert:
        # Vertical boxes
        ax.bxp(boxes, showfliers=False, vert=vert)
        # Set the default axis labels if none have been supplied as ax_kw
        if 'xlabel' not in axkwargs:
            axkwargs['xlabel'] = 'Variable'

        if 'ylabel' not in axkwargs:
            axkwargs['ylabel'] = data.name

        ax.set(**axkwargs)
        # Set the X axis limit (including any dummy variables)
        ax.set_xlim(0.5, num_bars_in_row + 0.5)

        if abs(xlabels_rotate) > 0.0:
            ax.tick_params('x', labelrotation=xlabels_rotate)

        if time_axis:
            ax.yaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M:%S'))
    else:
        # Horizontal boxes
        ax.bxp(boxes, showfliers=False, vert=vert)
        # Set the default axis labels if none have been supplied as ax_kw
        if 'xlabel' not in axkwargs:
            axkwargs['xlabel'] = data.name

        if 'ylabel' not in axkwargs:
            axkwargs['ylabel'] = 'Variable'

        ax.set(**axkwargs)
        # Set the Y axis limit (including any dummy variables)
        ax.set_ylim(0.5, num_bars_in_row + 0.5)

        if abs(xlabels_rotate) > 0.0:
            ax.tick_params('x', labelrotation=xlabels_rotate)

        if time_axis:
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M:%S'))

    if ax_input is None:
        _draw_fig(filename, overwrite)


def violinplot(data, number_of_variables_per_row=None, ax_input=None, vert=True, xlabels_rotate=0.0, filename=None, overwrite=False, fig_kw={}, ax_kw={}, **kwargs):
    """
    Create a violin plot to show the distribution of numerical data.

    Parameters
    ----------
    data : series or dataframe
        The values to be plotted (each column is plotted separately)
    number_of_variables_per_row : int
        None (plot all variables in one bar chart) or the number of variables to show in each row
    ax_input: axis or None
        Matplotlib axis
    vert: boolean
        True (vertical bars; the default) or False (horizontal)
    xlabels_rotate : float
        Angle to rotate X axis labels by (only used if vert = True)
    filename : string
        None or a filename for the figure
    overwrite : boolean
        False (do not overwrite file; the default) or True (overwrite file if it exists)
    fig_kw : dictionary
        Keyword arguments for a Matplotlib Figure object
    ax_kw : dictionary
        Keyword arguments for a Matplotlib Axes object
    kwargs : dictionary
        Keyword arguments for a Matplotlib violinplot object

    Returns
    -------
    None.

    """
    axkwargs = ax_kw.copy()
    #
    # Plot axis
    #
    if ax_input is None:
        fig, ax = plt.subplots()
        fig.set(**fig_kw)
    else:
        ax = ax_input

    # Remove any missing values
    if isinstance(data, pd.Series):
        plotdata = data.dropna().values
        columns = [data.name]
    else:
        plotdata = [data[v].dropna().values for v in data.columns]
        columns = data.columns.tolist()

    # Create the violoin plots
    ax.violinplot(plotdata, vert=vert, **kwargs)

    if vert:
        # Set the default axis labels if none have been supplied as ax_kw
        if 'xlabel' not in axkwargs:
            axkwargs['xlabel'] = 'Variable'

        if 'ylabel' not in axkwargs:
            axkwargs['ylabel'] = 'Value'

        ax.set(**axkwargs)

        #if abs(xlabels_rotate) > 0.0:
        #    ax.tick_params('x', labelrotation=xlabels_rotate)

        ax.set_xticks(range(1, len(columns)+1))
        ax.set_xticklabels(columns, rotation=xlabels_rotate)
    else:
        # Set the default axis labels if none have been supplied as ax_kw
        if 'xlabel' not in axkwargs:
            axkwargs['xlabel'] = 'Value'

        if 'ylabel' not in axkwargs:
            axkwargs['ylabel'] = 'Variable'

        ax.set(**axkwargs)

        if abs(xlabels_rotate) > 0.0:
            ax.tick_params('x', labelrotation=xlabels_rotate)

        ax.set_yticks(range(1, len(columns)+1))
        ax.set_yticklabels(columns)#, rotation=xlabels_rotate)

    if ax_input is None:
        _draw_fig(filename, overwrite)


def dot_whisker(data, number_of_variables_per_row=None, ax_input=None, vert=True, xlabels_rotate=0.0, filename=None, overwrite=False, fig_kw={}, ax_kw={}, **kwargs):
    """
    Create a dot-or-whisker plot (e.g., to show value lengths for each variable)

    Parameters
    ----------
    data : series
        Series containing the variable names (index) and data quality attribute to be plotted (e.g., number of missing values in each variable)
    number_of_variables_per_row : int
        None (plot all variables in one bar chart) or the number of variables to show in each row
    ax_input: axis or None
        Matplotlib axis
    vert: boolean
        True (vertical bars; the default) or False (horizontal)
    xlabels_rotate : float
        Angle to rotate X axis labels by (only used if vert = True)
    filename : string
        None or a filename for the figure
    overwrite : boolean
        False (do not overwrite file; the default) or True (overwrite file if it exists)
    fig_kw : dictionary
        Keyword arguments for a Matplotlib Figure object
    ax_kw : dictionary
        Keyword arguments for a Matplotlib Axes object
    kwargs : dictionary
        Keyword arguments for a Matplotlib Axes.scatter or Axes.errorbar object

    Returns
    -------
    None.

    """
    axkwargs = ax_kw.copy()
    kw = kwargs.copy()

    if 'color' not in kw:
        # Apply the 2nd colour in the colourmap
        kw['color'] = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
    #
    # Plot axis
    #
    if ax_input is None:
        fig, ax = plt.subplots()
        fig.set(**fig_kw)
    else:
        ax = ax_input
    #
    # Calculate the dot or whisker data that will be plotted for each variable
    #
    dotx = []
    doty = []
    whiskerx = []
    whiskery = []
    whiskererr = []
    pos = 0

    for index, value in data.items():
        # The index is the variable name. Convert value (a comma-separated string) into a list
        strvals = value.split(',')

        if len(strvals) == 2:
            #print(strvals)
            # The number of values is correct for a dot or whisker plot
            # Convert the values to floats
            vals = list(map(float, strvals))
            # Append this variable's parameters to the list
            if abs(vals[0] - vals[1]) <= 0.0:
                dotx.append(pos)
                doty.append(vals[0])
            else:
                whiskerx.append(pos)
                whiskery.append(0.5 * sum(vals))
                whiskererr.append(abs(vals[0] - vals[1]) * 0.5)

        pos = pos + 1


    if len(dotx) == 0 and len(whiskerx) == 0:
        print('** WARNING ** vizdataquality, plot.py, dot_whisker(): No variables to be plotted.')
    else:
        # The index contains the variable names
        xlabels = data.index.to_list()

        try:
            # If necessary, extend the arrays to accommodate any dummy variables at the end of the plot
            num_bars_in_row = max(len(data), number_of_variables_per_row)
            xlabels = xlabels + [''] * (num_bars_in_row - len(data))
        except:
            # Just plot the variables
            pass
            num_bars_in_row = len(data)

        if vert:
            if len(dotx) > 0:
                ax.scatter(dotx, doty, **kw)

            if len(whiskerx) > 0:
                ax.errorbar(whiskerx, whiskery, yerr=whiskererr, fmt='none', capsize=4, **kw)

            # Set the default axis labels if none have been supplied as ax_kw
            if 'xlabel' not in axkwargs:
                axkwargs['xlabel'] = 'Variable'

            if 'ylabel' not in axkwargs:
                axkwargs['ylabel'] = data.name

            # For multiplot(), this prevents Matplotlib from autoscaling the shared Y axis
            #if 'ylim' not in axkwargs:
            #    # By default, start the Y axis at the origin
            #    axkwargs['ylim'] = 0

            ax.set(**axkwargs)
            # Create the X ticks (including any dummy variables)
            ax.set_xticks(range(num_bars_in_row))
            ax.set_xticklabels(xlabels, rotation=xlabels_rotate)
            ax.set_xlim(-0.5, num_bars_in_row - 0.5)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        else:
            if len(dotx) > 0:
                ax.scatter(doty, dotx, **kw)

            if len(whiskerx) > 0:
                ax.errorbar(whiskery, whiskerx, xerr=whiskererr, fmt='none', capsize=4, **kw)

            # Set the default axis labels if none have been supplied as ax_kw
            if 'xlabel' not in axkwargs:
                axkwargs['xlabel'] = data.name

            # For multiplot(), this prevents Matplotlib from autoscaling the shared Y axis
            #if 'xlim' not in axkwargs:
            #    # By default, start the X axis at the origin
            #    axkwargs['xlim'] = 0

            if 'ylabel' not in axkwargs:
                axkwargs['ylabel'] = 'Variable'

            ax.set(**axkwargs)

            if abs(xlabels_rotate) > 0.0:
                ax.tick_params('x', labelrotation=xlabels_rotate)

            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            # Create the Y ticks (including any dummy variables)
            ax.set_yticks(range(num_bars_in_row))
            ax.set_yticklabels(xlabels)
            ax.set_ylim(-0.5, num_bars_in_row - 0.5)

    if ax_input is None:
        _draw_fig(filename, overwrite)


def lollipop(data, number_of_variables_per_row=None, ax_input=None, vert=True, xlabels_rotate=0.0, datalabels=False, continuous_value_axis=True, filename=None, overwrite=False, fig_kw={}, ax_kw={}, **kwargs):
    """
    Create a lollipop plot (e.g., to show value counts for a variable)

    Parameters
    ----------
    data : series
        Value counts for a variable names.
    number_of_variables_per_row : int
        None (plot all variables in one bar chart) or the number of variables to show in each row. The default is None.
    ax_input: axis or None
        Matplotlib axis. The default is None.
    vert: boolean
        True (vertical bars) or False (horizontal). The default is True.
    xlabels_rotate : float
        Angle to rotate X axis labels by (only used if vert=True). The default is 0.0.
    datalabels : boolean
        Label each data point. The default is False.
    continuous_value_axis : boolean
        Plot numerical/datetime values on a continuous axis to show any gaps in values. The default is True.
    filename : string
        Filename for the figure. The default is None.
    overwrite : boolean
        False (do not overwrite file) or True (overwrite file if it exists). The default is False.
    fig_kw : dictionary
        Keyword arguments for a Matplotlib Figure object. The default is {}.
    ax_kw : dictionary
        Keyword arguments for a Matplotlib Axes object. The default is {}.
    kwargs : dictionary
        Keyword arguments for a Matplotlib Axes.scatter and Axes.plot objects

    Returns
    -------
    None.

    """
    axkwargs = ax_kw.copy()
    kw = kwargs.copy()

    if 'color' not in kw:
        # Apply the 3rd colour in the colourmap
        kw['color'] = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
    #
    # Plot axis
    #
    if ax_input is None:
        fig, ax = plt.subplots()
        fig.set(**fig_kw)
    else:
        ax = ax_input

    if continuous_value_axis:
        # Determine whether the index is just numerical values
        numerical_values = True if all(type(e) in (int, float) for e in data.index) else False
    else:
        numerical_values = False

    if numerical_values:
        plotdata = data
    else:
        # Sort the series so that the values are plotted in order
        plotdata = data.sort_index()#ascending=False)

    #print('numerical_values', numerical_values)
    #print(plotdata)
    num_values = len(plotdata)
    # The index contains the variable names
    dotx = []
    doty = []
    pos = 0

    for index, value in plotdata.items():

        if numerical_values:
            dotx.append(index)
        else:
            dotx.append(pos)

        doty.append(value)
        pos = pos + 1

    if len(dotx) > 0:
        # The index contains the variable names
        xlabels = plotdata.index.to_list()

        try:
            # If necessary, extend the arrays to accommodate any dummy variables at the end of the plot
            num_values = max(len(data), number_of_variables_per_row)
            xlabels = xlabels + [''] * (num_values - len(data))
        except:
            # Just plot the variables
            pass
            num_values = len(data)

        if vert:
            ax.scatter(dotx, doty, **kw)

            for l1 in range(len(dotx)):
                ax.plot([dotx[l1], dotx[l1]], [0, doty[l1]], **kw)

            # Set the default axis labels if none have been supplied as ax_kw
            if 'xlabel' not in axkwargs:
                axkwargs['xlabel'] = 'Value'

            if 'ylabel' not in axkwargs:
                axkwargs['ylabel'] = 'Count'#plotdata.name.capitalize()

            ax.set(**axkwargs)

            if datalabels:
                # This is a hack, but it works for default-sized markers
                limits = ax.get_ylim()
                offset = (limits[1] - limits[0]) / 50.0

                for l1 in range(len(dotx)):
                    ax.text(dotx[l1], doty[l1] + offset, str(doty[l1]), horizontalalignment='center', verticalalignment='bottom')

            if numerical_values:
                ax.set_xlim(plotdata.index.min()-0.5, plotdata.index.max()+0.5)
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            else:
                # Just plot the series values on the X axis
                #ax.set_xlim(-1, len(data))
                ax.set_xlim(-0.5, num_values - 0.5)
                # Create the X ticks (including any dummy variables)
                ax.set_xticks(range(num_values))
                ax.set_xticklabels(xlabels, rotation=xlabels_rotate)

            # Commented out (see below)
            #ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        else:
            ax.scatter(doty, dotx, **kw)

            for l1 in range(len(dotx)):
                ax.plot([0, doty[l1]], [dotx[l1], dotx[l1]], **kw)

            # Set the default axis labels if none have been supplied as ax_kw
            if 'xlabel' not in axkwargs:
                axkwargs['xlabel'] = 'Count'#plotdata.name.capitalize()

            if 'ylabel' not in axkwargs:
                axkwargs['ylabel'] = 'Value'

            ax.set(**axkwargs)

            if datalabels:
                # This is a hack, but it works for default-sized markers
                limits = ax.get_xlim()
                offset = (limits[1] - limits[0]) / 50.0

                for l1 in range(len(dotx)):
                    ax.text(doty[l1] + offset, dotx[l1], str(doty[l1]), horizontalalignment='left', verticalalignment='center')

            if numerical_values:
                ax.set_ylim(plotdata.index.min()-0.5, plotdata.index.max()+0.5)
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            else:
                # Just plot the series values on the X axis
                #ax.set_ylim(-1, len(data))
                ax.set_ylim(-0.5, num_values - 0.5)
                # Create the X ticks (including any dummy variables)
                ax.set_yticks(range(num_values))
                ax.set_yticklabels(xlabels, rotation=xlabels_rotate)

            # NB: AT one point this was commented out because it causes a crash for some plots if axkwargs contains 'xticks'
            #
            # Now commented out because it sometimes causes too many tick labels
            #ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

            #if abs(xlabels_rotate) > 0.0:
            #    ax.tick_params('x', labelrotation=xlabels_rotate)

    else:
        print('** WARNING ** vizdataquality, plot.py, lollipop(): No variables to be plotted.')

    if ax_input is None:
        _draw_fig(filename, overwrite)


def datetime_counts(data, component='raw data', gap_threshold=None, show_gaps=True, ax_input=None, xlabels_rotate=0.0, filename=None, overwrite=False, fig_kw={}, ax_kw={}, **kwargs):
    """
    Plot the overall distribution of a datetime variable, or the distribution of a specific component (e.g., month).

    Parameters
    ----------
    data : series
        Values of a variable.
    component : string
        Component to plot ('year', 'month', 'dayofweek', 'hour', 'minute' or 'second'; case independent) or 'raw data' (default)
    gap_threshold: None, int or datetime
        None (threshold will be based on the component of the data; the default) or value (threshold to use). Only used if component is specified.
    show_gaps: boolean
        True (the default) or False (draw lines across gaps). Only used if component is specified.
    ax_input: axis or None
        Matplotlib axis
    xlabels_rotate : float
        Angle to rotate X axis labels by (only used if vert = True)
    filename : string
        None or a filename for the figure
    overwrite : boolean
        False (do not overwrite file; the default) or True (overwrite file if it exists)
    fig_kw : dictionary
        Keyword arguments for a Matplotlib Figure object
    ax_kw : dictionary
        Keyword arguments for a Matplotlib Axes object
    kwargs : dictionary
        Keyword arguments for a Matplotlib Axes.plot object

    Returns
    -------
    None.

    """
    if len(data) == 0:
        print('** WARNING ** vizdataquality, plot.py, datetime_counts(): There is no data to be plotted.')
    elif component.lower() in ['raw data', 'year', 'month', 'dayofweek', 'hour', 'minute', 'second']:
        axkwargs = ax_kw.copy()
        kw = kwargs.copy()

        if 'color' not in kw:
            # Use the first color
            kw['color'] = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

        # Calculate the number of times each value occurs, sort them and store the result in a data frame
        #value_counts = data.value_counts().to_frame().reset_index().rename(columns={'index': 'Value', data.name: 'Count'})
        #value_counts = data.value_counts().sort_index().to_frame().reset_index().rename(columns={data.name: 'Value', 'count': 'Count'})
        value_counts = data.value_counts().sort_index().to_frame().reset_index()
        cols = value_counts.columns
        # The dataframe's first column contains the values and the second column contains the count
        value_counts.rename(columns={cols[0]: 'Value', cols[1]: 'Count'}, inplace=True)
        gthresh = 1 if gap_threshold is None else gap_threshold
        xticklabels = None
        xinterval = None
        #xinteger_component = False

        comp = component.lower()
        xlabel = comp[0].upper() + comp[1:]

        if comp == 'raw data':
            # The width of the plot in pixels
            num_pixels = plt.rcParams['figure.figsize'][0] * plt.rcParams['figure.dpi']
            # Max number of unique values to be plotted
            max_unique = int(num_pixels / 5)

            if len(value_counts) > max_unique:
                # Try to find a coarser datetime granularity
                granularity = None

                for u in ['Y', 'M', 'D', 'h', 'm', 's']:
                    s = value_counts['Value'].apply(lambda x: np.datetime64(x, u))
                    #if len(value_counts['Value'].apply(lambda x: np.datetime64(x, u)).unique()) <= max_unique:
                    if len(s.unique()) <= max_unique:
                        value_counts['Value2'] = s
                        granularity = u
                    else:
                        # Too fine grained
                        break

                if granularity is not None:
                    #value_counts['Value2'] = value_counts['Value'].apply(lambda x: np.datetime64(x, granularity))
                    grouped = value_counts[['Value2', 'Count']].groupby('Value2', sort=True)['Count'].sum().reset_index()
                    value_counts = grouped.rename(columns={'Value2': 'Value'})

            xmin = value_counts['Value'].min()
            xmax = value_counts['Value'].max()
        else:
            # Plot a component (e.g., 'year')
            xmin = 0

            if comp == 'year':
                value_counts['Part'] = value_counts['Value'].dt.year
                xmin = value_counts['Part'].min()
                xmax = value_counts['Part'].max()

                if xmax - xmin <= 0.0:
                    # The data is only for a single year, so make the X axis span 3 years
                    #xticklabels = [xmin]
                    xmin -= 0.5
                    xmax += 0.5
                #xinteger_component= True
            elif comp == 'month':
                # Months are 1 (Jan) - 12 (Dec)
                value_counts['Part'] = value_counts['Value'].dt.month
                xmin = 1
                xmax = 12
                xticklabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            elif comp == 'dayofweek':
                # Days of the week are 0 (Mon) - 6 (Sun)
                value_counts['Part'] = value_counts['Value'].dt.dayofweek
                xmax = 6
                xlabel = 'Day of the week'
                xticklabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            elif comp == 'hour':
                value_counts['Part'] = value_counts['Value'].dt.hour
                xmax = 24#23
                xinterval = 6
                #xticklabels = [i for i in range(0, xmax+1, 6)]
                #xinteger_component= True
            elif comp == 'minute':
                value_counts['Part'] = value_counts['Value'].dt.minute
                xmax = 60#59
                xinterval = 15
                #xinteger_component= True
            elif comp == 'second':
                value_counts['Part'] = value_counts['Value'].dt.second
                xmax = 60#59
                xinterval = 15
                #xinteger_component= True

            # Adjust the min/max to leave a space at both ends of the X axis
            #xmin -= 0.5
            #xmax += 0.5

            # Calculate the number of times each value of Part occurs
            groups = value_counts.groupby('Part')['Count'].sum().sort_index()

        # Set the default axis labels if none have been supplied as ax_kw
        if 'xlabel' not in axkwargs:
            axkwargs['xlabel'] = xlabel
        #
        # Plot axis
        #
        if ax_input is None:
            fig, ax = plt.subplots()
            fig.set(**fig_kw)
            fig.autofmt_xdate()
        else:
            ax = ax_input

        if 'xlim' not in axkwargs:
            axkwargs['xlim'] = (xmin, xmax)

        if 'ylim' not in axkwargs:
            axkwargs['ylim'] = 0

        if 'ylabel' not in axkwargs:
            axkwargs['ylabel'] = 'Count'

        if comp == 'raw data' or len(groups) > 0:

            if comp != 'raw data' and show_gaps:
                # The first item
                x = [groups.index[0]]
                y = [groups.values[0]]

                for index, value in groups.iloc[1:].items():
                    # Iterate over the other items
                    if index <= x[-1] + gthresh:
                        # This item is part of a continuous sequence of months, etc.
                        x.append(index)
                        y.append(value)
                    else:
                        if len(x) == 1:
                            # Plot a point (the last item was discrete)
                            ax.scatter(x, y, **kw)
                        elif len(x) >= 2:
                            # Plot a polyline (the last item(s) were in a sequence)
                            ax.plot(x, y, **kw)

                        x = [index]
                        y = [value]

                # Plot any remaining data
                if len(x) == 1:
                    # Plot a point
                    ax.scatter(x, y, **kw)
                elif len(x) >= 2:
                    # Plot a polyline
                    ax.plot(x, y, **kw)
            else:
                # Plot the value counts without showing any gaps
                if comp == 'raw data':
                    x = value_counts['Value']
                    y = value_counts['Count']
                else:
                    x = groups.index
                    y = groups.values

                if len(x) == 1:
                    # A single point
                    ax.scatter(x, y, **kw)
                else:
                    # Line chart
                    ax.plot(x, y, **kw)

            ax.set(**axkwargs)

            if xticklabels is None:
                #ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
                if True or abs(xlabels_rotate) > 0.0:
                    # Set to true because Matplotlib otherwise rotates the labels automatically
                    ax.tick_params('x', labelrotation=xlabels_rotate)

                if comp == 'year':
                    # Now commented out because it sometimes causes too many tick labels
                    #ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

                    # If there is a small range of years then Matplotlib's tick labels may include decimals
                    xlabs = ax.get_xticklabels()
                    new_ticks = []
                    new_labels = []
                    # Loop over the tick labels, appending only integer labels
                    for xl in xlabs:
                        try:
                            fval = float(xl.get_text())
                            ival = int(fval)
                            if abs(fval - ival) <= 0:
                                # Integer label
                                new_ticks.append(ival)
                                new_labels.append(str(ival))
                        except:
                            raise

                    # Specify the new, integer-only labels
                    ax.set_xticks(new_ticks, new_labels, rotation=xlabels_rotate)
                    # Commented out because all the labels will now be integers
                    #ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x))))

                if xinterval is not None:
                    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xinterval))
            else:
                #ax.set_xticks(range(len(xticklabels)))
                #ax.set_xticklabels(xticklabels, rotation=xlabels_rotate)
                if comp == 'year' and len(xticklabels) == 1:
                    ax.set_xticks(range(xticklabels[0], xticklabels[0]+1), xticklabels, rotation=xlabels_rotate)
                else:
                    # xmin used in range because months start from 1, whereas days of the week start from 0
                    ax.set_xticks(range(xmin, xmin+len(xticklabels)), xticklabels, rotation=xlabels_rotate)

            # Now commented out because it sometimes causes too many tick labels
            #ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        else:
            print('** WARNING ** vizdataquality, plot.py, datetime_counts(): No variables to be plotted.')
    else:
        print('** WARNING ** vizdataquality, plot.py, datetime_counts(): Invalid component:', component)

    if ax_input is None:
        _draw_fig(filename, overwrite)
