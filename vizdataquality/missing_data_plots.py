# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:54:11 2022

Functions for plotting visualizations of missing data.

Internal functions are prefixed by '_'. The functions are grouped as follows:
    Functions for summary plots
    Functions for purity plots
    Explanation graph functions

@author: Roy Ruddle, University of Leeds
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from vizdataquality import plot as vdqp
from vizdataquality import missing_data_utils as mdu

# =============================================================================
# Colours
# =============================================================================
def get_default_colours(name):
    """
    Get default colours used for a plot or part of a plot.

    Parameters
    ----------
    name : str
        Name of a plot or part of a plot. Valid values are 'summary missingness values', 'summary missingness variables', 'purity heatmap', 'purity heatmap disjoint', 'purity heatmap block', 'purity heatmap monotone' and 'explanation graph summary'

    Returns
    -------
    list
        The colours for the plot.

    """
    if name == 'summary missingness values':
        colours = [[188/255.0, 189/255.0, 34/255.0], [219/255.0, 219/255.0, 141/255.0], [1, 1, 1]]
    elif name == 'summary missingness variables':
        colours = [[23/255.0, 190/255.0, 207/255.0], [158/255.0, 218/255.0, 229/255.0], [1, 1, 1]]
    elif name == 'purity heatmap':
        colours = [mpl.colormaps['tab20'].colors[i] for i in [1, 0, 3, 2, 11, 10]]
    elif name == 'purity heatmap disjoint':
        colours = [mpl.colormaps['tab20'].colors[i] for i in [1, 0]]
    elif name == 'purity heatmap block':
        colours = [mpl.colormaps['tab20'].colors[i] for i in [3, 2]]
    elif name == 'purity heatmap monotone':
        colours = [mpl.colormaps['tab20'].colors[i] for i in [11, 10]]
    elif name == 'explanation graph summary':
        colours = [[148/255.0, 103/255.0, 189/255.0], [197/255.0, 176/255.0, 213/255.0], [1,1,1,1]]
    else:
        colours = None
        
    return colours

                   
# =============================================================================
# Functions for summary plots
# =============================================================================
def plot_summary_missingness(num_records, num_missing, num_rows=1, perceptual_threshold=0.05, ax_input=None, vert=True, filename=None, overwrite=False, fig_kw={}, ax_kw={}, legend_kw={}, **kwargs):
    """
    Plot an overview of a dataset's missingness, using stacked bar charts of the number of values and variables.

    Parameters
    ----------
    num_records : int
        The number of records in the dataset
    num_missing : series
        The number of missing values in each variable
    num_rows : int, optional
        The number of rows of plots, which arranges the plots beside or above each other. The default is 1 (beside each other).
    perceptual_threshold : float, optional
        Perceptual discontinuity threshold (0.0 - 1.0) or None. The default is 0.05.
    ax_input: axis or None, optional
        Matplotlib axis. The default is None.
    vert: boolean, optional
        True (vertical bars; the default) or False (horizontal). The default is True.
    filename : string, optional
        None or a filename for the figure. The default is None.
    overwrite : boolean, optional
        False (do not overwrite file; the default) or True (overwrite file if it exists). The default is False.
    fig_kw : dictionary, optional
        Keyword arguments for a Matplotlib Figure object. The default is an empty dictionary.
    ax_kw : dictionary, optional
        Keyword arguments for a Matplotlib Axes object. The default is an empty dictionary.
    legend_kw : dictionary, optional
        Keyword arguments for a Matplotlib legend. The default is an empty dictionary.
    kwargs :
        Keyword arguments for a Matplotlib Axes.bar object

    Returns
    -------
    None.

    """
    colours = [get_default_colours('summary missingness values')] + [get_default_colours('summary missingness variables')]
    clists = []
    elist = [[0, 0, 0]] * 3
        
    if ax_input is None:
        nrows = num_rows
        ncols = int(2 / nrows)
            
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set(**fig_kw)
    else:
        ax = ax_input

    num_empty_variables = len(list(filter(lambda x: x == num_records, num_missing)))
    # List of the number of missing values in partially complete variables
    partly_missing_variables = list(filter(lambda x: x > 0 and x < num_records, num_missing))
    
    for l1 in range(nrows*ncols):
        ax = axs[l1]

        axkwargs = ax_kw.copy()
        # Set default values if they aren't defined in ax_kw
        key = 'xlabel' if vert else 'ylabel'
        axkwargs[key] = ax_kw.get(key, '')
        
        if l1 == 0:
            key = 'ylabel' if vert else 'xlabel'
            axkwargs[key] = ax_kw.get(key, 'Number of values')
            key = 'ylim' if vert else 'xlim'
            axkwargs[key] = ax_kw.get(key, (0, num_records * len(num_missing)))
            
            categories = []
            counts = []
            cl = []
            # Number of values that are present in the data
            num_present = num_records * len(num_missing) - sum(num_missing)
            
            if num_present > 0:
                categories.append('Present values')
                counts.append(num_present)
                cl.append(colours[l1][0])
                
            if len(partly_missing_variables) > 0:
                categories.append('Missing from partially complete variables')
                counts.append(sum(partly_missing_variables))
                cl.append(colours[l1][1])
            
            if num_empty_variables > 0:
                categories.append('Missing from empty variables')
                counts.append(num_records * num_empty_variables)
                cl.append(colours[l1][2])
                
            # Append the colour list for this plot
            clists.append(cl)
        else:
            key = 'ylabel' if vert else 'xlabel'
            axkwargs[key] = ax_kw.get(key, 'Number of variables')
            key = 'ylim' if vert else 'xlim'
            axkwargs[key] = ax_kw.get(key, (0, len(num_missing)))

            categories = []
            counts = []
            cl = []
            num_complete_variables = list(filter(lambda x: x == 0, num_missing))
            
            if len(num_complete_variables) > 0:
                categories.append('Complete variables')
                counts.append(len(num_complete_variables))
                cl.append(colours[l1][0])
                
            if len(partly_missing_variables) > 0:
                categories.append('Partly missing variables')
                counts.append(len(partly_missing_variables))
                cl.append(colours[l1][1])
            
            if num_empty_variables > 0:
                categories.append('Empty variables')
                counts.append(num_empty_variables)
                cl.append(colours[l1][2])
                
            # Append the colour list for this plot
            clists.append(cl)
                
        # Create the series to be plotted
        stacked_series = pd.Series(counts, index=categories)

        vdqp.stacked_bar(stacked_series, perceptual_threshold=0.05, ax_input=ax, vert=vert, clist=clists[l1], elist=elist, ax_kw=axkwargs, legend_kw=legend_kw, **kwargs)

    if ax_input is None:
        vdqp._draw_fig(filename, overwrite)


# =============================================================================
# Visualizations that are equivalent to some provided by setvis
# =============================================================================
def plot_intersection_heatmap(data, row_col_order=None, ax_input=None, transpose=True, xlabels_rotate=0.0, ylabels_rotate=0.0, datalabels=False, grid_spacing=None, filename=None, overwrite=False, fig_kw={}, ax_kw={}, cbar_kw={}, **kwargs):
    """
    Plot a heatmap of showing number of times that different combinations of variables are missing.

    Parameters
    ----------
    data : data frame
        Heat map data (one row per intersection and one column per variable; value is cardinality of the intersection or np.nan (variable is not part of the intersection)).
    row_col_order : list, optional
        The order of columns in the heatmap. The default is None (they are plotted alphabetically).
    ax_input: axis or None, optional
        Matplotlib axis. The default is None.
    transpose: boolean, optional
        True (intersections on X axis; variables on Y axis) or False (the other way around). The default is True.
    xlabels_rotate : float, optional
        Angle to rotate X axis labels by. The default is 0.0.
    ylabels_rotate : float, optional
        Angle to rotate Y axis labels by. The default is 0.0.
    datalabels : boolean, optional
        Label each heatmap cell. The default is False.
    grid_spacing : int, optional
        Draw gridlines on the heatmap every N rows/columns. The default is None.
    filename : string, optional
        None or a filename for the figure. The default is None.
    overwrite : boolean, optional
        False (do not overwrite file; the default) or True (overwrite file if it exists). The default is False.
    fig_kw : dictionary, optional
        Keyword arguments for a Matplotlib Figure object. The default is an empty dictionary.
    ax_kw : dictionary, optional
        Keyword arguments for a Matplotlib Axes object. The default is an empty dictionary.
    cbar_kw : dictionary, optional
        Keyword arguments for a Matplotlib Colorbar object. The default is an empty dictionary.
    kwargs :
        Keyword arguments for a Matplotlib Axes.imshow object

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

    variables = data.columns.tolist()

    if transpose:
        plotdata = data.transpose()
    else:
        plotdata = data
        
    im = ax.imshow(plotdata, **kwargs)
    
    # Colourbar
    ax.figure.colorbar(im, ax=ax, **cbar_kw)

    ax.set(**axkwargs)
        
    # Tick labels (major ticks naming each variable)
    if transpose:
        ax.set_yticks(np.arange(plotdata.shape[0]))
        ax.set_yticklabels(variables, rotation=ylabels_rotate)
    else:
        ax.set_xticks(np.arange(plotdata.shape[1]))
        ax.set_xticklabels(variables, rotation=xlabels_rotate)
        
    if isinstance(grid_spacing, int):
        # Gridlines based on minor ticks, every Nth row/column
        ax.set_xticks(np.arange(-0.5, plotdata.shape[1], grid_spacing), minor=True)
        ax.set_yticks(np.arange(-0.5, plotdata.shape[0], grid_spacing), minor=True)
        ax.grid(visible=True, which='minor', color='black')#, linestyle='-', linewidth=2)
        # Remove minor ticks
        ax.tick_params(which='minor', bottom=False, left=False)
    
    #
    # Label each cell with its purity        
    #
    if datalabels:
        # Loop over rows
        for i in range(plotdata.shape[0]):
            # Loop over columns
            for j in range(plotdata.shape[1]):
                vv = plotdata.iloc[i][j]
                
                if pd.notnull(vv):
                    ax.text(j, i, int(vv), ha="center", va="center", color="w")
        
    if ax_input is None:
        vdqp._draw_fig(filename, overwrite)


# =============================================================================
# Functions for purity plots
# =============================================================================
def plot_purity_of_patterns(data, num_rows=None, num_cols=None, perceptual_threshold=0.05, ax_input=None, filename=None, overwrite=False, fig_kw={}, ax_kw={}, **kwargs):
    """
    Convenience function for plotting the purity of one or more patterns of missingness.

    Parameters
    ----------
    data : series or a list of series
        The purity of the pattern, for each pair of variables
    num_rows : int, optional
        The number of rows in the grid. The default is None.
    num_cols : TYPE, optional
        The number of columns in the grid. The default is None.
    perceptual_threshold : float, optional
        Perceptual discontinuity threshold (0.0 - 1.0) or None. The default is 0.05.
    filename : string, optional
        None or a filename for the figure. The default is None.
    overwrite : boolean, optional
        False (do not overwrite file; the default) or True (overwrite file if it exists). The default is False.
    fig_kw : dictionary, optional
        Keyword arguments for a Matplotlib Figure object. The default is an empty dictionary.
    ax_kw : dictionary, optional
        Keyword arguments for a Matplotlib Axes object. The default is an empty dictionary.
    kwargs :
        Keyword arguments for a Matplotlib Axes.plot object

    Returns
    -------
    None.

    """
    axkwargs = ax_kw.copy()
    
    if isinstance(data, list):
        # Calculate the number of rows/columns of plots
        if num_rows is None and num_cols is None:
            # Default is a single row and as many columns as required
            ncols = len(data)
            nrows = 1
        elif num_rows is None:
            ncols = num_cols
            nrows = int((len(data) + ncols - 1) / ncols)
        elif num_cols is None:
            nrows = num_rows
            ncols = int((len(data) + nrows - 1) / nrows)
        else:
            nrows = num_rows
            ncols = num_cols

        dfp = data
    else:
        nrows = 1
        ncols = 1
        dfp = [data]
    #
    # Plot axis
    #
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set(**fig_kw)
    
    if 'title' in axkwargs and isinstance(axkwargs['title'], str):
        # There is a single title for all the plots
        fig.suptitle(ax_kw['title'])
        axkwargs['title'] = None
    #
    # Create the plots
    #
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

            if pnum < len(data):
                # If a list of titles is provided then apply the title for this plot
                kw = 'title'

                if kw in ax_kw and isinstance(ax_kw[kw], list):
                    # Need to check whether title is a list, so the suptitle() isn't used
                    axkwargs[kw] = vdqp._get_parameter(ax_kw[kw], pnum)

                # Create the plot
                plot_pattern_purity(dfp[pnum], perceptual_threshold, ax, filename, overwrite, fig_kw, axkwargs, **kwargs)
            else:
                # This cell in the grid is not used
                ax.axis('off')

            # Increment the parameter number
            pnum += 1
                

    # Draw the figure
    vdqp._draw_fig(filename, overwrite)
        
        
def plot_pattern_purity(data, perceptual_threshold=0.05, ax_input=None, filename=None, overwrite=False, fig_kw={}, ax_kw={}, **kwargs):
    """
    Plot the purity of a pattern of missingness.

    Parameters
    ----------
    data : series
        The purity of a type of pattern, for each pair of variables
    perceptual_threshold : float, optional
        Perceptual discontinuity threshold (0.0 - 1.0) or None. The default is 0.05.
    ax_input: axis or None, optional
        Matplotlib axis. The default is None.
    filename : string, optional
        None or a filename for the figure. The default is None.
    overwrite : boolean, optional
        False (do not overwrite file; the default) or True (overwrite file if it exists). The default is False.
    fig_kw : dictionary, optional
        Keyword arguments for a Matplotlib Figure object. The default is an empty dictionary.
    ax_kw : dictionary, optional
        Keyword arguments for a Matplotlib Axes object. The default is an empty dictionary.
    kwargs :
        Keyword arguments for a Matplotlib Axes.plot object

    Returns
    -------
    None.

    """
    axkwargs = ax_kw.copy()
    # Set defaults
    axkwargs['xlim'] = ax_kw.get('xlim', (0, len(data)-1))
    axkwargs['xlabel'] = ax_kw.get('xlabel', 'Pair of variables')
    axkwargs['ylabel'] = ax_kw.get('ylabel', 'Purity')
    #
    # Plot axis
    #
    if ax_input is None:
        fig, ax = plt.subplots()
        fig.set(**fig_kw)
    else:
        ax = ax_input

    if perceptual_threshold is None:
        ax.plot(np.arange(len(data)), data.sort_values().values, **kwargs)
    else:
        # Purity is normalised, so always in the range 0.0 - 1.0
        max_value = (0.0, 1.0)
        dfp2 = vdqp.apply_perceptual_discontinuity_individually(data, perceptual_threshold, max_value)
        # One of these two columns contains the value to be plotted, and the other column is zero
        dfp2['_purity'] = dfp2[['Perceptual discontinuity', 'Original number']].max(axis=1)
        ax.plot(np.arange(len(dfp2)), dfp2['_purity'].sort_values().values, **kwargs)

    ax.set(**axkwargs)
    
    if ax_input is None:
        vdqp._draw_fig(filename, overwrite)

        
def plot_purity_heatmap(df_patterns, threshold=0.0, row_col_order=None, ax_input=None, xlabels_rotate=0.0, ylabels_rotate=0.0, clist=[], datalabels=False, grid_spacing=None, filename=None, overwrite=False, fig_kw={}, ax_kw={}, cbar_kw={}, **kwargs):
    """
    Plot a heatmap of showing most pure pattern for each pair of variables.

    Parameters
    ----------
    data : data frame
        Heat map data (one row per intersection; value is cardinality of the intersection or np.nan (variable is not part of the intersection)).
    threshold : float, optional
        A purity threshold, below which a pattern will not be plotted.
    row_col_order : list, optional
        The order of columns in the heatmap. The default is None (they are plotted alphabetically).
    ax_input: axis or None, optional
        Matplotlib axis. The default is None.
    xlabels_rotate : float, optional
        Angle to rotate X axis labels by.
    ylabels_rotate : float, optional
        Angle to rotate Y axis labels by.
    clist : list, optional
        A list of the six colours to use in the heatmap. The default is an empty list (use the default colours).
    datalabels : boolean, optional
        Label each data point. The default is False.
    grid_spacing : int, optional
        Draw gridlines on the heatmap every N rows/columns. The default is None.
    filename : string, optional
        None or a filename for the figure. The default is None.
    overwrite : boolean, optional
        False (do not overwrite file; the default) or True (overwrite file if it exists). The default is False.
    fig_kw : dictionary, optional
        Keyword arguments for a Matplotlib Figure object. The default is an empty dictionary.
    ax_kw : dictionary, optional
        Keyword arguments for a Matplotlib Axes object. The default is an empty dictionary.
    cbar_kw : dictionary, optional
        Keyword arguments for a Matplotlib Colorbar object. The default is an empty dictionary.
    kwargs :
        Keyword arguments for a Matplotlib Axes.imshow object

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

    data = []
    df_filtered = df_patterns[df_patterns['Purity']>= threshold]
    grouped = df_filtered.sort_values(by=['Purity', 'Pattern'], ascending=False).groupby(by=['Column1', 'Column2'])
    
    for name, group in grouped:
        for index, row in group.iterrows():
            data.append([row['Column1'], row['Column2'], row['Pattern'], row['Purity']])
            break
    
    if row_col_order is None:
        # Get the columns in alphabetical order
        columns = list(set(df_filtered['Column1'].values.tolist() + df_filtered['Column2'].values.tolist()))
        columns.sort()
    else:
        columns = row_col_order
        
    # Create a dataframe for the heatmap with every value set to np.nan
    df_heatmap = pd.DataFrame(data=np.full(shape=(len(columns), len(columns)), fill_value=np.nan), index=columns, columns=columns)
    # Create a similar heatmap to store the purities in the heatmap
    df_purity = pd.DataFrame(data=np.full(shape=(len(columns), len(columns)), fill_value=np.nan), index=columns, columns=columns)
    
    for row in data:
        value = mdu._get_pattern_colournum(row[2], row[3])
        
        if value is not None:
            df_heatmap.loc[row[0]][row[1]] = value
            df_heatmap.loc[row[1]][row[0]] = value
            df_purity.loc[row[0]][row[1]] = row[3]
            df_purity.loc[row[1]][row[0]] = row[3]

    legend = mdu._get_pattern_legend()
    
    # Matplotlib version (see https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html)
    norm = mpl.colors.BoundaryNorm(np.linspace(-0.5, len(legend) - 0.5, len(legend) + 1), len(legend))
    fmt = mpl.ticker.FuncFormatter(lambda x, pos: legend[norm(x)])

    # Set the colourmap
    default_colours = get_default_colours('purity heatmap')
    
    try:
        if isinstance(clist, list) and len(clist) == 6:
            # User-defined colourmap
            kw = {'cmap': mpl.colors.ListedColormap(clist), 'norm': norm}
        else:
            # Use the default colours
            kw = {'cmap': mpl.colors.ListedColormap(default_colours), 'norm': norm}
    except:
        pass
        # Use the default colours
        kw = {'cmap': mpl.colors.ListedColormap(default_colours), 'norm': norm}

    im = ax.imshow(df_heatmap[columns], **kw)
    
    # Colourbar
    ax.figure.colorbar(im, ax=ax, ticks=np.arange(0, len(legend)), format=fmt, **cbar_kw)

    ax.set(**axkwargs)
    
    # Tick labels (major ticks naming each variable)
    ax.set_xticks(np.arange(df_heatmap.shape[1]))
    ax.set_xticklabels(columns, rotation=xlabels_rotate)
    ax.tick_params( bottom=False, top=True, labelbottom=False,labeltop=True)
    
    ax.set_yticks(np.arange(df_heatmap.shape[0]))
    ax.set_yticklabels(columns, rotation=ylabels_rotate)
    
    if isinstance(grid_spacing, int):
        # Gridlines based on minor ticks, every Nth row/column
        ax.set_xticks(np.arange(-0.5, df_heatmap.shape[1], grid_spacing), minor=True)
        ax.set_yticks(np.arange(-0.5, df_heatmap.shape[0], grid_spacing), minor=True)
        ax.grid(visible=True, which='minor', color='black')#, linestyle='-', linewidth=2)
        # Remove minor ticks
        ax.tick_params(which='minor', bottom=False, left=False)

    #
    # Label each cell with its purity        
    #
    if datalabels:
        # Add label showing each cell's value
        for i in range(len(df_heatmap.columns)):
            col1 = df_heatmap.columns[i]
            
            for j in range(len(df_heatmap)):
                col2 = df_heatmap.columns[j]
                vv = df_purity.loc[col1][col2]
                
                if pd.notnull(vv):
                    if abs(vv - int(vv)) <= 0:
                        ax.text(j, i, int(vv), ha="center", va="center", color="w")       
                    else:
                        ax.text(j, i, '{:.2f}'.format(min(vv, 0.99)), ha="center", va="center", color="w")       
        
    if ax_input is None:
        vdqp._draw_fig(filename, overwrite)


# =============================================================================
# Explanation graph functions
# =============================================================================
def plot_explanation_graph_summary(data, perceptual_threshold=0.05, ax_input=None, vert=True, xlabels_rotate=0.0, clist=[], elist=[], datalabels=False, legend=True, filename=None, overwrite=False, fig_kw={}, ax_kw={}, legend_kw={}, **kwargs):
    """
    Plot a summary of an Explanation_Graph that contains the number of intersections/rows/columns/values that are completely/partly/not explained by the graph's nodes.
    The summary is obtained by calling Explanation_Graph.get_summary()

    Parameters
    ----------
    data : data frame
        The summary (output by Explanation_Graph.get_summary())
    perceptual_threshold : float
        Preceptual discontinuity threshold (0.0 - 1.0) or None. The default is 0.05.
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
    # Make a copy of the axis kwargs and set missing keys to their default values
    axkwargs = ax_kw.copy()
    keys = ['title'] + (['xlabel'] if vert else ['ylabel']) + (['ylabel', 'ylim'] if vert else ['xlabel', 'xlim'])
    values = ['Explanation graph summary', '', '% explained', (0, 100)]
    
    for l1 in range(len(keys)):
        axkwargs[keys[l1]] = ax_kw.get(keys[l1], values[l1])

    # Set the colourmap (dark, purple, light purple and white from the Tableau20 colourmap)
    default_colours = get_default_colours('explanation graph summary')
    
    colors = clist if isinstance(clist, list) and len(clist) == 3 else default_colours
    ec = elist if isinstance(elist, list) and len(elist) == 3 else [[0, 0, 0]] * 3
    
    # Calculate the percentages the percentages
    dftmp = data.copy()
    tmpcols = []
    
    for col in data.columns:
        dftmp['% ' + col] = 100.0 * dftmp[col] / dftmp[col].sum()
        tmpcols.append('% ' + col)
    
    vdqp.stacked_bar(dftmp[tmpcols], perceptual_threshold=perceptual_threshold, ax_input=ax_input, vert=vert, xlabels_rotate=xlabels_rotate, clist=colors, elist=ec, datalabels=datalabels, legend=legend, filename=filename, overwrite=overwrite, fig_kw=fig_kw, ax_kw=axkwargs, legend_kw=legend_kw, **kwargs)


def plot_explanation_graph_diagram(graph, ax_input=None, node_positions={}, vert=True, link_shape='straight', filename=None, overwrite=False, link_kw={}, fig_kw={}, ax_kw={}, **kwargs):
    """
    Plot an Explanation_Graph. The nodes are positioned on an integer grid starting at (0, 0).

    Parameters
    ----------
    graph : Explanation_Graph
        A missing data explanation graph
    ax_input: axis or None, optional
        Matplotlib axis. The default is None.
    node_positions : dictionary, optional
        The keys are node IDs and the value is a dictionary with keys of 'x' and/or 'y' and values that specify the relevant coordinate (e.g. {'x': 2, 'y': 4}).
    vert: boolean, optional
        True (vertical layout; the default) or False (horizontal). The default is True.
    link_shape : str, optional
        'straight' (all links are drawn as a single straight line) or 'elbow' (links are drawn as elbow lines if nodes are offset)
    filename : string, optional
        None or a filename for the figure. The default is None.
    overwrite : boolean, optional
        False (do not overwrite file; the default) or True (overwrite file if it exists). The default is False.
    link_kw : dictionary, optional
        Keyword arguments for the links (a Matplotlib plot object)
    fig_kw : dictionary, optional
        Keyword arguments for a Matplotlib Figure object. The default is an empty dictionary.
    ax_kw : dictionary, optional
        Keyword arguments for a Matplotlib Axes object. The default is an empty dictionary.
    kwargs :
        Keyword arguments for a Matplotlib Text object

    Returns
    -------
    None.

    """
    lkwargs = link_kw.copy()
    
    if 'color' not in lkwargs:
        # Default colour for the links (Matplotlib would make each one a different colour!)
        lkwargs['color'] = 'black'

    axkwargs = ax_kw.copy()
    textkw = kwargs.copy()
    
    # Set default box style if none is defined
    if 'bbox' not in textkw:
        textkw['bbox'] = {'boxstyle': 'square'}
    elif 'boxstyle' not in textkw['bbox']:
        textkw['bbox']['boxstyle'] = 'square'
    
    # Set default box facecolor if none is defined
    if 'facecolor' not in textkw['bbox']:
        textkw['bbox']['facecolor'] = 'w'

    # Set default text justification if none is defined
    if 'horizontalalignment' not in textkw:
        textkw['horizontalalignment'] = 'center' if vert else 'left'
    #
    # Plot the graph
    #
    if ax_input is None:
        fig, ax = plt.subplots()
        fig.set(**fig_kw)
    else:
        ax = ax_input

    ann = []
    minmax = None
    # Calculate the graph layout
    graph.calc_layout()
    
    for node_id in range(graph.num_nodes()):
        px, py, text, node_attr = graph.get_coords(node_id, vert)
        
        # User-defined position
        try:
            udict = node_positions[node_id]
            
            try:
                px = udict['x']
            except:
                pass
            
            try:
                py = udict['y']
            except:
                pass
        except:
            pass
        
        # Set the node's position
        xy = (px, py)
        # Update the min/max position
        try:
            minmax[0][0] = min(minmax[0][0], px)
            minmax[0][1] = max(minmax[0][1], px)
            minmax[1][0] = min(minmax[1][0], py)
            minmax[1][1] = max(minmax[1][1], py)
        except:
            minmax = [[px, px], [py, py]]
            pass
            
        # Apply this node's attributes
        # *** START EDIT
        attribs = {} if node_attr is None else node_attr.copy()
        # *** END EDIT
            
        for key, value in textkw.items():
            if key not in attribs:
                # This attribute is not specified for the node so set it to the default
                attribs[key] = value
            elif key == 'bbox':
                # Both the node and the default have bbox attributes, so check them individually
                
                for k2, v2 in textkw['bbox'].items():
                    if k2 not in attribs['bbox']:
                        attribs['bbox'][k2] = v2
                
        aa = ax.annotate(text, xy, **attribs)
            
        ann.append(aa)
        
    # Set the axis limits
    if vert:
        margin = (0.6, 0.25)
        ax.set_xlim(minmax[0][0] - margin[0], minmax[0][1] + margin[0])
        ax.set_ylim(minmax[1][0] - margin[1], minmax[1][1] + margin[1])
    else:
        xmargin = [0.25, 0.75]
        ymargin = 0.6
        ax.set_xlim(minmax[0][0] - xmargin[0], minmax[0][1] + xmargin[1])
        ax.set_ylim(minmax[1][0] - ymargin, minmax[1][1] + ymargin)

    # Calculate the coordinates of each annotation bbox
    # See https://stackoverflow.com/questions/41267733/getting-the-coordinates-of-a-matplotlib-annotation-label-in-figure-coordinates
    fig.canvas.draw()
    
    # Calculate the coordinates of each annotation bbox
    coords = []
    
    for aa in ann:
        patch = aa.get_bbox_patch()
            
        box  = patch.get_extents()
        cc = ax.transData.inverted().transform(box)
        coords.append(cc)
    
    # Draw the links between the nodes
    l1 = 0
    for l1 in range(graph.num_nodes()):
        parent = graph.get_node(l1)
        node_num_parent = parent.get_id()
        # Specify the first coordinate, and set the second to zero
        if vert:
            x1 = 0.5 * (coords[node_num_parent][0][0] + coords[node_num_parent][1][0])
            y1 = coords[node_num_parent][0][1]
        else:
            x1 = coords[node_num_parent][1][0]
            y1 = 0.5 * (coords[node_num_parent][0][1] + coords[node_num_parent][1][1])
    
        ## This node has children
        for child in parent.get_children():
            node_num_child = child.get_id()
            # Specify the second coordinate
            if vert:
                x2 = 0.5 * (coords[node_num_child][0][0] + coords[node_num_child][1][0])
                y2 = coords[node_num_child][1][1]
                
                if link_shape == 'straight' or parent._y == child._y:
                    xx = [x1, x2]
                    yy = [y1, y2]
                else:
                    xx = [x1, x1, x2, x2]
                    ymid = 0.5 * (y1 + y2)
                    yy = [y1, ymid, ymid, y2]
            else:
                xx = [x1, coords[node_num_child][0][0]]
                yy = [y1, 0.5 * (coords[node_num_child][0][1] + coords[node_num_child][1][1])]
    
            ax.plot(xx, yy, **lkwargs)
    
    # Draw a box around the graph
    ax.tick_params( bottom=False, left=False, labelbottom=False,labelleft=False)
    
    # This should allow axes settings to be overridden by a user
    ax.set(**axkwargs)
    
    if ax_input is None:
        vdqp._draw_fig(filename, overwrite)
