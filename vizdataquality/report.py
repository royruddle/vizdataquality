# -*- coding: utf-8 -*-
"""
Created on Thu Jan 4 14:00:00 2024

   Copyright 2024 Roy Ruddle

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

# =============================================================================
# Report class
# =============================================================================
class Report:
    """This class allows users to write a report while they investigate data quality and profiling a dataset.
    Reports may have headings, text, figures and tables.
    The overall structure may follow the six steps we suggest or be freeform.
    Reports may be output as a webpage, in Latex or in a text file.
    """
    def __init__(self):
        self.report = {}
        self.num_items = 0
        

    def get_report_dict(self):
        """
        Get the content of the report.
    
        Parameters
        ----------
        None.
    
        Returns
        -------
        dict
            The report items in a dictionary.
        """
        return self.report
        

    def add_acknowledgements(self, text=None, key=None):
        """
        Add the supplied acknowledgements to the report.
    
        Parameters
        ----------
        text : string
            Text to add to the report. The default is None.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the acknowledgements in the report dictionary.
        """
        txt = ('' if text is None else text + ' ') + 'This report was created using the vizdataquality Python package.'
        return self.add_heading('Acknowledgements', text=txt, key=key)
    
    
    def step1(self, text=None, key=None):
        """
        Add the Step 1 heading to the report.
    
        Parameters
        ----------
        text : string
            Text to add to the report. The default is None.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the step in the report dictionary.
        """
        return self.add_heading('Step 1: Look at your data (is anything obviously wrong?)', text=text, key=key)
    
    
    def step2(self, text=None, key=None):
        """
        Add the Step 2 heading to the report.
    
        Parameters
        ----------
        text : string
            Text to add to the report. The default is None.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the step in the report dictionary.
        """
        return self.add_heading('Step 2: Watch out for special values', text=text, key=key)
    
    
    def step3(self, text=None, key=None):
        """
        Add the Step 3 heading to the report.
    
        Parameters
        ----------
        text : string
            Text to add to the report. The default is None.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the step in the report dictionary.
        """
        return self.add_heading('Step 3: Is any data missing?', text=text, key=key)
    
    
    def step4(self, text=None, key=None):
        """
        Add the Step 1 heading to the report.
    
        Parameters
        ----------
        text : string
            Text to add to the report. The default is None.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the step in the report dictionary.
        """
        return self.add_heading('Step 4: Check each variable', text=text, key=key)
    
    
    def step5(self, text=None, key=None):
        """
        Add the Step 5 heading to the report.
    
        Parameters
        ----------
        text : string
            Text to add to the report. The default is None.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the step in the report dictionary.
        """
        return self.add_heading('Step 5: Check combinations of variables', text=text, key=key)
    
    
    def step6(self, text=None, key=None):
        """
        Add the Step 6 heading to the report.
    
        Parameters
        ----------
        text : string
            Text to add to the report. The default is None.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the step in the report dictionary.
        """
        return self.add_heading('Step 6: Profile the cleaned data', text=text, key=key)
    
    
    def add_title(self, heading, key=None):
        """
        Add the supplied title to the report.
    
        Parameters
        ----------
        title : string
            The title.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the title in the report dictionary.
        """
        self.num_items += 1
        k = self.num_items if key is None else key
        self.report[k] = {'Num': self.num_items, 'Title': heading}
        
        return k
    
    
    def add_heading(self, heading, level=1, text=None, key=None):
        """
        Add the supplied heading to the report.
    
        Parameters
        ----------
        heading : string
            The heading.
        level : int
            1 - n. The default is 1.
        text : string
            Text to add to the report. The default is None.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the heading in the report dictionary.
        """
        self.num_items += 1
        k = self.num_items if key is None else key
        self.report[k] = {'Num': self.num_items, 'Heading': heading, 'Level': level, 'Text': text}
        
        return k
    
    
    def dataset_size(self, name, num_rows, num_cols, text=None, caption=None, key=None):
        """
        Add a table summarising the size of a dataset to the report.
    
        Parameters
        ----------
        name : string
            The name of the dataset
        num_rows : int
            Number of rows in the dataset
        num_cols : int
            Number of columns in the dataset
        text : string
            Text to add to the report. The default is None.
        caption : string
            A caption for the table. The default is None.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the table in the report dictionary.
        """
        data = {'Dataset': [name], 'Number of rows': [num_rows], 'Number of columns': [num_cols]}
        df = pd.DataFrame.from_dict(data)
        cap = 'The size of the dataset.' if caption is None else caption
        return self.add_table(df, index=False, text=text, caption=cap, key=key)
    
    
    def add_descriptive_stats(self, df, text=None, caption=None, key=None):
        """
        Add descriptive statistics (e.g., calculated by calc() in calculate.py) to the report.
    
        Parameters
        ----------
        df : dataframe
            The descriptive statistics to be added. The index is the variable names.
        text : string
            Text to add to the report. The default is None.
        caption : string
            A caption for the table. The default is None.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the table in the report dictionary.
        """
        # Move the index (variable names) into a column
        df2 = df.reset_index()
        df2.rename(columns={df2.columns[0]: 'Variable'}, inplace=True)
        cap = 'Descriptive statistics.' if caption is None else caption
        return self.add_table(df2, index=False, text=text, caption=cap, key=key)
    
    
    def add_table(self, df=None, index=False, filename=None, text=None, caption=None, key=None):
        """
        Add the supplied dataframe to the report.
    
        Parameters
        ----------
        df : dataframe
            The table to be added. The default is None.
        index : boolean
            Whether or not to output the dataframe index in the report. The default is False.
        filename : filename
            The file containing the table. The default is None.
        text : string
            Text to add to the report. The default is None.
        caption : string
            A caption for the table. The default is None.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the table in the report dictionary.
        """
        if df is not None and df.shape[0] * df.shape[1] > 0:
            self.num_items += 1
            k = self.num_items if key is None else key
            self.report[k] = {'Text': text, 'Table_df': df, 'Index': index, 'Caption': caption}
        elif filename is not None:
            self.num_items += 1
            k = self.num_items if key is None else key
            self.report[k] = {'Text': text, 'Table_file': filename, 'Caption': caption}
        else:
            k = None
            print('** WARNING ** vizdataquality, report.py, add_table(): The input dataframe is empty.')

        return k
    
    
    def add_figure(self, filename, text=None, caption=None, key=None):
        """
        Add the supplied figure to the report.
    
        Parameters
        ----------
        filename : filename
            The file containing the figure.
        text : string
            Text to add to the report. The default is None.
        caption : string
            A caption for the table. The default is None.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the figure in the report dictionary.
        """
        self.num_items += 1
        k = self.num_items if key is None else key
        self.report[k] = {'Text': text, 'Figure': filename, 'Caption': caption}

        return k
    
    
    def paragraph(self, text, key=None):
        """
        Add a paragraph of text to the report.
    
        Parameters
        ----------
        text : string
            The text of the paragraph.
        key : string
            User-defined name of this report item. The default is None.
    
        Returns
        -------
        int
            The key used for the paragraph in the report dictionary.
        """
        self.num_items += 1
        k = self.num_items if key is None else key
        self.report[k] = {'Paragraph': text}

        return k
        
        
    def save(self, filename, overwrite=True, table_kw={}, **kwargs):
        """
        Save the report in a text format file.
    
        Parameters
        ----------
        filename : TYPE, optional
            DESCRIPTION. The default is filename.
        overwrite : TYPE, optional
            DESCRIPTION. The default is True.
        table_kw : dictionary
            Keyword arguments for pd.DataFrame.to_html() or to_latex(). Default is an empty dictionary.
        **kwargs : dictionary
            Keyword arguments for open()
    
        Returns
        -------
        None.
    
        """
        filetype = 'text'
        try:
            extension = os.path.splitext(filename)[1]
            
            if extension == '.htm' or extension == '.html':
                filetype = 'html'
            elif extension == '.tex':
                filetype = 'latex'
                # If 'position' is defined then it is also applied to the tables/figures that this function writes 
                try:
                    position = '[' + table_kw['position'] + ']'
                except:
                    position = ''
                    pass

        except:
            pass
            
        try:
            with open(filename, 'w', **kwargs) as fout:
                report_dir = os.path.dirname(filename)
                dir_length = len(report_dir)
                
                # Document header
                if filetype == 'html':
                    fout.write('<!DOCTYPE html>' + '\n')
                    fout.write('<html>' + '\n')
                    
                    fout.write('<head>' + '\n')
                    fout.write('<meta charset="utf-8">' + '\n')
                    # Centre tables
                    #fout.write('<style>' + '\n')
                    #fout.write('table.center {margin-left: auto; margin-right: auto}' + '\n')
                    #fout.write('</style>' + '\n')
                    
                    fout.write('</head>' + '\n')
                    
                    fout.write('<body>' + '\n')
                    
                    fout.write('\n')
                elif filetype == 'latex':
                    fout.write('\documentclass{article}' + '\n')
                    
                    # Output the title, if there is one
                    for name, item in self.report.items():
                        try:
                            fout.write('\\title{' + item['Title'] + '}' + '\n')
                            break
                        except:
                            pass
                            
                    fout.write('\\usepackage{booktabs} % Required for inserting tables' + '\n')
                    fout.write('\\usepackage{graphicx} % Required for inserting images' + '\n')
                    fout.write('\\begin{document}' + '\n')
                    fout.write('\n')
                    
                table_num = 0
                figure_num = 0
                # Loop over each item in the report
                for name, item in self.report.items():
                    output_index = item.get('Index', False)
                    # Loop over each element of this report item
                    for key, value in item.items():
                        if value is not None:
                            if key == 'Title':
                                if filetype == 'html':
                                    prefix = '<h1>'
                                    suffix = '</h1>'
                                    vv = value
                                elif filetype == 'latex':
                                    prefix = ''
                                    vv = '\maketitle'
                                    suffix = ''
                                else:
                                    prefix = 'TITLE '
                                    suffix = ''
                                    vv = value
                                    
                                # Output the title
                                fout.write(prefix + vv + suffix + '\n')
                                fout.write('\n')
                            if key == 'Heading':
                                if filetype == 'html':
                                    # Headings are <h2> onward, because <h1> is used for the title
                                    tag = 'h' + str(item['Level'] + 1)
                                    prefix = '<' + tag + '>'
                                    suffix = '</' + tag + '>'
                                elif filetype == 'latex':
                                    pp = ['\section{', '\subsection{', '\subsubsection{']
                                    prefix = pp[item['Level']-1] if item['Level'] <= len(pp) else pp[-1]
                                    suffix = '}'
                                else:
                                    prefix = 'HEADING' + str(item['Level']) + ' '
                                    suffix = ''
                                    
                                # Output the heading
                                fout.write(prefix + value + suffix + '\n')
                                fout.write('\n')
                                
                                # Output the paragraph of text that part of this heading
                                if item['Text'] is not None:

                                    if filetype == 'html':
                                        prefix = '<p>'
                                        suffix = '</p>'
                                    else:
                                        prefix = ''
                                        suffix = ''
                                        
                                    fout.write(prefix + item['Text'] + suffix + '\n')
                                    fout.write('\n')
                                
                            elif key == 'Paragraph':
                                if filetype == 'html':
                                    prefix = '<p>'
                                    suffix = '</p>'
                                else:
                                    prefix = ''
                                    suffix = ''
                                
                                # Output the text
                                fout.write(prefix + value + suffix + '\n')
                                fout.write('\n')
                            elif key == 'Figure':
                                figure_num += 1
                                
                                label = 'fig:' + str(figure_num) if filetype == 'latex' else str(figure_num)
                                #
                                # Output the paragraph of text that part of this figure
                                #
                                if item['Text'] is not None:
                                    ref = '~\\ref{' + label + '}' if filetype == 'latex' else ' ' + label
                                    txt = item['Text'].replace('$figure', 'Figure' + ref)

                                    if filetype == 'html':
                                        prefix = '<p>'
                                        suffix = '</p>'
                                    else:
                                        prefix = ''
                                        suffix = ''
                                        
                                    fout.write(prefix + txt + suffix + '\n')
                                    fout.write('\n')
                                #
                                # Output the figure itself, trimming the file's path if it starts the same as the report_dir
                                #
                                vv = value
                                
                                try:
                                    # Exclude the final directory delimiter from the trimmed filename
                                    vv = value[dir_length+1:] if dir_length > 0 and value[:dir_length] == report_dir else value
                                except:
                                    pass
                                        
                                if filetype == 'html':
                                    fout.write('<p><img src="' + vv + '"></p>' + '\n')
                                    fout.write('\n')

                                    # Prefix the caption by the figure number
                                    try:
                                        prefix = '<p><i>'
                                        suffix = '</i></p>'
                                        fout.write(prefix + 'Figure ' + str(figure_num) + ': ' + item['Caption'] + suffix + '\n')
                                        fout.write('\n')
                                    except:
                                        pass
                                elif filetype == 'latex':
                                    fout.write('\\begin{figure}' + position + '\n')
                                    fout.write('  \includegraphics[width=\linewidth]{' + vv + '}' + '\n')

                                    # Add caption
                                    try:
                                        prefix = '  \caption{'
                                        suffix = '}'
                                        fout.write(prefix + item['Caption'] + suffix + '\n')
                                    except:
                                        pass
                                    
                                    fout.write('  \label{' + label + '}' + '\n')
                                    fout.write('\end{figure}' + '\n')
                                    fout.write('\n')
                                else:
                                    # Output the filename of the figure
                                    fout.write(vv + '\n')
                                    fout.write('\n')
                                    
                                    # Prefix the caption by the figure number
                                    try:
                                        fout.write('Figure ' + str(figure_num) + ': ' + item['Caption'] + '\n')
                                        fout.write('\n')
                                    except:
                                        pass
                            elif key == 'Table_file' or key == 'Table_df':
                                table_num += 1
                                
                                label = 'tab:' + str(table_num) if filetype == 'latex' else str(table_num)
                                #
                                # Output the paragraph of text that part of this table
                                #
                                if item['Text'] is not None:
                                    ref = '~\\ref{' + label + '}' if filetype == 'latex' else ' ' + label
                                    txt = item['Text'].replace('$table', 'Table' + ref)

                                    if filetype == 'html':
                                        prefix = '<p>'
                                        suffix = '</p>'
                                    else:
                                        prefix = ''
                                        suffix = ''
                                        
                                    fout.write(prefix + txt + suffix + '\n')
                                    fout.write('\n')
                                #
                                # Output the table itself, trimming the file's path if it starts the same as the report_dir
                                #
                                vv = value
                                
                                try:
                                    # Exclude the final directory delimiter from the trimmed filename
                                    vv = value[dir_length+1:] if dir_length > 0 and value[:dir_length] == report_dir else value
                                except:
                                    pass
                                
                                if filetype == 'html':
                                    if key == 'Table_file':
                                        fout.write('<img src="' + vv + '">' + '\n')
                                        fout.write('\n')
                                    else:
                                        tab_kw = table_kw.copy()
                                        # Use the internal value of Index
                                        tab_kw['index'] = item['Index']
                                        value.to_html(fout, **tab_kw)#, classes=['center'])
                                        fout.write('\n')
                                        fout.write('\n')
                                        
                                    # Prefix the caption by the table number
                                    try:
                                        #prefix = '<p style="text-align:center"><i>'
                                        prefix = '<p><i>'
                                        suffix = '</i></p>'
                                        fout.write(prefix + 'Table ' + str(table_num) + ': ' + item['Caption'] + suffix + '\n')
                                        fout.write('\n')
                                    except:
                                        pass
                                elif filetype == 'latex':
                                    if key == 'Table_file':
                                        fout.write('\\begin{table}' + position + '\n')
                                        fout.write('  \includegraphics[width=\linewidth]{' + vv + '}' + '\n')
    
                                        # Add caption
                                        try:
                                            prefix = '  \caption{'
                                            suffix = '}'
                                            fout.write(prefix + item['Caption'] + suffix + '\n')
                                        except:
                                            pass
                                        
                                        fout.write('  \label{tab:' + str(table_num) + '}' + '\n')
                                        fout.write('\end{table}' + '\n')
                                    else:
                                        tab_kw = table_kw.copy()
                                        # The possible keywords include position and label
                                        # Use the internal value of Index and Caption, and force the position of the table
                                        tab_kw['index'] = item['Index']
                                        tab_kw['caption'] = item['Caption']
                                        tab_kw['label'] = label
                                        value.to_latex(fout, **tab_kw)

                                    fout.write('\n')
                                else:
                                    if key == 'Table_file':
                                        # Output the filename of the table
                                        fout.write(value + '\n')
                                    else:
                                        # Value should be a dataframe. Print it tab-delimited, optionally including the index
                                        line = ['Variable'] if output_index else []
                                        line += value.columns.tolist()
                                        fout.write('\t'.join(line) + '\n')
                                        # Output each row of the dataframe
                                        for row in value.itertuples():
                                            line = [str(row[0])] if output_index else []
                                            line += [str(row[i+1]) for i in range(value.shape[1])]
                                            fout.write('\t'.join(line) + '\n')
                                
                                    fout.write('\n')
                                
                                    # Prefix the caption by the table number
                                    try:
                                        fout.write('Table ' + str(table_num) + ': ' + item['Caption'] + '\n')
                                        fout.write('\n')
                                    except:
                                        pass

                # Document footer
                if filetype == 'html':
                    fout.write('</html>' + '\n')
                    fout.write('</body>' + '\n')
                elif filetype == 'latex':
                    fout.write('\end{document}' + '\n')
                    
        except Exception:
            raise
