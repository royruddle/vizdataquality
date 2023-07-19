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
import sys

import re
import io
import datetime

import chardet
import csv
import logging
import traceback


##############################################################################    
def output_dataframe_desc_stats(df, input_filename, output_filename, output_file_delimiter, output_file_encoding='utf_8', new_output_file=True, overwrite_output_file=False, num_input_file_records=None, options=None, special_values=None):
    """
    Output the descriptive statistics for a data frame
    
    :param df: A data frame
    :param input_filename: Name of the input file
    :param output_filename: The full path name of the output file
    :param output_file_delimiter: Output file delimiter
    :param output_file_encoding: The output file encoding (default is utf_8)
    :param new_output_file: True (create a new output file) or False (append to the file, if it exists)
    :param overwrite_output_file: True (overwrite the output file if it exists) or False (do not; default). Only used if new_output_file is True.
    :param num_input_file_records: None (defined by the length of the series) or >= 0 (the number of records in the original series; the supplied series may have omitted some or all missing values)
    :param options: A dictionary that specifies the descriptive statistics to output (default is None; output everything)
    :param special_values: A dictionary of dictionaries, or None (default). The top-level dictionary specifies the heading for a set of special values and the 2nd level specifies the value or a list of values for a variable (every value is stored as a string). If a variable is not mentioned then it does not have any special values for that heading.
    """
    function_name = 'print_dataframe_desc_stats()'
    logger = logging.getLogger('QCprofiling')
    logger.debug('%s,input_filename,%s' %(function_name, input_filename))
    
    if new_output_file and not overwrite_output_file and os.path.exists(output_filename):
        logger.warning('%s Output file already exists: %s' %(function_name, output_filename))
    else:
        st = datetime.datetime.now()
        
        try:
            with open(output_filename, 'w' if new_output_file else 'a', encoding=output_file_encoding) as fout:
                #if newfile:
                newline = '\n'
                try:
                    if options is None or options.get('File info', False):
                        fout.write('Input file' + output_file_delimiter + input_filename + newline)
                        fout.write('Number of rows' + output_file_delimiter + str(len(df)) + newline)
                        fout.write('Number of columns' + output_file_delimiter + str(len(df.columns)) + newline)
                except Exception as e:
                    logger.exception('%s %s. Unexpected exception: %s' %(function_name, type(e).__name__, getattr(e, 'message', str(e))))
                    #logger.exception('%s Unexpected exception writing to file %s. %s' % (function_name, input_filename, sys.exc_info()[0]))
                    raise
                else:
                    # Print the descriptive statistics for each column of the data frame
                    output_header = True
                    
                    for col in df.columns:
                        try:
                            output_column_desc_stats(df[col], num_input_file_records, fout, output_header, output_file_delimiter, newline, options, special_values)
                            output_header = False
                        except: 
                            #logger.error("%s: Exception writing to output file: %s %s" %(function_name, output_filename, col))
                            # A message should already have been printed
                            break
            
            #except IOError as e:
            #logger.error("%s: Output file: %s " %(function_name, output_filename) + "I/O error({0}): {1}".format(e.errno, e.strerror))
        except: 
            logger.warning('%s Cannot open file: %s' %(function_name, output_filename))
        else:
            et = datetime.datetime.now()
            logger.info("%s,input_filename,%s,time (s),%s" %(function_name, input_filename, str((et-st).total_seconds())))


##############################################################################
def output_column_desc_stats(series, num_records, fout, output_header, output_file_delimiter, newline, options=None, special_values=None):
    """
    Output the descriptive statistics for a data series (a column of a data frame). 
    
    :param series: The data series
    :param num_records: None (defined by the length of the series) or >= 0 (the number of records in the original series; the supplied series may have omitted some or all missing values)
    :param fout: Output file (must be open)
    :param output_header: True (output header line) or False (do not)
    :param output_file_delimiter: Output file delimiter
    :param newline: character for newline
    :param options: A dictionary that specifies the descriptive statistics to output (default is None; output everything)
    :param special_values: A dictionary of dictionaries, or None (default). The top-level dictionary specifies the heading for a set of special values and the 2nd level specifies the value or a list of values for a variable (every value is stored as a string). If a variable is not mentioned then it does not have any special values for that heading.
    """
    function_name = 'print_col_desc_stats()'
    logger = logging.getLogger('QCprofiling')
    logger.debug("%s,column,%s" %(function_name, series.name))

    try:
        num_examples = 1 if options is None else options.get('Example values', 1)
        
        if type(special_values) is dict:
            special_value_keys = list(special_values.keys())
            special_value_keys.sort()
        else:
            special_value_keys = []
            
        if output_header:
            fout.write('Variable')
            
            if options is None or options.get('Type', False):
                fout.write(output_file_delimiter + 'Type')
            
            if options is None or options.get('Rows', False):
                fout.write(output_file_delimiter + 'Rows')
                
            if options is None or options.get('Missing', False):
                fout.write(output_file_delimiter + 'Number of missing values')
                
            for key in special_value_keys:
                fout.write(output_file_delimiter + 'Number of values: ' + key)
            
            if options is None or options.get('Unique', False):
                fout.write(output_file_delimiter + 'Number of unique values')# + output_file_delimiter + 'Uniqueness')
                
            if num_examples > 0:
                for l1 in range(num_examples):
                    fout.write(output_file_delimiter + 'Example value ' + str(l1 + 1))
            
            if options is None or options.get('Value lengths', False):
                fout.write(output_file_delimiter + 'Value lengths (min)' + output_file_delimiter + 'Value lengths (max)')
            
            if options is None or options.get('Character pattern', False):
                fout.write(output_file_delimiter + 'Patterns')
            
            if options is None or options.get('Number percentiles', False):
                fout.write(output_file_delimiter + 'Min' + output_file_delimiter + '25th' + output_file_delimiter + '50th' + output_file_delimiter + '75th' + output_file_delimiter + 'Max')
            
            if options is None or options.get('Datetime percentiles', False):
                fout.write(output_file_delimiter + 'Min date' + output_file_delimiter + '25th date' + output_file_delimiter + '50th date' + output_file_delimiter + '75th date' + output_file_delimiter + 'Max date')
            #
            # *** END NEW CODE
            #
            fout.write(newline)
    
        # Output the name of the column
        fout.write(series.name)
                
        if options is None or options.get('Type', False):
            fout.write(output_file_delimiter + str(series.dtype))
        
        if options is None or options.get('Rows', False):
            fout.write(output_file_delimiter + str(len(series)))
        
        if options is None or options.get('Missing', False):
            
            if num_records is None:
                num_missing = series.isnull().sum()
            else:
                num_missing = series.isnull().sum() + (num_records - len(series))
        
            fout.write(output_file_delimiter + str(num_missing))

        # series.unique() includes NaT as a value, but series.value_counts() does not
        # num_unique = len(series.unique())
        value_counts = series.value_counts()
        num_unique = len(value_counts)
            
        # Only output the remaining descriptive statistics if there is at least one non-missing value
        if num_unique > 0:
    
            # Loop over the special value dictionaries and output one column for each
            for key in special_value_keys:
                value_dict = special_values.get(key, None)
                try:
                    # Obtain the special values for this column
                    values = value_dict.get(series.name, None)
                    
                    if values is None:
                        # This column has no special values in this dictionary
                        fout.write(output_file_delimiter)
                    else:
                        
                        if type(values) is not list:
                            # Put the single value in a list
                            values = [values]

                        error = False
                        cnt = 0
                        
                        for val in values:
                            try:
                                if series.dtype == 'int64':
                                    cnt += value_counts[int(val)]
                                else:
                                    cnt += value_counts[val]
                            except KeyError as e:
                                # None of the variable's values are equal to val
                                pass
                            except Exception as e:
                                error = True
                                logger.warning('%s column %s. Cannot calculate count for special value: %s' %(function_name, series.name, str(val)))
                                logger.exception('%s %s. Unexpected exception with column %s. %s' %(function_name, type(e).__name__, series.name, getattr(e, 'message', str(e))))
                        
                        if error:
                            fout.write(output_file_delimiter + 'ERROR')
                        else:
                            fout.write(output_file_delimiter + str(cnt))
                except:
                    # This column has no special values in this dictionary
                    fout.write(output_file_delimiter)
            
            if options is None or options.get('Unique', False):
                fout.write(output_file_delimiter + str(num_unique))
                # Uniqueness: number of distinct values / number of values present for this variable
                #out.write(output_file_delimiter + str(float(num_unique) / float(num_rec)))
                
            if num_examples > 0:
                # Output example values
                l1 = 1
                
                for ind in value_counts.index:
                    fout.write(output_file_delimiter + str(ind))
                    l1 += 1
                    
                    if l1 > num_examples:
                        break
                    
                # Output delimiters if the variable has fewer distinct values than num_examples
                for l1 in range(num_examples - len(value_counts.index)):
                    fout.write(output_file_delimiter)
            
            if options is None or options.get('Value lengths', False):
                # Output the MIN/MAX value lengths    
                # This is more memory and CPU efficient than the pandas-style code that uses 'apply'
                col_series = series.dropna()
                min_length = len(str(col_series.values[0]))
                max_length = min_length
                for l1 in range(1, len(col_series)):
                    value_length = len(str(col_series.values[l1]))
                    min_length = min(min_length, value_length)
                    max_length = max(max_length, value_length)
                
                if False:
                    # Old code, which produced a bug if the first value was missing
                    min_length = len(str(series.values[0]))
                    max_length = min_length
                    for l1 in range(1, len(series)):
                        if pd.notnull(series.values[l1]):
                            value_length = len(str(series.values[l1]))
                            min_length = min(min_length, value_length)
                            max_length = max(max_length, value_length)

                fout.write(output_file_delimiter + str(min_length) + output_file_delimiter + str(max_length))
                
                # To make this more memory-efficient, it could calculate the min/max as it loops through the values of the series
                ###value_lengths = series.apply(lambda x: len(x))
                ###fout.write(delim + str(value_lengths.min()) + delim + str(value_lengths.max()))
                ###del value_lengths
                #
                # Or you could use series.str.len()
                # Not sure how fast that is or what memory it uses
                #
        
            if options is None or options.get('Character pattern', False):
                # Only output character pattern if the column is of type 'object'
                if series.dtype == 'object':
                    # Output the patterns that are in the series
                    #regex_pat_list = [r' ', re.compile(r'[a-z]', flags=re.IGNORECASE), re.compile(r'[0-9]'), re.compile(r'\.'), r'\-', r'/']
                    # Replace [0-9] by \d
                    regex_pat_list = [r' ', re.compile(r'[a-z]', flags=re.IGNORECASE), re.compile(r'\d'), re.compile(r'\.'), r'\-', r'/']
                    pattern_keys = [' ', 'a', '0', '.', '-', '/']
                    #regex_pat_other = re.compile(r'[^ a-z0-9\.\-/]', flags=re.IGNORECASE)
                    # Replace a-z0-9 by \w
                    regex_pat_other = re.compile(r'[^ \w\.\-/]', flags=re.IGNORECASE)
                    patterns = get_column_patterns(series, regex_pat_list, pattern_keys, regex_pat_other)
                    fout.write(output_file_delimiter)
                
                    for item in patterns:
                        # ### Old code that cleaned pattern text
                        # regex_pat_nonascii = r'[^\w\s]+'
                        # cleaned_list = re.findall(regex_pat_nonascii, item, re.ASCII)
                        # print(series.name + ' PATTERN ' + str(len(item)) + '   ' + str(len(cleaned_list)))
                        # print(cleaned_list)
                        fout.write(item)
                
                else:
                    fout.write(output_file_delimiter)
            
            if options is None or options.get('Number percentiles', False):
                
                if np.issubdtype(series.dtype, np.number) and not np.issubdtype(series.dtype, np.datetime64):
                    quantiles = series.dropna().quantile([0.0, 0.25, 0.5, 0.75, 1.0])
                
                    for ind, val in quantiles.iteritems():
                        fout.write(output_file_delimiter + str(val))
                    # if np.issubdtype(series.dtype, np.timedelta64):
                        # For dates the quantiles are output in days
                        # fout.write(delim + str(val.days))
                    # else:
                        # fout.write(delim + str(val))
                else:
                    fout.write(output_file_delimiter + output_file_delimiter + output_file_delimiter + output_file_delimiter + output_file_delimiter)
            
            if options is None or options.get('Datetime percentiles', False):
                
                if np.issubdtype(series.dtype, np.datetime64):
                    quantiles = series.dropna().quantile([0.0, 0.25, 0.5, 0.75, 1.0])
                
                    for ind, val in quantiles.iteritems():
                        fout.write(output_file_delimiter + str(val))
                else:
                    fout.write(output_file_delimiter + output_file_delimiter + output_file_delimiter + output_file_delimiter + output_file_delimiter)
        #
        # *** START NEW CODE
        #
        # Removed code from QCdataProcessing.py that generated output for sparklines-type visual descriptive statistics
        #
        # *** END NEW CODE
        #
    
        fout.write(newline)
    except (SystemExit, KeyboardInterrupt):
        logger.warning("%s KeyboardInterrupt" %(function_name))
        raise
    except Exception as e:
        logger.exception('%s %s. Unexpected exception: %s' %(function_name, type(e).__name__, getattr(e, 'message', str(e))))
        traceback.print_exc(file=sys.stdout)
        raise


##############################################################################    
def get_column_patterns(series, regex_pat_list, pattern_keys, regex_pat_other):
    """
    Finds whether each of the supplied regex patterns occurs anywhere in a series
    
    :param series: A data series
    :param regex_pat_list: A list of compiled regular expressions (e.g., [r'[a-z]', r['0-9']])
    :param pattern_keys: A list of single-character strings that are used to indicate the corresponding items in regex_pat_list  (e.g., ['a', '0']). The regex_pat_list and pattern_keys lists must be the same length.
    :param regex_pat_other: A regular expression that tests for the occurrence of anything not covered by any entry in the regex_pat_list (e.g., r'[^a-z0-9]')
    
    :return A list of the patterns that occur in the series, starting with the pattern_keys for patterns that occur in the series, and then followed by any other characters that matched regex_pat_other.
    """
    function_name = 'get_column_patterns()'
    logger = logging.getLogger('QCprofiling')
    logger.debug("%s,column,%s" %(function_name, series.name))

    does_contain = [False for l1 in range(len(regex_pat_list))]
    pattern_dict = {}
    l1 = 0
    
    for l1 in range(len(series)):
        # If null values are checked then Pandas seems to read the actual entry, so 'NULL' in a float series is classed as containing alphabetic characters!
        if pd.notnull(series.values[l1]):
            # Python converts each value, so a digit-only value in a string column is returned as numerical
    
            # BUT INEFFICIENT to convert every value - need to modify the code below    
            value = str(series.values[l1])
            
            if len(value) > 0:
                # Check for each of the patterns in regex_pat_list
                for l2 in range(len(does_contain)):
                    if re.search(regex_pat_list[l2], value):
                        does_contain[l2] = True
                    
                # Check for the regex_pat_other patterns
                other_list = re.findall(regex_pat_other, value)
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




