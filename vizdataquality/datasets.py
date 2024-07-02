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
import datetime
import math


# =============================================================================
# Functions
# =============================================================================
def get_dataset(option):
    """Get a dataset as a data frame.

    Parameters
    ----------
    option : str
        'simple', 'missing 1', 'numeric 1', 'date 1', 'time 1', 'datetime' or 'value counts 1'

    Returns
    -------
    int, dataframe
        Number of rows in the dataset and a dataframe containing it.

    """
    df = pd.DataFrame()
    
    if option == 'simple':
        data = {}
        data['String'] = ['a'] * 3 + ['0'] * 3 + ['B1234'] * 2 + [np.nan] * 2
        data['Integer'] = list(range(0, 10))
        data['Float'] = [x/10 for x in range(6)] + [np.nan] * 4
        data['Date'] = [datetime.datetime(x, 1, 1) for x in range(2000, 2008)] + [np.nan] * 2
        #data['Date months'] = [datetime.datetime(1999, x, 8) for x in range(1, 11)]
        #data['Date days'] = [datetime.datetime(1989, 3, x) for x in range(10, 20)]
        df = pd.DataFrame.from_dict(data)
    elif option == 'missing 1':
        num_records = 10000
        data = {}
        # All values present
        data['A'] = [1] * num_records
        # Only one value missing
        data['B'] = [1] * (num_records - 1) + [np.nan]
        # Half of the values are present
        data['C'] = [1] * int(num_records/2) + [np.nan] * int(num_records/2)
        # Only one value present
        data['D'] = [1] + [np.nan] * (num_records - 1)
        # All values missing
        data['E'] = [np.nan] * num_records

        df = pd.DataFrame.from_dict(data)
    elif option == 'numeric 1':
        data = {}
        
        for l1 in range(1, 6):
            data['Numeric' + str(l1)] = [x for x in range(0, 10)]
            
        df = pd.DataFrame.from_dict(data)
    elif option == 'date 1':
        data = {}
        
        for l1 in range(1, 6):
            data['Date' + str(l1)] = [datetime.date(x, 1, 1) for x in range(2000, 2010)]
            
        df = pd.DataFrame.from_dict(data).apply(pd.to_datetime)
    elif option == 'time 1':
        data = {}
        customdate = datetime.datetime(2000, 1, 1, 0, 0)
        
        for l1 in range(1, 6):
            data['Time' + str(l1)] = [customdate + datetime.timedelta(hours=i) for i in range(0, 24)]
            
        df = pd.DataFrame.from_dict(data)#.apply(pd.to_datetime)
    elif option == 'datetime 1':
        data = {}
        array = []
        # Every day
        for day in range(365):
            # 09:00 - 17:00 hours
            for hour in range(8, 18):
                for n in range(int(10.0 * math.sin(math.radians(float(hour-8)/10.0*180.0)))):
                    # Every minute
                    for minute in range(60):
                        # Every 30 seconds
                        for seconds in [0, 30]:
                            array.append(datetime.datetime(2022, 9, 1, hour, minute, seconds) + datetime.timedelta(day))
        
        data['Date'] = array

        #data['Date'] = [datetime.date(x, 1, 1) for x in range(2000, 2010)]
            
        df = pd.DataFrame.from_dict(data).apply(pd.to_datetime)
    elif option == 'value counts 1':
        data = {}
        data['Categorical'] = ['c'] * 10 + ['b'] * 30 + ['a'] * 60
        data['Categorical (ints)'] = [1] * 80 + [3] * 5 + [4] * 15
        data['Integer'] = [12] * 10 + [11] * 30 + [-1] * 60
        data['Float'] = [0.9] * 10 + [0.2] * 30 + [0.1] * 60
        data['Date'] = ['2000-01-01'] * 10 + ['2020-01-01'] * 30 + ['2021-01-01'] * 60
        df0 = pd.DataFrame.from_dict(data)
        #print(df)
        #print(df.astype({'Categorical (ints)': 'object'}).dtypes)
        df = df0.astype({'Categorical (ints)': 'object'})
        df['Date'] = pd.to_datetime(df['Date'], yearfirst=True)
    elif option == 'value counts 2':
        num_records = 10000
        data = {}
        data['Q1'] = [True] * int(num_records / 2) + [False] * int(num_records / 2)
        data['Q2'] = [True] * (num_records - 1) + [False]
        data['Q3'] = [True] * num_records
        data['Q4'] = [True] + [False] * (num_records - 1)
        data['Q5'] = [False] * num_records
        df = pd.DataFrame.from_dict(data)
        
    return len(df), df
    