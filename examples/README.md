# Examples
This folder contains example datasets and reports. Details are as follows.
## Simple example
The output from [Report.ipynb](https://github.com/royruddle/vizdataquality/blob/main/notebooks/Report.ipynb) shows how that class is used.
## Parking fines
The dataset and output from ["Workflow (parking fines).ipynb"](https://github.com/royruddle/vizdataquality/blob/main/notebooks/Workflow%20(parking%20fines).ipynb) shows a 6-step workflow for investigating data quality issues and profiling a dataset.
### 1. Look at your data (is anything obviously wrong?)
Bar charts reveal two unexpected issues: (a) the dataset contains a strange variable (Unnamed: 8) that is empty, and (b) PCN (the penalty charge notice identifier) is not unique. The empty variable issue occurs because each row of the data file ends with a comma, so we clean the data by removing that column. The second issue indicates that one of our assumptions about the data was wrong. The rows do not have unique PCNs.
### 2. Watch out for special values
Some datasets use values with a special meaning, e.g., 999999 to indicate an integer value that is missing. We check this by visualizing the range and distribution of all the date/time and numerical variables with box plots. The date ISSUED and Total Paid distributions are very skewed, and the interquartile interval of Balance is very small. However, after listing the unique values we conclude that none of them are special.
### 3. Is any data missing?
Apart from the empty variable that we have already removed, there are no missing values.
### 4. Check each variable
First, we use a dot & whisker plot to visualize value lengths of all the variables. This shows a dot if every value of a variable has the same number of characters, and a whisker if the number of characters varies. LOCATION and CONTRAVENTION have suspiciously large value lengths ranges. That occurs because some values have trailing spaces, so we clean the data by trimming the values.

Next we use lollipop plots to visualize value counts (the number of times each value occurs for a variable). This shows that there are two values of FINE, but an unexpected number of different Balance and Total Paid values. In addition, while a third of the PCNs appear once, the others appear as many as five times.

With a single function call, vizdataquality automatically visualizes date/time data at seven different levels of detail. That shows that ISSUED and Last Pay Date only contain year, month and day (not time stamps).
### 5. Check combinations of variables
Start this step by asking “what are your assumptions or business rules?”. Then investigate whether any of those assumptions/rules are violated.

In this dataset, it would be logical to expect FINE - Total Paid = Balance. However, that is only true for 11% of the records. For thousands of other records the difference is the whole value of the FINE (£50 or £75) or half its value (£25 or £35). Other records have unusual differences (e.g., £49.06).
### 6. Profile the cleaned data
This example uses:
- Bar charts to visualize the number of unique values in each variable, and that they are all complete.
- A stacked bar chart to visualize the number of PCNs that occur 1 – 5 times.
- Lollipop plots to visualize value counts for LOCATION and CONTRAVENTION.
- Line plots to visualize the year, month and day of the week distributions of ISSUED and Last Pay Date.
- Lollipop plots to visualize value counts for FINE, Balance and Total Paid.
- A table summarising the relationship between FINE, Balance and Total Paid for different time periods.
##Import/export
The data files that are exported and imported by the [importing and exporting](https://github.com/royruddle/vizdataquality/blob/main/notebooks/missing%20data%20import%20and%20export.ipynb) missing data example.
## Import/export
The data files that are used by the [importing and exporting](https://github.com/royruddle/vizdataquality/blob/main/notebooks/missing%20data%20import%20and%20export.ipynb) missing data example.
