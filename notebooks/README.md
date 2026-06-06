# Notebooks
The vizdataquality package includes five types of notebook, each in a separate subfolder.

## Calculations
- [Calculate a set of data quality attributes and output them to a file](https://github.com/royruddle/vizdataquality/blob/main/notebooks/calculate/Simple%20example.ipynb)

## Visualizations
There is a separate notebook for each type of visualization.
Some of them first do the above calculations and then visualize the output.
Others create visualizations directly from a dataframe containing the dataset.
- Visualizing missing values with a [bar chart](https://github.com/royruddle/vizdataquality/blob/main/notebooks/visualize/Bar%20chart%20(missing%20values).ipynb)
- Visualizing [datetime distributions at multiple levels of detail](https://github.com/royruddle/vizdataquality/blob/main/notebooks/visualize/Datetime%20value%20distribution.ipynb)
- Visualizing value lengths (the number of characters in values) with a [dot & whisker chart](https://github.com/royruddle/vizdataquality/blob/main/notebooks/visualize/Dot%20%26%20whisker%20(value%20lengths).ipynb)
- Visualizing value counts (the number of times each value occurs in a variable) with:
  - [lollipop chart](https://github.com/royruddle/vizdataquality/blob/main/notebooks/visualize/Lollipop%20(value%20counts).ipynb)
  - [stacked bar chart](https://github.com/royruddle/vizdataquality/blob/main/notebooks/visualize/Stacked%20bar%20chart%20(value%20counts).ipynb)
  - [line chart](https://github.com/royruddle/vizdataquality/blob/main/notebooks/visualize/Line%20chart%20(value%20counts).ipynb), showing any gaps in a sequence of values
- Displaying attributes such as data types in:
  - [table](https://github.com/royruddle/vizdataquality/blob/main/notebooks/visualize/Table%20(data%20type%2C%20example%20value).ipynb)
  - [text](https://github.com/royruddle/vizdataquality/blob/main/notebooks/visualize/Text%20(data%20type).ipynb)
- Visualizing numerical distributions with:
  - [boxplot](https://github.com/royruddle/vizdataquality/blob/main/notebooks/visualize/Box%20plot%20(value%20distribution).ipynb)
  - [histogram](https://github.com/royruddle/vizdataquality/blob/main/notebooks/visualize/Histogram%20(value%20distribution).ipynb)
  - [violin chart](https://github.com/royruddle/vizdataquality/blob/main/notebooks/visualize/Violin%20(value%20distribution).ipynb)
- Visualizing numerical and datetime distributions with a [box plot](https://github.com/royruddle/vizdataquality/blob/main/notebooks/visualize/Box%20plot%20(numeric%2C%20date%20%26%20time).ipynb)

## Report
- [Create a report](https://github.com/royruddle/vizdataquality/blob/main/notebooks/report/Report.ipynb) while you investigate data quality and profile a dataset

## Workflow
- Apply a [six-step workflow](https://github.com/royruddle/vizdataquality/blob/main/notebooks/workflow/Workflow%20(parking%20fines).ipynb) to investigate data quality and profile an open parking fines dataset

## Missing data structures
- Investigating a simple [monotone pattern of missing values](https://github.com/royruddle/vizdataquality/blob/main/notebooks/missingdatastructure/missing%20data%20structure%201.ipynb)
- Investigating [interwoven patterns of missing values](https://github.com/royruddle/vizdataquality/blob/main/notebooks/missingdatastructure/missing%20data%20structure%202.ipynb)
- For large datasets, [reading data in chunks from a file](https://github.com/royruddle/vizdataquality/blob/main/notebooks/missingdatastructure/missing%20data%20from%20file.ipynb)
- For saving your analysis, [importing and exporting](https://github.com/royruddle/vizdataquality/blob/main/notebooks/missingdatastructure/missing%20data%20import%20and%20export.ipynb)



