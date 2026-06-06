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
These six notebooks are the ones that illustrate the [6-step Data Quality Method](https://github.com/royruddle/6-step-data-quality-method) on YouTube:
- [Step 1 notebook](https://github.com/royruddle/vizdataquality/blob/main/notebooks/workflow/Step_1_video.ipynb) and corresponding [Step 1 video](https://www.youtube.com/watch?v=m7zjU5ojoBo)
- [Step 2 notebook](https://github.com/royruddle/vizdataquality/blob/main/notebooks/workflow/Step_2_video.ipynb) and corresponding [Step 2 video](https://www.youtube.com/watch?v=ibY5oAvSC-w)
- [Step 3 notebook](https://github.com/royruddle/vizdataquality/blob/main/notebooks/workflow/Step_3_video.ipynb) and corresponding [Step 3 video](https://www.youtube.com/watch?v=uHSxjYqwY_Q)
- [Step 4 notebook](https://github.com/royruddle/vizdataquality/blob/main/notebooks/workflow/Step_4_video.ipynb) and corresponding [Step 4 video](https://www.youtube.com/watch?v=QRo_kHNol6A)
- [Step 5 notebook](https://github.com/royruddle/vizdataquality/blob/main/notebooks/workflow/Step_5_video.ipynb) and corresponding [Step 5 video](https://www.youtube.com/watch?v=BIrYUAYY7K4)
- [Step 6 notebook](https://github.com/royruddle/vizdataquality/blob/main/notebooks/workflow/Step_6_video.ipynb) and corresponding [Step 6 video](https://www.youtube.com/watch?v=XYL80v2_vjc)

There is also a YouTube video that provides an [overview of the 6-step method](https://www.youtube.com/watch?v=qymfk1inGVg).

A 7th notebook applies the 6-step method to:
- [Investigate data quality and profile an open parking fines dataset](https://github.com/royruddle/vizdataquality/blob/main/notebooks/workflow/Workflow%20(parking%20fines).ipynb)

## Missing data structures
- Investigating a simple [monotone pattern of missing values](https://github.com/royruddle/vizdataquality/blob/main/notebooks/missingdatastructure/missing%20data%20structure%201.ipynb)
- Investigating [interwoven patterns of missing values](https://github.com/royruddle/vizdataquality/blob/main/notebooks/missingdatastructure/missing%20data%20structure%202.ipynb)
- For large datasets, [reading data in chunks from a file](https://github.com/royruddle/vizdataquality/blob/main/notebooks/missingdatastructure/missing%20data%20from%20file.ipynb)
- For saving your analysis, [importing and exporting](https://github.com/royruddle/vizdataquality/blob/main/notebooks/missingdatastructure/missing%20data%20import%20and%20export.ipynb)



