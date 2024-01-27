[![Python Package](https://github.com/royruddle/vizdataquality/actions/workflows/main.yml/badge.svg)](https://github.com/royruddle/vizdataquality/actions/workflows/main.yml)
# vizdataquality
This is a Python package for visualizing data quality, and includes this six-step workflow:
1. Look at your data (is anything obviously wrong?)
2. Watch out for special values
3. Is any data missing?
4. Check each variable
5. Check combinations of variables
6. Profile the cleaned data

## Documentation
[The vizdataquality documentation](https://vizdataquality.readthedocs.io/en/latest/index.html) is hosted on Read the Docs.

## Installation
We recommend installing vizdataquality in a python virtual environment or Conda environment.

To install vizdataquality, most users should run:

```
pip install 'vizdataquality'
```

## Tutorials
The package includes notebooks that show you how to:
- [Calculate a set of data quality attributes and output them to a file](https://github.com/royruddle/vizdataquality/blob/main/notebooks/Simple%20example.ipynb)
- Use each type of plot, e.g., [datetime value distribution](https://github.com/royruddle/vizdataquality/blob/main/notebooks/Datetime%20value%20distribution.ipynb)
- [Create a report](https://github.com/royruddle/vizdataquality/blob/main/notebooks/Report.ipynb) while you investigate data quality and profile a dataset
- [Apply the six-step workflow to an open parking fines dataset](https://github.com/royruddle/vizdataquality/blob/main/notebooks/Workflow%20(parking%20fines).ipynb)

After installing vizdataquality, to follow theses tutorials interactively you will need to clone or download this repository. Then start jupyter from within it:

```
python -m jupyter notebook notebooks
```

## Notice
The vizdataquality software is released under the Apache Licence, version 2.0. See [LICENCE](./LICENCE) for details.

## Acknowledgements
The development of the vizdataquality software was supported by funding from the Engineering and Physical Sciences Research Council (EP/N013980/1; EP/R511717/1) and the Alan Turing Institute.
