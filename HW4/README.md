# Machine Learning for Public Policy - Spring 2019

## Homework 5

Improving Pipeline of Homework 3.

Improvements:

-Feature generation after split

-Includes Bagging

-Calculares precision and recall for top-k% and not for absolute threshold. The same for the plots.

-Uses one function called run_model to build any classifier

-Runs every classifier with different parameters


### Organization:
- report.pdf: PDF report of prediction.

- prediction_grid.csv: csv of table of models, threshold, cross validation and evaluation metrics.

- prediction.ipynb: Jupyter notebook used to run prediction of school project
with cross validationl. It produces the table needed. Uses prediction.py, pipeline.py and classifiers.py.

- pipeline_demo.ipynb: Demo of pipeline improved usage. Uses pipeline.py and classifiers.py

- prediction.py: Functions to run prediction with cross validation.

- pipeline.py: Main functions to load and clean dat for the pipeline.

- classifiers.py: Script with functions to build classifiers.

- projects_2012_2013.csv: Has the projects csv.

- us_shapefile: Simple US shapefile


### Libraries used:
- Numpy 1.15.4

- Pandas 0.24.2

- Geopandas 0.4.0+67.g08ad2bf

- sodapy 1.5.2

- matplotlib 3.0.3

- shapely 1.6.4.post2

- scikit-learn 0.20.3


