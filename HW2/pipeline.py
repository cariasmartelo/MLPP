
'''
CAPP 30254 1 Machine Learning for Public Policy
Pipeline functions for HW2
Spring 2019
Professor: Rayid Ghani
Camilo Arias
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sodapy import Socrata
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint, shape
import chi_opdat

def load_credit_data(csv_file_path):
    '''
    Load csv file with credit data to a Pandas DataFrame.
    Inputs:
        csv_file: Str
    Output:
        Pandas DataFrame
    '''
    dtypes = {'PersonID': int,
              'SeriousDlqin2yrs': int,
              'RevolvingUtilizationOfUnsecuredLines': float,
              'age': int,
              'zipcode': int,
              'NumberOfTime30-59DaysPastDueNotWorse': int,
              'DebtRatio': float,
              'MonthlyIncome': float,
              'NumberOfOpenCreditLinesAndLoans': int,
              'NumberOfTimes90DaysLate': int,
              'NumberRealEstateLoansOrLines': int,
              'NumberOfTime60-89DaysPastDueNotWorse': int,
              'NumberOfDependents': float}

    credit_df = pd.read_csv(csv_file_path, dtype=dtypes)

    return credit_df

def load_zipcode_area():
    '''
    Load 2017 and 2018 Chicago crime data from City of Chicago Open Data portal
    using Socrata API.
    Input:
        Limit: int
    Output:
        Pandas Data Frame
    '''
    ziparea_df = chi_opdat.load_data(chi_opdat
                                          .API_ENDPOINTS['ZIP_CODE_BOUNDARIES'])
    ziparea_df.rename(columns={'zip' : 'zipcode'}, 
                                inplace = True)
    ziparea_gdf = chi_opdat.convert_to_geopandas(ziparea_df)
    return ziparea_gdf

def see_histograms(credit_df, columns=None):
    '''
    Produce histograms of the numeric columns in credit_df. If columns is
    specified, it produces histograms of those columns.
    Inputs:
        credit_df: Pandas DataFrame
        columns: [colname]
    Output:
        Individual Graphs (Num of graphs = Num of numeric cols)
    '''
    plt.clf()
    figs = {}
    axs = {}
    if not columns:
        columns = credit_df.columns
    for column in columns:
        if not credit_df[column].dtype.kind in 'ifbc':
            continue
        if credit_df[column].dtype.kind in 'if':
            num_bins = 20
        if credit_df[column].dtype.kind in 'bc':
            num_bins = credit_df[column].nunique()

        figs[column] = plt.figure()
        axs[column] = figs[column].add_subplot(111)
        n, bins, patches = axs[column].hist(credit_df[column], num_bins,
                                            facecolor='blue', alpha=0.5)
        axs[column].set_title(column)
    plt.show()

def see_summary_stats(credit_df, columns=None):
    '''
    Return summary statistics for all columns credit df. Is columns specified,
    produces summary statistics of those columns.
    Input:
        credit_df: Pandas DataFrame
    Output:
        Print
    '''
    if not columns:
        print(credit_df.describe())
    else:
        print(credit_df[columns].describe())

def see_scatterplot(credit_df, xcol, ycol, colorcol=None):
    '''
    Print scatterplot of columns specified of the credit df. If color column
    is specified, the scatterplot will be colored by that column.
    Input:
        credit_df: Pandas DataFrame
        xcol: String
        ycol: String
        colorcol: String
    Output:
        Graph
    '''
    plt.clf()
    if not colorcol:
        credit_df.plot.scatter(x=xcol, y=ycol, legend=True)
    else:
        credit_df.plot.scatter(x=xcol, y=ycol, c=colorcol, colormap='viridis',
                               legend=True)
    plt.title('Scatterplot of Credit DataFrame \n {} and {}'
                  .format(xcol, ycol))
    plt.show()

def map(credit_df, zip_gdf, colorcol, funct='mean'):
    '''
    Map by zip code the value of the column indicated in colorcol aggregated
    with the function specified.
    Inputs:
        credit_df: Pandas DataFrame
        zip_gdf: ZipCode Boundaries GeodataFrame
        colorcol: Str
        funct: str


