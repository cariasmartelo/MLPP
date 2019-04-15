
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

OUTCOME_VAR = 'SeriousDlqin2yrs'

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
    ziparea_df['zipcode'] = pd.to_numeric(ziparea_df['zipcode'])
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
        credit_df.plot.scatter(x=xcol, y=ycol, c=colorcol, cmap='viridis',
                               legend=True)
    plt.title('Scatterplot of Credit DataFrame \n {} and {}'
                  .format(xcol, ycol))
    plt.show()


def summary_by_objective(credit_df, funct='mean'):
    '''
    See data by Objective column, aggregated by function.
    Input:
        credit_df: Pandas DataFrame
        funct: str
    Output:
        Pandas DF
    '''
    return credit_df.groupby(OUTCOME_VAR).agg('mean').T


def map(credit_df, zip_gdf, colorcol=OUTCOME_VAR, funct='mean',
        count=False):
    '''
    Map by zip code the value of the column indicated in colorcol aggregated
    with the function specified.
    Inputs:
        credit_df: Pandas DataFrame
        zip_gdf: ZipCode Boundaries GeodataFrame
        colorcol: Str
        funct: str
    Output:
        Map
    '''
    if count:
        credit_by_zip = credit_df.groupby('zipcode').size().reset_index()
        credit_by_zip.rename(columns={0:'count'}, inplace = True)
        colorcol = 'count'

    else:
        credit_by_zip = credit_df.groupby('zipcode').agg({colorcol: funct})
    
    credit_gdf = zip_gdf.merge(credit_by_zip, on='zipcode', how='left')
    ax = credit_gdf.plot(color="grey")
    credit_gdf.dropna().plot(ax=ax, column=colorcol, cmap='viridis',
                             legend=True)
    if count:
        ax.set_title('Frequency of reports in Chicago by zipcode\n'
                     '(Zip codes without data in grey)')
    else:
        ax.set_title('{} {} in Chicago by zipcode\n(Zip codes without data'
                     ' in grey)'.format(funct.capitalize(), colorcol))
    plt.show()


