
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
from sklearn import tree
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


def restrict_df(credit_df, restrict):
    '''
    Will return a DataFrame restricted by the percentile values indicated
    in the restrict dictionary.
    Inputs:
        credit_df: Pandas DataFrame
        restirc: dict
    Output:
        Pandas DataFrame
    '''
    restricted_df = credit_df
    for column in restrict:
        min_val = credit_df[column].quantile(restrict[column][0])
        max_val = credit_df[column].quantile(restrict[column][1])
        restricted_df = restricted_df.loc[(
            (restricted_df[column] <= max_val)
            | (restricted_df[column].isna())) 
            & ((restricted_df[column] >= min_val)
            | (restricted_df[column].isna()))]

    return restricted_df


def see_histograms(credit_df, columns=None, restrict=None):
    '''
    Produce histograms of the numeric columns in credit_df. If columns is
    specified, it produces histograms of those columns. If restrict dictionary
    is specified, restricts to the values inside the percentile range specified.
    Inputs:
        credit_df: Pandas DataFrame
        columns: [colname]
        restrict = dict
    Output:
        Individual Graphs (Num of graphs = Num of numeric cols)
    '''
    plt.clf()
    figs = {}
    axs = {}
    if not restrict:
        restrict = {}
    if not columns:
        columns = credit_df.columns
    for column in columns:
        if not credit_df[column].dtype.kind in 'ifbc':
            if credit_df[column].nunique() <= 15:
                credit_df[column].value_counts(sort=False).plot(kind='bar')
                plt.tight_layout()
                plt.title(column)
                continue
        if credit_df[column].dtype.kind in 'c':
            credit_df[column].value_counts().plot(kind='bar')
            continue
        if column in restrict:
            min_val = credit_df[column].quantile(restrict[column][0])
            max_val = credit_df[column].quantile(restrict[column][1])
            col_to_plot = (credit_df.loc[(credit_df[column] <= max_val)
                             & (credit_df[column] >= min_val), column])
        else:
            col_to_plot = credit_df[column]

        num_bins = min(20, col_to_plot.nunique())

        figs[column] = plt.figure()
        axs[column] = figs[column].add_subplot(111)
        n, bins, patches = axs[column].hist(col_to_plot, num_bins,
                                            facecolor='blue', alpha=0.5)
        axs[column].set_title(column)
    plt.show()


def see_summary_stats(credit_df, columns=None, percentiles=None):
    '''
    Return summary statistics for all columns credit df. Is columns specified,
    produces summary statistics of those columns.
    Input:
        credit_df: Pandas DataFrame
    Output:
        Print
    '''
    if not percentiles:
        percentiles = [.25, .5, .75]
    if not columns:
        print(credit_df.describe(percentiles=percentiles))
    else:
        print(credit_df[columns].describe(percentiles=percentiles))


def see_scatterplot(credit_df, xcol, ycol=OUTCOME_VAR, colorcol=None, logx=False,
                    logy=False, xjitter=False, yjitter=False):
    '''
    Print scatterplot of columns specified of the credit df. If color column
    is specified, the scatterplot will be colored by that column.
    Input:
        credit_df: Pandas DataFrame
        xcol: String
        ycol: String
        colorcol: String
        logx, logy: bool
        xiitter, yitter: bool
    Output:
        Graph
    '''
    df_to_plot = credit_df.loc[:]
    if xjitter:
        df_to_plot[xcol] = df_to_plot[xcol] +\
            np.random.uniform(-0.5, 0.5, len(df_to_plot[xcol]))\
            *df_to_plot[xcol].std()
    if yjitter:
        df_to_plot[ycol] = df_to_plot[ycol] +\
            np.random.uniform(-0.5, 0.5, len(df_to_plot[ycol]))\
            *df_to_plot[ycol].std()

    plt.clf()
    if not colorcol:
        df_to_plot.plot.scatter(x=xcol, y=ycol, legend=True, logx=logx,
                               logy=logy)
    else:
        df_to_plot.plot.scatter(x=xcol, y=ycol, c=colorcol, cmap='viridis',
                               legend=True, logx=logx, logy=logy)
    plt.title('Scatterplot of Credit DataFrame \n {} and {}'
                  .format(xcol, ycol))
    plt.show()


def summary_by_objective(credit_df, funct='mean', count=False):
    '''
    See data by Objective column, aggregated by function.
    Input:
        credit_df: Pandas DataFrame
        funct: str
    Output:
        Pandas DF
    '''
    if not count:
        sum_table =  credit_df.groupby(OUTCOME_VAR).agg('mean').T
    else:
        sum_table = credit_df.groupby(OUTCOME_VAR).size().T
    sum_table['perc diff'] = ((sum_table[1] / sum_table[0]) -1) * 100

    return sum_table


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


def fillna(credit_df, columns=None, funct='mean'):
    '''
    Fill the np.nan values of the DataFrame with the function specified.
    If column is specified, only fills that column.
    Inputs:
        credit_df: Pandas DataFrame
        columns: [str]
        funct: str
    Output:
        Pandas DataFrame
    '''
    if not columns:
        return credit_df.fillna(credit_df.agg(funct))
    filled_df = credit_df[:]
    for column in columns:
        filled_df[column] = (filled_df[column]
                             .fillna(filled_df[column].agg(funct)))
    return filled_df


def discretize(serie, bins, equal_width=False, string=False):
    '''
    Function to discretize a pandas series based on number of bins and
    wether bins are equal width, or have equal number of observations.
    If string = True, returns string type.
    Inputs:
        serie: Pandas Series
        bins: int(num of bins)
        equal_width: bool
    Output:
        Pandas Series.
    '''
    if not equal_width:
        return_serie =  pd.qcut(serie, bins)
    else:
        return_serie = pd.cut(serie, bins)
    if string:
    	return_serie = return_serie.astype(str)

    return return_serie


def make_dummies_from_categorical(serie, dummy_na=False):
    '''
    Function that takes a categorical Pandas series and returns a Pandas
    Dataframe of dummy variables. If dummy_na True, make dummy for NA values.
        serie: Pandas Series
        dummy_na: bool
    Output:
        Pandas Data Frame.
    '''
    return pd.get_dummies(serie, serie.name, dummy_na=  dummy_na)


def build_tree_classifier(train_df, xcol, ycol=OUTCOME_VAR):
    '''
    Fit a decision tree model to the data provided, given the ycol and the set
    of x cols.
    Inputs:
        credit_df: Pandas DataFrame
        ycol: string
        xcol: [str]
    Output:

    Classifier
    '''
    assert not (train_df[xcol].isna().any().any()), "No NaN values in x allowed"
    assert not (train_df[ycol].isna().any().any()), "No NaN values in y allowed"

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_df[xcol], train_df[ycol])
    return clf


def obtain_predicted_probabilities(tree, test_df, xcol):
    '''
    Obtain predicted probabilities of success for test data given a tree
    classifier and a list of the x columns. 
    '''
    return tree.predict_proba(test_df[xcol])[:,1]


def compute_accuracy(yvar_real, predict_proba, threshold=0.5):
    '''
    Compute the accuracy of a predition based on the predicted probabilities,
    the real value of y and the threshold to consider a predicted success.
    Inputs:
        yvar_real: Pandas Series
        predict_proba: Pandas Series
        threshold: floar
    Returns:
        Pandas DataFrame
    '''
    predict_proba = pd.Series(predict_proba)
    yvar_real = yvar_real.to_frame().reset_index().iloc[:,1]
    results = pd.DataFrame(index=np.arange(0, len(yvar_real)))
    results['yvar_real'] = yvar_real
    results['predict_proba'] = predict_proba
    results['yvar_predicted'] = np.where(results['predict_proba']
                                    > threshold, 1, 0)
    results['correct'] = np.where(results['yvar_predicted'] == results['yvar_real'],
                                  'Correct', 'Incorrect')
    return pd.crosstab(results['yvar_real'], results['correct'], margins=True,
                       normalize='index')
