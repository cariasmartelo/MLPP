
'''
CAPP 30254 1 Machine Learning for Public Policy
HW1
Spring 2019
Professor: Rayid Ghani
Camilo Arias
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sodapy import Socrata
import geopandas as gpd

CRIME_CLASS = \
    {'Violent Crime': ['01A', '02', '03', '04A', '04B'],
     'Property Crime': ['05', '06', '07', '09'],
     'Less serious offences': ['01B', '08A', '08B', '10', '11', '12', '13',\
                               '14', '15', '16', '17', '18', '19', '20', '22',\
                                '24', '26']}

CHICAGO_OPEN_DATA = "data.cityofchicago.org"

def load_crime_data(limit):
    '''
    Load 2017 and 2018 Chicago crime data from City of Chicago Open Data portal
    using Socrata API.
    Input:
        Limit: int
    Output:
        Pandas Data Frame
    '''

    client = Socrata(CHICAGO_OPEN_DATA, None)
    crime = client.get("6zsd-86xi", where="year = 2017 or year = 2018",
                       limit=limit)

    crime_df = pd.DataFrame.from_dict(crime)

    #Seting up types
    for col in ['year', 'ward', 'y_coordinate', 'x_coordinate', 'latitude',\
                'longitude']:
        crime_df[col] = pd.to_numeric(crime_df[col], errors = "coerce")

    for col in ['arrest', 'domestic']:
        crime_df[col] = crime_df[col].astype(bool)

    for col in ['date', 'updated_on']:
        crime_df[col] = pd.to_datetime(crime_df[col])

    #Adding clasification column
    crime_class_inv = {}
    for k, v in CRIME_CLASS.items():
        for code in v:
            crime_class_inv[code] = k

    crime_df['crime_class'] = crime_df['fbi_code'].map(crime_class_inv)

    return crime_df


##Crosstab of primary type and year
def make_cross_var_year(df, var):
    '''
    Make cross tab of type of crime and year.
    Input:
        df: Pandas DF
        var: str
    Output:
        Pandas DF
    '''
    cross_var_year = pd.crosstab(df[var], df.year, margins = True)
    cross_var_year.rename(columns={'All' : 'Total'}, index={'All' : 'Total'},\
        inplace = True)
    cross_var_year['Perc Change'] = (cross_var_year[2018] / 
                                      cross_var_year[2017] - 1) * 100
    cross_var_year['Perc Change'] = cross_var_year['Perc Change'].round(2)
    cross_var_year = cross_var_year[['Total', 2017, 2018, 'Perc Change']]
    cross_var_year.replace(float('inf'), np.NaN, inplace = True)
    cross_var_year.sort_values('Total', ascending = False)
    cross_var_year.index = cross_var_year.index.str.capitalize()
    cross_var_year.rename_axis(var.upper().replace("_", " "), inplace = True)
    cross_var_year.rename_axis("", axis = 1,  inplace = True)
    #cross_class_year.sort_values('Total', ascending = False, inplace = True)

    return cross_var_year

def graph_crimes_year(df, crime_class):
    '''
    Bar graph of type of crime of classificaiton given by year
    Input:
        df: Pandas DF
        crime_class: str
    Output:
        Displays graph
    Addapted code from
    https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html
    https://preinventedwheel.com/easy-matplotlib-bar-chart/
    '''
    cross_tab = pd.crosstab([df.crime_class, df.primary_type], df.year)
    df = cross_tab.loc[crime_class] 
    ind = np.arange(len(df))  # the x locations for the groups
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, df[2017], width,
                    color='SkyBlue', label=2017)
    rects2 = ax.bar(ind + width/2, df[2018], width,
                    color='IndianRed', label=2018)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Freq')
    ax.set_title("{}s in Chicago (2017 "
                 "and 2018) and % change"
                 .format(crime_class.replace("_", " ")))
    ax.set_xticks(ind)
    ax.set_xticklabels(df.index.str.capitalize())
    ax.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()

    perc_change = ((df[2018] / df[2017] - 1) * 100).round(2)
    perc_change = ["{} %".format(x) for x in perc_change]
    pairs = len(df)
    make_pairs = zip(*[ax.get_children()[:pairs],ax.get_children()\
                 [pairs:pairs*2]])
    for i,(left, right) in enumerate(make_pairs):
        ax.text(i,max(left.get_bbox().y1,right.get_bbox().y1)+2,
                perc_change[i], horizontalalignment ='center')

    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x,
                                       loc: "{:,}".format(int(x))))
    ax.patch.set_facecolor('#FFFFFF')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(1)

    plt.show()


def load_community_area_data():
    '''
    Load 2017 and 2018 Chicago crime data from City of Chicago Open Data portal
    using Socrata API.
    Input:
        Limit: int
    Output:
        Pandas Data Frame
    '''

    client = Socrata(CHICAGO_OPEN_DATA, None)
    community_area= client.get("igwz-8jzy")

    community_area_df = pd.DataFrame.from_dict(community_area)

    return community_area_df





