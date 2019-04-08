
'''
CAPP 30254 1 Machine Learning for Public Policy
HW1
Spring 2019
Professor: Rayid Ghani
Camilo Arias
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sodapy import Socrata

CRIME_CLASS = \
    {'Violent Crime': ['01A', '02', '03', '04A', '04B'],
     'Property Crime': ['05', '06', '07', '09'],
     'Less serious offences': ['01B', '08A', '08B', '10', '11', '12', '13',\
                               '14', '15', '16', '17', '18', '19', '20', '22',\
                                '24', '26']}

def load_crime_data(limit):
    '''
    Load 2017 and 2018 Chicago crime data from City of Chicago Open Data portal
    using Socrata API.
    Input:
        Limit: int
    Output:
        Pandas Data Frame
    '''

    client = Socrata("data.cityofchicago.org", None)
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
def make_cross_type_year(df):
    '''
    Make cross tab of type of crime and year.
    Input:
        df: Pandas DF
    Output:
        Pandas DF
    '''
    cross_type_year = pd.crosstab(df.primary_type, df.year, margins = True)
    cross_type_year.rename(columns={'All' : 'Total'}, index={'All' : 'Total'}, inplace = True)
    cross_type_year['Perc Change'] = (cross_type_year[2018] / cross_type_year[2017] - 1) * 100
    cross_type_year['Perc Change'] = cross_type_year['Perc Change'].round(2)
    cross_type_year = cross_type_year[['Total', 2017, 2018, 'Perc Change']]
    cross_type_year.replace(float('inf'), np.NaN, inplace = True)
    cross_type_year.sort_values('Total', ascending = False)
    cross_type_year.index = cross_type_year.index.str.capitalize()
    cross_type_year.rename_axis("TYPE OF CRIME", inplace = True)  
    cross_type_year.rename_axis("", axis = 1,  inplace = True)
    cross_type_year.sort_values('Total', ascending = False, inplace = True)

    return cross_type_year



def graph_cross_type_year(cross_type_year):
    '''
    Bar graph of type of crime by year
    '''
    plt.clf
    df = cross_type_year.iloc[1:21]
    ind = np.arange(len(df))  # the x locations for the groups
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, df[2017], width,
                    color='SkyBlue', label=2017)
    rects2 = ax.bar(ind + width/2, df[2018], width,
                    color='IndianRed', label=2018)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Freq')
    ax.set_title('Frequency of 20 most common types of crimes in Chicago (2017 and 2018)')
    ax.set_xticks(ind)
    ax.set_xticklabels(df.index)
    ax.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()







