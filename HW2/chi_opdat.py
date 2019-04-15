'''
CAPP 30254 1 Machine Learning for Public Policy
Chicago Open Data Portal functions for Pipeline
Spring 2019
Professor: Rayid Ghani
Camilo Arias
'''
import pandas as pd
import numpy as np
from sodapy import Socrata
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint, shape


CHICAGO_OPEN_DATA = "data.cityofchicago.org"
API_ENDPOINTS = {
    "CRIME_ENDPOINT": "6zsd-86xi",
    "COMMUNITY_ENDPOINT": "igwz-8jzy",
    "TRACT_ENDPOINT": "74p9-q2aq",
    "ZIP_CODE_BOUNDARIES": 'unjd-c2ca'
}

def load_data(api_endpoint, limit=10000):
    '''
    Load data from Chicago Open Data portal using Socrata API and the api_endpoint. If
    limit is specified, load no more than limit number of observations.
    Input:
        api_endpoint: str
        limit: int
    Output:
        Pandas Data Frame
    '''

    client = Socrata(CHICAGO_OPEN_DATA, None)
    data_dict= client.get(api_endpoint, limit=limit)

    data_df = pd.DataFrame.from_dict(data_dict)
    if 'the_geom' in data_df.columns:
        data_df.rename(columns={'the_geom' : 'location'}, 
                                inplace = True)
    return data_df


def convert_to_geopandas(df):
    '''
    Converts the pandas dataframe to geopandas DataFrame
    Inputs:
        df: Pandas DataFrame
    Output:
        Geopandas DataFrame
    '''
    def shape_(x):

        '''
        Convert JSON location attribute to shapely.
        '''
        if isinstance(x, float):
            return np.NaN
        return shape(x)


    df['geometry'] = df.location.apply(shape_)
    geo_df = gpd.GeoDataFrame(df, crs = 'epsg:4326', geometry = df['geometry'])

    return geo_df
