
'''
CAPP 30254 1 Machine Learning for Public Policy
HW1
Spring 2019
Professor: Rayid Ghani
Camilo Arias
'''
import pandas as pd
import numpy as np
from sodapy import Socrata

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
    dtypes = {'arrest': np.bool_,
              'beat': str,
              'block': str,
              'case_number': str,
              'community_area': str,
              'description': str,
              'district': str,
              'domestic': bool,
              'fbi_code': str,
              'id': str,
              'iucr': str,
              'latitude': float,
              'location_description': str,
              'longitude': float,
              'primary_type': str,
              'ward': np.int,
              'x_coordinate': float,
              'y_coordinate': float,
              'year': int}

    crime_df = pd.DataFrame.from_dict(crime)
    crime_df = crime_df.astype(dtypes)
    crime_df['date'] = pd.to_datetime(crime_df['date'])
    crime_df['updated_on'] = pd.to_datetime(crime_df['updated_on'])


    return crime_df
