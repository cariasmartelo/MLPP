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
from sklearn import tree

def load_from_csv(csv_file_path):
    '''
    Load csv file with project data to a Pandas DataFrame.
    Inputs:
        csv_file: Str
    Output:
        Pandas DataFrame
    '''
    dtypes = {'projectid': str,
              'teacher_acctid': str,
              'schoolid ': str,
              'school_ncesid': str,
              'school_latitude': float,
              'school_longitude': float,
              'school_city': str,
              'school_state': str,
              'school_metro': str,
              'school_district': str,
              'school_county': str,
              'school_charter': str,
              'school_magnet': str,
              'teacher_prefix': str,
              'primary_focus_subject': str,
              'primary_focus_are': str,
              'secondary_focus_subject': str,
              'secondary_focus_area': str,
              'resource_type': str,
              'poverty_level': str,
              'grade_level': str,
              'total_price_including_optional_support': float,
              'students_reached': float,
              'eligible_double_your_impact_match': str,
              'date_posted': str,
              'datefullyfunded': str}
    to_boolean = {'f': False,
                  't': True}

    projects_df = pd.read_csv(csv_file_path, dtype=dtypes)
    projects_df['date_posted'] = pd.to_datetime(projects_df['date_posted'],
                                                format='%m/%d/%y')
    projects_df['datefullyfunded'] = pd.to_datetime(projects_df['datefullyfunded'],
                                                    format='%m/%d/%y')
    for col in ['school_charter', 'school_magnet',\
                'eligible_double_your_impact_match']:
        projects_df[col] = projects_df[col].map(to_boolean)

    return projects_df


def create_outcome_var(projects_df, days):
    '''
    Create variable True if project was fully funded within days days.
    Inputs:
        projects_df: Pandas DataFrame
        days: int
    Output:
        Pandas DataFrame
    '''
    df = projects_df.loc[:]
    df['days_untill_funded'] = (df['datefullyfunded'] - df['date_posted']).dt.days
    outcome_var = "funded_in_{}_days".format(days)
    df[outcome_var] = df['days_untill_funded'] <= days

    return df



