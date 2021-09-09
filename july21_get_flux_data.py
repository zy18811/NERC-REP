"""
Constructs a dataframe containing flux data for July 2021
"""

import pandas as pd
import datetime


def get_july21_flux_data(raw_or_interp='raw'):
    """
    Gets flux data for July 2021 and returns a dataframe
    :param raw_or_interp: raw_or_interp = 'raw' returns a dataframe with the raw data
                          raw_or_interp = 'interp' returns a dataframe with the data interpolated to 30min intervals
    :return: dataframe
    """

    # Reads in csv with July 2021 flux data
    df = pd.read_csv("Hanger Field July 2021/Flux Data/July_flux_data_HF.csv",parse_dates=[['date','time']],dayfirst=True)
    df = df[:-1]

    # Selects time and flux columns
    df = df[['date_time','Tau','H','LE','co2_flux','h2o_flux']]

    """
    resamples data at a 30min frequency
    """
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.set_index('date_time')

    ix = pd.date_range(datetime.datetime(2021, 7, 1, 0, 30, 0, 0), datetime.datetime(2021, 7, 31, 23, 30, 0, 0),
                       freq='30min')

    df = df.reindex(ix)

    if raw_or_interp == 'raw':
        return df
    # interpolates data
    elif raw_or_interp == 'interp':
        df = df.apply(pd.Series.interpolate, method='time',limit_direction = 'both')
        return df


if __name__ == '__main__':
    """
    prints dataframe
    """
    print(get_july21_flux_data('interp'))