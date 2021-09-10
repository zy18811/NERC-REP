"""
Constructs a dataframe containing COSMOS data for July 21
"""
import pandas as pd


def get_july21_cosmos_data(raw_or_interp = 'raw'):
    """
    Gets COSMOS data for July 21 and returns a dataframe
    :param raw_or_interp: raw_or_interp = 'raw' returns a dataframe with the raw data
                          raw_or_interp = 'interp' returns a dataframe with the data interpolated to 30min intervals
    :return: dataframe
    """
    # reads column names
    col_names = pd.read_csv("Hanger Field July 2021/COSMOS Data/COSMOS and Hanger Field Data July 2021.csv",
                            usecols=[i for i in range(1, 28)], nrows=0).columns
    col_names = col_names.insert(0, 'date_time')

    # reads csv containing COSMOS data for July 2021
    df = pd.read_csv("Hanger Field July 2021/COSMOS Data/COSMOS and Hanger Field Data July 2021.csv",
                     header=0, usecols=[i for i in range(0,28)], names=col_names, skiprows=4,
                     parse_dates=['date_time'], dayfirst=True)

    # drops empty column
    df = df.drop('SNOWD_DISTANCE_COR_LEVEL2',axis = 1)
    df = df.set_index('date_time')
    df.index.name = None

    if raw_or_interp == 'raw':
        return df
    # interpolates data
    elif raw_or_interp =='interp':
        df = df.apply(pd.Series.interpolate, method='time',limit_direction = 'both')
        return df


if __name__ == '__main__':
    """
    prints dataframe
    """
    print(get_july21_cosmos_data('interp'))
