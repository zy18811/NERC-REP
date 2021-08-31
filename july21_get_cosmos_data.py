import pandas as pd
import numpy as np
import datetime


def get_july21_cosmos_data(raw_or_interp = 'raw'):
    col_names = pd.read_csv("COSMOS and Hanger Field Data July 2021.csv",usecols=[i for i in range(1,28)],nrows=0).columns
    col_names = col_names.insert(0,'date_time')
    df = pd.read_csv("COSMOS and Hanger Field Data July 2021.csv",header = 0, usecols = [i for i in range(0,28)],
                     names = col_names, skiprows=4, parse_dates=['date_time'],dayfirst=True)
    df = df.drop('SNOWD_DISTANCE_COR_LEVEL2',axis = 1)
    df = df.set_index('date_time')
    df.index.name = None
    if raw_or_interp == 'raw':
        return df
    elif raw_or_interp =='interp':
        df = df.apply(pd.Series.interpolate, method='time',limit_direction = 'both')
        return df


def get_july_21_uol_data(raw_or_interp = 'raw'):
    col_names = pd.read_csv("COSMOS and Hanger Field Data July 2021.csv",nrows=0,usecols=[i for i in range(30,65)]).columns
    df = pd.read_csv("COSMOS and Hanger Field Data July 2021.csv", header=0, usecols=[i for i in range(30, 65)],
                     names=col_names, skiprows=4)
    print(df)


if __name__ == '__main__':

    print(get_july21_cosmos_data('interp'))
    #get_july_21_uol_data('interp')