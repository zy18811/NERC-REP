import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing


def get_july21_flux_data(raw_or_interp='raw'):
    df = pd.read_csv("July_flux_data_HF.csv",parse_dates=[['date','time']],dayfirst=True)
    df = df[:-1]
    df = df[['date_time','Tau','H','LE','co2_flux','h2o_flux']]


    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.set_index('date_time')

    ix = pd.date_range(datetime.datetime(2021, 7, 1, 0, 30, 0, 0), datetime.datetime(2021, 7, 31, 23, 30, 0, 0),
                       freq='30min')

    df = df.reindex(ix)
    if raw_or_interp == 'raw':
        return df
    elif raw_or_interp =='interp':
        df = df.apply(pd.Series.interpolate, method='time',limit_direction = 'both')
        return df

if __name__ == '__main__':
    print(get_july21_flux_data('interp').index)