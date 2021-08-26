import pandas as pd
from july21_get_midas_data import get_july21_midas_data
from july21_get_flux_data import get_july21_flux_data

def get_july21_all_data():
    midas_df = get_july21_midas_data('interp')
    flux_df = get_july21_flux_data('interp')
    all_df = pd.concat([midas_df,flux_df],axis = 1)
    return all_df

if __name__== '__main__':
    df = get_july21_all_data()
    print(df)