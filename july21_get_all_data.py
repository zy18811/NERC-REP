import pandas as pd
from july21_get_midas_data import get_july21_midas_data
from july21_get_flux_data import get_july21_flux_data
from july21_get_cosmos_data import get_july21_cosmos_data

def get_july21_all_data(raw_or_interp):
    midas_df = get_july21_midas_data(raw_or_interp)

    flux_df = get_july21_flux_data(raw_or_interp)

    cosmos_df = get_july21_cosmos_data(raw_or_interp)

    all_df = midas_df.merge(flux_df, left_index=True, right_index=True)
    all_df = all_df.merge(cosmos_df, left_index=True, right_index=True)

    return all_df

if __name__== '__main__':
    df = get_july21_all_data('interp')
    print(df)
    print(df.columns[df.isna().any()].tolist())
    #cosmos_df = get_july21_cosmos_data('interp')
    #print(df.index)
    #print(cosmos_df.index)
    #print(df.index.difference(cosmos_df.index))