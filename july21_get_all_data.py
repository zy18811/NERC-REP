"""
Constructs a dataframe containing the MIDAS, COSMOS and flux data for July 2021
"""
from july21_get_midas_data import get_july21_midas_data
from july21_get_flux_data import get_july21_flux_data
from july21_get_cosmos_data import get_july21_cosmos_data


def get_july21_all_data(raw_or_interp):
    """
    Gets MIDAS, COSMOS, and flux data for July 2021 and returns a dataframe
    :param raw_or_interp: raw_or_interp = 'raw' returns a dataframe with the raw data
                          raw_or_interp = 'interp' returns a dataframe with the data interpolated to 30min intervals
    :return: dataframe
    """
    # gets midas, flux and cosmos dataframes
    midas_df = get_july21_midas_data(raw_or_interp)
    flux_df = get_july21_flux_data(raw_or_interp)
    cosmos_df = get_july21_cosmos_data(raw_or_interp)

    # merges dataframes into one dataframe
    all_df = midas_df.merge(flux_df, left_index=True, right_index=True)
    all_df = all_df.merge(cosmos_df, left_index=True, right_index=True)

    return all_df


if __name__== '__main__':
    """
    prints dataframe
    """
    df = get_july21_all_data('interp')
    print(df)
