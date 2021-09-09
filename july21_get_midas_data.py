"""
Constructs a dataframe containing MIDAS data for July 21
"""

import datetime

import pandas as pd


def get_july21_temp_data():
    """
    Gets MIDAS temperature data for July 21
    :return: dataframe
    """
    # reads the soil temperature data for 2021-22
    soil_df = pd.read_csv(
        "Hanger Field July 2021/MIDAS Data/MIDAS Temperature Data/Soil/midas_soiltemp_202101-202112.txt",
        header=0, low_memory=False, names=['id',
                                           'id_type',
                                           'ob_time',
                                           'met_domain_name',
                                           'version_num',
                                           'src_id',
                                           'rec_st_ind',
                                           'q5cm_soil_temp',
                                           'q10cm_soil_temp',
                                           'q20cm_soil_temp',
                                           'q30cm_soil_temp',
                                           'q50cm_soil_temp',
                                           'q100cm_soil_temp',
                                           'q5cm_soil_temp_q',
                                           'prcp_amt_q',
                                           'q20cm_soil_temp_q',
                                           'q30cm_soil_temp_q',
                                           'q50cm_soil_temp_q',
                                           'q100cm_soil_temp_q',
                                           'meto_stmp_time',
                                           'midas_stmp_etime',
                                           'q5cm_soil_temp_j',
                                           'prcp_amt_j',
                                           'q20cm_soil_temp_j',
                                           'q30cm_soil_temp_j',
                                           'q50cm_soil_temp_j',
                                           'q100cm_soil_temp_j'])

    soil_df = soil_df[pd.to_datetime(soil_df['ob_time']).dt.month == 7]  # selects July
    soil_df = soil_df[soil_df.version_num == 1]  # selects quality controlled data
    soil_df = soil_df[soil_df.src_id == 534][
        ['ob_time', 'q10cm_soil_temp']]  # selects time and 10cm soil temp for Spen Farm

    """
    resamples data at a 30min frequency and interpolates to fill gaps
    """
    soil_df['ob_time'] = pd.to_datetime(soil_df['ob_time'])
    soil_df = soil_df.set_index('ob_time')
    soil_df = soil_df.resample('30min').asfreq()
    ix = pd.date_range(datetime.datetime(2021, 7, 1, 0, 30, 0, 0), datetime.datetime(2021, 7, 31, 23, 30, 0, 0),
                       freq='30min')
    soil_df = soil_df.reindex(ix)
    soil_df['q10cm_soil_temp'] = pd.to_numeric(soil_df['q10cm_soil_temp'])
    soil_df = soil_df.rename(columns={"q10cm_soil_temp": "q10cm_soil_temp_raw"})
    soil_df['q10cm_soil_temp_interp'] = soil_df['q10cm_soil_temp_raw'].interpolate(method='time',
                                                                                   limit_direction='both')

    """
    same as above but for air temperature file
    """
    air_df = pd.read_csv(
        "Hanger Field July 2021/MIDAS Data/MIDAS Temperature Data/Air/midas_tempdrnl_202101-202112.txt", header=0,
        low_memory=False, names=['ob_end_time',
                                 'id_type',
                                 'id',
                                 'ob_hour_count',
                                 'version_num',
                                 'met_domain_name',
                                 'src_id',
                                 'rec_st_ind',
                                 'max_air_temp',
                                 'min_air_temp',
                                 'min_grss_temp',
                                 'min_conc_temp',
                                 'max_air_temp_q',
                                 'min_air_temp_q',
                                 'min_grss_temp_q',
                                 'min_conc_temp_q',
                                 'meto_stmp_time',
                                 'midas_stmp_etime',
                                 'max_air_temp_j',
                                 'min_air_temp_j',
                                 'min_grss_temp_j',
                                 'min_conc_temp_j', ])

    air_df = air_df[pd.to_datetime(air_df['ob_end_time']).dt.month == 7]
    air_df = air_df[air_df.version_num == 1]
    air_df = air_df[air_df.src_id == 534][
        ['ob_end_time', 'max_air_temp', 'min_air_temp', 'min_grss_temp', 'min_conc_temp']]

    air_df['ob_end_time'] = pd.to_datetime(air_df['ob_end_time'])
    air_df = air_df.set_index('ob_end_time')
    air_df = air_df.resample('30min').asfreq()
    ix = pd.date_range(datetime.datetime(2021, 7, 1, 0, 30, 0, 0), datetime.datetime(2021, 7, 31, 23, 30, 0, 0),
                       freq='30min')
    air_df = air_df.reindex(ix)
    air_numeric_cols = ['max_air_temp', 'min_air_temp', 'min_grss_temp', 'min_conc_temp']
    air_df[air_numeric_cols] = air_df[air_numeric_cols].apply(pd.to_numeric, errors='coerce')

    air_df[['max_air_temp_interp', 'min_air_temp_interp', 'min_grss_temp_interp', 'min_conc_temp_interp']] = air_df[
        air_numeric_cols].apply(pd.Series.interpolate, method='time', limit_direction='both')
    air_df = air_df.rename(columns={'max_air_temp': 'max_air_temp_raw', 'min_air_temp': 'min_air_temp_raw',
                                    'min_grss_temp': 'min_grss_temp_raw', 'min_conc_temp': 'min_conc_temp_raw'})

    # joins soil and air temperature dataframes
    temp_df = pd.concat([soil_df, air_df], axis=1)

    return temp_df


def get_july21_rain_data():
    """
    Gets MIDAS precipitation data for July 21
    :return: dataframe
    """
    # reads the hourly rainfall data for 2021-22
    df = pd.read_csv("Hanger Field July 2021/MIDAS Data/MIDAS Rain Data/Hourly/midas_rainhrly_202101-202112.txt",
                     header=0, low_memory=False,
                     index_col=False, names=['ob_end_time',
                                             'id',
                                             'id_type',
                                             'ob_hour_count',
                                             'version_num',
                                             'met_domain_name',
                                             'src_id',
                                             'rec_st_ind',
                                             'prcp_amt',
                                             'prcp_dur',
                                             'prcp_amt_q',
                                             'prcp_dur_q',
                                             'meto_stmp_time',
                                             'midas_stmp_etime',
                                             'prcp_amt_j'])

    df = df[pd.to_datetime(df['ob_end_time']).dt.month == 7]  # selects July
    df = df[df.src_id == 534]  # selects Spen Farm
    df = df[df.version_num == 1]
    df = df[df.ob_hour_count == 1][['ob_end_time', 'prcp_amt']]  # selects time and rainfall

    """
    resamples data at a 30min frequency and interpolates to fill gaps
    """
    df['ob_end_time'] = pd.to_datetime(df['ob_end_time'])
    df = df.set_index('ob_end_time')
    df = df.resample('30min').asfreq()
    ix = pd.date_range(datetime.datetime(2021, 7, 1, 0, 30, 0, 0), datetime.datetime(2021, 7, 31, 23, 30, 0, 0),
                       freq='30min')
    df = df.reindex(ix)
    df['prcp_amt'] = pd.to_numeric(df['prcp_amt'])
    df = df.rename(columns={"prcp_amt": "prcp_amt_raw"})
    df['prcp_amt_interp'] = df['prcp_amt_raw'].interpolate(method='time', limit_direction='both')

    return df


def get_july21_sol_data():
    """
    Gets MIDAS solar irradiation data for July 21
    :return: dataframe
    """
    # reads the solar irradiation data for 2021-22
    df = pd.read_csv("Hanger Field July 2021/MIDAS Data/MIDAS Solar Data/midas_radtob_202101-202112.txt", header=0,
                     low_memory=False, names=['id',
                                              'id_type',
                                              'ob_end_time',
                                              'ob_hour_count',
                                              'version_num',
                                              'met_domain_name',
                                              'src_id',
                                              'rec_st_ind',
                                              'glbl_irad_amt',
                                              'difu_irad_amt',
                                              'glbl_irad_amt_q',
                                              'difu_irad_amt_q',
                                              'meto_stmp_time',
                                              'midas_stmp_etime',
                                              'direct_irad',
                                              'irad_bal_amt',
                                              'glbl_s_lat_irad_amt',
                                              'glbl_horz,ilmn',
                                              'direct_irad_q',
                                              'irad_bal_amt_q',
                                              'glbl_s_lat_irad_amt_q',
                                              'glbl_horz,ilmn_q'])

    df = df[pd.to_datetime(df['ob_end_time']).dt.month == 7]  # selects July
    df = df[df.src_id == 534]  # selects Spen Farm
    df = df[df.version_num == 1]
    df = df[df.ob_hour_count == 1][['ob_end_time', 'glbl_irad_amt']]  # selects time and radiation amount

    """
    resamples data at a 30min frequency and interpolates to fill gaps
    """
    df['ob_end_time'] = pd.to_datetime(df['ob_end_time'])
    df = df.set_index('ob_end_time')
    df = df.resample('30min').asfreq()
    ix = pd.date_range(datetime.datetime(2021, 7, 1, 0, 30, 0, 0), datetime.datetime(2021, 7, 31, 23, 30, 0, 0),
                       freq='30min')
    df = df.reindex(ix)
    df['glbl_irad_amt'] = pd.to_numeric(df['glbl_irad_amt'])
    df = df.rename(columns={"glbl_irad_amt": "glbl_irad_amt_raw"})
    df['glbl_irad_amt_interp'] = df['glbl_irad_amt_raw'].interpolate(method='time', limit_direction='both')

    return df


def get_july21_wind_data():
    """
    Gets MIDAS wind data for July 21
    :return: dataframe
    """
    # reads the wind data for 2021-22
    df = pd.read_csv("Hanger Field July 2021/MIDAS Data/MIDAS Wind Data/midas_wind_202101-202112.txt", header=0,
                     low_memory=False, names=['ob_end_time',
                                              'id_type',
                                              'id',
                                              'ob_hour_count',
                                              'met_domain_name',
                                              'version_num',
                                              'src_id',
                                              'rec_st_ind',
                                              'mean_wind_dir',
                                              'mean_wind_speed',
                                              'max_gust_dir',
                                              'max_gust_speed',
                                              'max_gust_ctime',
                                              'mean_wind_dir_q',
                                              'mean_wind_speed_q',
                                              'max_gust_dir_q',
                                              'max_gust_speed_q',
                                              'max_gust_ctime_q',
                                              'meto_stmp_time',
                                              'midas_stmp_etime',
                                              'mean_wind_dir_j',
                                              'mean_wind_speed_j',
                                              'max_gust_dir_j',
                                              'max_gust_speed_j'])
    df = df[pd.to_datetime(df['ob_end_time']).dt.month == 7]  # selects July
    df = df[df.src_id == 534]  # selects Spen Farm
    df = df[df.version_num == 1]
    df = df[df.ob_hour_count == 1][  # selects time, mean direction, mean speed, max speed direction, and max speed
        ['ob_end_time', 'mean_wind_dir', 'mean_wind_speed', 'max_gust_dir', 'max_gust_speed']]

    """
    resamples data at a 30min frequency and interpolates to fill gaps
    """
    df['ob_end_time'] = pd.to_datetime(df['ob_end_time'])
    df = df.set_index('ob_end_time')
    df = df.resample('30min').asfreq()
    ix = pd.date_range(datetime.datetime(2021, 7, 1, 0, 30, 0, 0), datetime.datetime(2021, 7, 31, 23, 30, 0, 0),
                       freq='30min')
    df = df.reindex(ix)

    numeric_cols = ['mean_wind_dir', 'mean_wind_speed', 'max_gust_dir', 'max_gust_speed']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    df[['mean_wind_dir_interp', 'mean_wind_speed_interp', 'max_gust_dir_interp', 'max_gust_speed_interp']] = df[
        numeric_cols].apply(pd.Series.interpolate, method='time', limit_direction='both')
    df = df.rename(columns={'mean_wind_dir': 'mean_wind_dir_raw', 'mean_wind_speed': 'mean_wind_speed_raw',
                            'max_gust_dir': 'max_gust_dir_raw', 'max_gust_speed': 'max_gust_speed_raw'})
    return df


def get_july21_midas_data(interp_or_raw='raw'):
    """
    Gets MIDAS data for July 2021
    :param interp_or_raw: raw_or_interp = 'raw' returns a dataframe with the raw data
                          raw_or_interp = 'interp' returns a dataframe with the data interpolated to 30min intervals
    :return: dataframe
    """

    # gets temperature, rainfall, solar irradiation, and wind dataframes
    temp_df = get_july21_temp_data()
    rain_df = get_july21_rain_data()
    sol_df = get_july21_sol_data()
    wind_df = get_july21_wind_data()

    # joins data frames
    midas_df = pd.concat([temp_df, rain_df, sol_df, wind_df], axis=1)

    # if raw selected, ignore interpolated columns
    if interp_or_raw == 'raw':
        midas_df = midas_df[midas_df.columns.drop(list(midas_df.filter(regex='interp')))]
        midas_df = midas_df.rename(columns=lambda x: x.replace('_raw', ''))
    # if interp selected, ignore raw columns
    elif interp_or_raw == 'interp':
        midas_df = midas_df[midas_df.columns.drop(list(midas_df.filter(regex='raw')))]
        midas_df = midas_df.rename(columns=lambda x: x.replace('_interp', ''))

    midas_df[midas_df.columns] = midas_df[midas_df.columns].apply(pd.to_numeric)

    return midas_df


if __name__ == '__main__':
    """
    prints dataframe
    """
    midas0721 = get_july21_midas_data()
    print(midas0721)
