import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing

def get_july21_temp_data():
    soil_df = pd.read_csv("Full Temperature Data/Full Soil Temp/midas_soiltemp_202101-202112.txt", header=0,
                            low_memory = False, names=['id',
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

    soil_df = soil_df[pd.to_datetime(soil_df['ob_time']).dt.month == 7]
    soil_df = soil_df[soil_df.version_num == 1]
    soil_df = soil_df[soil_df.src_id == 534][['ob_time','q10cm_soil_temp']]

    soil_df['ob_time'] = pd.to_datetime(soil_df['ob_time'])
    soil_df = soil_df.set_index('ob_time')
    soil_df = soil_df.resample('30min').asfreq()
    ix = pd.date_range(datetime.datetime(2021,7,1,0,30,0,0), datetime.datetime(2021,7,31,23,30,0,0), freq='30min')
    soil_df = soil_df.reindex(ix)
    soil_df['q10cm_soil_temp'] = pd.to_numeric(soil_df['q10cm_soil_temp'])
    soil_df = soil_df.rename(columns={"q10cm_soil_temp": "q10cm_soil_temp_raw"})
    soil_df['q10cm_soil_temp_interp'] = soil_df['q10cm_soil_temp_raw'].interpolate(method='time')


    air_df = pd.read_csv("Full Temperature Data/Full Air Temp/midas_tempdrnl_202101-202112.txt",header=0,
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
    air_df = air_df[air_df.src_id == 534][['ob_end_time','max_air_temp','min_air_temp','min_grss_temp','min_conc_temp']]

    air_df['ob_end_time'] = pd.to_datetime(air_df['ob_end_time'])
    air_df = air_df.set_index('ob_end_time')
    air_df = air_df.resample('30min').asfreq()
    ix = pd.date_range(datetime.datetime(2021, 7, 1, 0, 30, 0, 0), datetime.datetime(2021, 7, 31, 23, 30, 0, 0), freq='30min')
    air_df = air_df.reindex(ix)
    air_numeric_cols = ['max_air_temp','min_air_temp','min_grss_temp','min_conc_temp']
    air_df[air_numeric_cols] = air_df[air_numeric_cols].apply(pd.to_numeric,errors = 'coerce')

    temp_df = pd.concat([soil_df,air_df],axis = 1)
    #temp_df = temp_df.reset_index(level=0).rename(columns={'index': 'ob_time'})

    return temp_df


def get_july21_rain_data():
    df = pd.read_csv("Full Rain Data/Full Hourly/midas_rainhrly_202101-202112.txt", header=0, low_memory=False, index_col=False, names=['ob_end_time',
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

    df = df[pd.to_datetime(df['ob_end_time']).dt.month == 7]
    df = df[df.src_id == 534]
    df = df[df.version_num == 1]
    df = df[df.ob_hour_count == 1][['ob_end_time', 'prcp_amt']]
    df['ob_end_time'] = pd.to_datetime(df['ob_end_time'])
    df = df.set_index('ob_end_time')
    df = df.resample('30min').asfreq()
    ix = pd.date_range(datetime.datetime(2021, 7, 1, 0, 30, 0, 0), datetime.datetime(2021, 7, 31, 23, 30, 0, 0),
                       freq='30min')
    df = df.reindex(ix)
    df['prcp_amt'] = pd.to_numeric(df['prcp_amt'])
    df = df.rename(columns={"prcp_amt": "prcp_amt_raw"})
    df['prcp_amt_interp'] = df['prcp_amt_raw'].interpolate(method='time')

    return df

def get_july21_sol_data():
    df = pd.read_csv("Full Solar Data/midas_radtob_202101-202112.txt", header=0, low_memory=False, names=['id',
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

    df = df[pd.to_datetime(df['ob_end_time']).dt.month == 7]
    df = df[df.src_id == 534]
    df = df[df.version_num == 1]
    df = df[df.ob_hour_count == 1][['ob_end_time', 'glbl_irad_amt']]

    df['ob_end_time'] = pd.to_datetime(df['ob_end_time'])
    df = df.set_index('ob_end_time')
    df = df.resample('30min').asfreq()
    ix = pd.date_range(datetime.datetime(2021, 7, 1, 0, 30, 0, 0), datetime.datetime(2021, 7, 31, 23, 30, 0, 0),
                       freq='30min')
    df = df.reindex(ix)
    df['glbl_irad_amt'] = pd.to_numeric(df['glbl_irad_amt'])
    df = df.rename(columns={"glbl_irad_amt": "glbl_irad_amt_raw"})
    df['glbl_irad_amt_interp'] = df['glbl_irad_amt_raw'].interpolate(method='time')
    return df


def get_july21_wind_data():
    df = pd.read_csv("Full Wind Data/midas_wind_202101-202112.txt", header=0, low_memory=False, names=['ob_end_time',
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
    df = df[pd.to_datetime(df['ob_end_time']).dt.month == 7]
    df = df[df.src_id == 534]
    df = df[df.version_num == 1]
    df = df[df.ob_hour_count == 1][['ob_end_time', 'mean_wind_dir','mean_wind_speed','max_gust_dir','max_gust_speed']]
    df['ob_end_time'] = pd.to_datetime(df['ob_end_time'])
    df = df.set_index('ob_end_time')
    df = df.resample('30min').asfreq()
    ix = pd.date_range(datetime.datetime(2021, 7, 1, 0, 30, 0, 0), datetime.datetime(2021, 7, 31, 23, 30, 0, 0),
                       freq='30min')
    df = df.reindex(ix)

    numeric_cols = ['mean_wind_dir','mean_wind_speed','max_gust_dir','max_gust_speed']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    df[['mean_wind_dir_interp','mean_wind_speed_interp','max_gust_dir_interp','max_gust_speed_interp']] = df[numeric_cols].apply(pd.Series.interpolate,method = 'time')
    df = df.rename(columns={'mean_wind_dir': 'mean_wind_dir_raw', 'mean_wind_speed': 'mean_wind_speed_raw',
                            'max_gust_dir': 'max_gust_dir_raw', 'max_gust_speed': 'max_gust_speed_raw'})
    return df

def get_july21_midas_data():
    temp_df = get_july21_temp_data()
    rain_df = get_july21_rain_data()
    sol_df = get_july21_sol_data()
    wind_df = get_july21_wind_data()

    midas_df = pd.concat([temp_df,rain_df,sol_df,wind_df],axis = 1)
    #midas_df = midas_df.reset_index(level=0).rename(columns={'index': 'ob_time'})
    return midas_df

if __name__ == '__main__':
    #print(get_july21_temp_data())
    #print(get_july21_rain_data())
    #print(get_july21_sol_data())
    #print(get_july21_wind_data())
    midas0721 = get_july21_midas_data()

    df = midas0721
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns = df.columns)
    for col in df.columns:

        if not col[-6:] == 'interp':
            plot_data = df[col].dropna()
            plt.plot(plot_data.index.values, plot_data.values, label=col)
        else:
            pass
    plt.legend()
    plt.show()
