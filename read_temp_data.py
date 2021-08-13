from temp_data_download import download_data
import numpy as np
import pandas as pd
import glob
import datetime


def construct_temp_dicts(csv_folder_path):
    all_soil_csvs = glob.iglob(csv_folder_path + "/Soil Temp/*.csv")
    all_air_csvs = glob.iglob(csv_folder_path + "/Air Temp/*.csv")

    soil_df_dict = {}

    air_df_dict = {}

    for soil_csv, air_csv in zip(all_soil_csvs,all_air_csvs):
        soil_df = pd.read_csv(soil_csv,skiprows=85)
        air_df = pd.read_csv(air_csv,skiprows=90)

        year = soil_csv.split('.')[0][-4:]

        soil_df_dict[year] = soil_df
        air_df_dict[year] = air_df

    return soil_df_dict, air_df_dict


def daily_soil_temp(date,soil_df):
    daily_temp_10_cm_mean = soil_df.loc[pd.to_datetime(soil_df['ob_time']).dt.date == date]['q10cm_soil_temp'].mean()
    return daily_temp_10_cm_mean


def daily_air_temp(date,air_df):
    daily_temps_mean = air_df.loc[pd.to_datetime(air_df['ob_end_time']).dt.date == date][['max_air_temp','min_air_temp']].mean()
    max_temp_mean = daily_temps_mean['max_air_temp']
    min_temp_mean = daily_temps_mean['min_air_temp']
    return max_temp_mean,min_temp_mean


def get_temps_4_date(date,csv_folder):
    download_data(range(1959,2020),csv_folder)
    soil_dict, air_dict = construct_temp_dicts(csv_folder)
    try:
        soil_df = soil_dict[str(date.year)]
        soil_df = soil_df[:-1]
        soil_10_cm_mean = daily_soil_temp(date, soil_df)
    except KeyError:
        soil_10_cm_mean = np.NaN

    try:
        air_df = air_dict[str(date.year)]
        air_df = air_df[:-1]
        max_temp_mean, min_temp_mean = daily_air_temp(date, air_df)
    except KeyError:
        max_temp_mean = min_temp_mean = np.NaN

    return soil_10_cm_mean,max_temp_mean,min_temp_mean


if __name__ == '__main__':
    test_date = datetime.date(2017,2,22)
    soil, max_air, min_air = get_temps_4_date(test_date,'Temperature Data')
    degree_sign = u"\N{DEGREE SIGN}"
    if not np.isnan(soil):
        print(f"Mean Soil Temp. = {soil}{degree_sign}C")
    else:
        print("No soil temperature data for this date")
    if not (np.isnan(max_air) and np.isnan(min_air)):
        print(f"Mean Max Air Temp = {max_air}{degree_sign}C, Mean Min Air Temp = {min_air}{degree_sign}C")
    else:
        print("No air temperature data for this date")
