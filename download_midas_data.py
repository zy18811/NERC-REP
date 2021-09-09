"""
Downloads temperature (soil and air), rain, solar irradation, and wind data from the CEDA website for given years

NOTE: a CEDA account with access to the full MIDAS dataset is required
"""

from ceda_download import download_file
import os
from tqdm import tqdm


def download_soil_temp(year,save_folder, u, p):
    """
    Downloads soil temperature data for year
    :param year: year
    :param save_folder: folder to save to
    :param u: CEDA username
    :param p: CEDA password
    :return: None
    """
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas/data/ST/yearly_files/midas_soiltemp_{year}01-{year}12.txt"
    download_file(url, save_folder, u, p)


def download_air_temp(year,save_folder, u, p):
    """
    Downloads air temperature data for year
    :param year: year
    :param save_folder: folder to save to
    :param u: CEDA username
    :param p: CEDA password
    :return: None
    """
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas/data/TD/yearly_files/midas_tempdrnl_{year}01-{year}12.txt"
    download_file(url,save_folder, u, p)


def download_daily_rain_full(year, save_folder, u, p):
    """
    Downloads daily rainfall data for year
    :param year: year
    :param save_folder: folder to save to
    :param u: CEDA username
    :param p: CEDA password
    :return: None
    """
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas/data/RD/yearly_files/midas_raindrnl_{year}01-{year}12.txt"
    download_file(url, save_folder, u, p)


def download_hourly_rain_full(year, save_folder, u, p):
    """
    Downloads hourly rainfall data for year
    :param year: year
    :param save_folder: folder to save to
    :param u: CEDA username
    :param p: CEDA password
    :return: None
    """
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas/data/RH/yearly_files/midas_rainhrly_{year}01-{year}12.txt"
    download_file(url, save_folder, u, p)


def download_solar_full(year, save_folder, u, p):
    """
    Downloads solar irradiation data for year
    :param year: year
    :param save_folder: folder to save to
    :param u: CEDA username
    :param p: CEDA password
    :return: None
    """
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas/data/RO/yearly_files/midas_radtob_{year}01-{year}12.txt"
    download_file(url,save_folder, u, p)


def download_wind_full(year, save_folder, u, p):
    """
    Downloads win data for year
    :param year: year
    :param save_folder: folder to save to
    :param u: CEDA username
    :param p: CEDA password
    :return: None
    """
    url = f"https://dap.ceda.ac.uk/badc/ukmo-midas/data/WM/yearly_files/midas_wind_{year}01-{year}12.txt"
    download_file(url, save_folder, u, p)


def download_midas_full(year_list,save_folder, soil_temp = True, air_temp = True, daily_rain = True,
                        hourly_rain = True, solar = True, wind = True):
    """
    Downloads MIDAS data for years in year_list
    :param year_list: list of years to download data for
    :param save_folder: folder to save data to
    :param soil_temp: Bool for downloading soil temperature data
    :param air_temp: Bool for downloading air temperature data
    :param daily_rain: Bool for downloading daily rainfall data
    :param hourly_rain: Bool for downloading hourly rainfall data
    :param solar: Bool for downloading solar data
    :param wind: Bool for downloading wind data
    :return: None
    """

    # makes folder paths
    soil_temp_folder = os.path.join(save_folder,'MIDAS Temperature Data/Soil')
    air_temp_folder = os.path.join(save_folder,'MIDAS Temperature Data/Air')
    daily_rain_folder = os.path.join(save_folder,'MIDAS Rain Data/Daily')
    hourly_rain_folder = os.path.join(save_folder,'MIDAS Rain Data/Hourly')
    solar_folder = os.path.join(save_folder, 'MIDAS Solar Data')
    wind_folder = os.path.join(save_folder, 'MIDAS Wind Data')

    def make_folder(arg_bool,folder_path):
        """
        Creates folder_path if arg_bool is true
        """
        if arg_bool:
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

    # makes folders
    make_folder(soil_temp, soil_temp_folder)
    make_folder(air_temp, air_temp_folder)
    make_folder(daily_rain, daily_rain_folder)
    make_folder(hourly_rain, hourly_rain_folder)
    make_folder(solar, solar_folder)
    make_folder(wind, wind_folder)

    # gets CEDA username and password
    print("Enter CEDA username")
    username = input()
    print("Enter CEDA password")
    password = input()

    # downloads data for each year in list
    for year in tqdm(year_list):
        if soil_temp:
            download_soil_temp(year, soil_temp_folder, username, password)
        if air_temp:
            download_air_temp(year, air_temp_folder, username, password)
        if daily_rain:
            download_daily_rain_full(year, daily_rain_folder, username, password)
        if hourly_rain:
            download_hourly_rain_full(year, hourly_rain_folder, username, password)
        if solar:
            download_solar_full(year, solar_folder, username, password)
        if wind:
            download_wind_full(year, wind_folder, username, password)


if __name__ == '__main__':
    """
    Tests download works
    """
    download_midas_full([2020],'Test Folder')





