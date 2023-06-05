"""
Data from https://www.visualcrossing.com/weather/weather-data-services
"""

import pandas as pd
import pylab as plt

forecast_path = '/media/bigdata/projects/tempstick_forecast/data/forecast/02453_2023-05-10_to_2023-06-05_historical.csv'
sensor_path = '/media/bigdata/projects/tempstick_forecast/data/sensor/118653--2023-06-05--three_months.csv'

def return_dat():

    forecast_dat = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
    forecast_wanted_cols = ['datetime','temp','humidity']
    forecast_dat = forecast_dat[forecast_wanted_cols]
    forecast_dat['datetime'] = pd.to_datetime(forecast_dat['datetime'])
    forecast_dat = forecast_dat.set_index('datetime')

    sensor_dat = pd.read_csv(sensor_path, index_col=0, parse_dates=True, skiprows=1)
    sensor_dat = sensor_dat.rename(columns={'Temperature': 'temp', 'Humidity': 'humidity'})
    sensor_dat = sensor_dat[['temp', 'humidity']]

    dat_dict = {'sensor_dat': sensor_dat, 'forecast_dat': forecast_dat}
    return dat_dict

