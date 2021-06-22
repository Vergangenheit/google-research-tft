from sqlalchemy.engine import Engine, Connection
from sqlalchemy import create_engine
from os import getenv
from typing import List, Dict, Optional
import pandas as pd
from pandas import DataFrame, Series, Timestamp
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from numpy import ndarray
import math
from etl.ts_cleaner import TsCleaner


def db_connection() -> Engine:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except:
        print('No ".env" file or python-dotenv not installed... Using default env variables...')
    dbname: Optional[str] = getenv('POSTGRES_DB_NAME')
    host: Optional[str] = getenv('POSTGRES_HOST')
    user: Optional[str] = getenv('POSTGRES_USERNAME')
    password: Optional[str] = getenv('POSTGRES_PASSWORD')
    port: Optional[str] = getenv('POSTGRES_PORT')

    postgres_str: str = f'postgresql://{user}:{password}@{host}:{port}/{dbname}'

    engine: Engine = create_engine(postgres_str)

    return engine


def group_hourly(df: DataFrame) -> DataFrame:
    df: DataFrame = df.copy()
    df['day']: Series = df['start_date_utc'].dt.year.astype('str') + '-' + df['start_date_utc'].dt.month.astype(
        'str') + '-' + df[
                            'start_date_utc'].dt.day.astype('str')
    df['day']: Series = pd.to_datetime(df['day'], infer_datetime_format=True)
    grouped: DataFrame = df.groupby(['plant_name_up', 'day', df.start_date_utc.dt.hour]).agg(
        {'kwh': 'mean'})
    grouped: DataFrame = grouped.reset_index(drop=False).rename(columns={'start_date_utc': 'time'})
    #     grouped: DataFrame = grouped.sort_values(by=['plant_name_up', 'day', 'time'], ascending=True, ignore_index=True)
    grouped['time'] = grouped['day'].astype('str') + ' ' + grouped['time'].astype('str') + ':00:00'
    grouped['time'] = grouped['time'].astype('datetime64[ns, UTC]')
    grouped: DataFrame = grouped.sort_values(by=['plant_name_up', 'time'], ascending=True, ignore_index=True)
    grouped.drop('day', axis=1, inplace=True)

    return grouped


def extract_weather(weather_sql: str, engine: Engine) -> DataFrame:
    weather_df: DataFrame = pd.read_sql_query(weather_sql, con=engine)
    weather_df['wind_gusts_100m_1h_ms'] = weather_df['wind_gusts_100m_1h_ms'].astype('float64')
    weather_df['wind_gusts_100m_ms'] = weather_df['wind_gusts_100m_ms'].astype('float64')
    weather_df: DataFrame = weather_df.sort_values(by=['timestamp_utc'], ascending=True, ignore_index=True)

    return weather_df


def overlap(row: Series) -> str:
    if math.isnan(row['speed_ms']) and math.isnan(row['energy_kwh']):
        return 'yes'
    else:
        return 'no'


def clean_row(row: Series) -> float:
    if (row['speed_ms'] < row['cut_in'] or row['speed_ms'] > row['cut_out']) and row['energy_kwh'] != 0:
        return 0.
    elif row['energy_kwh'] < 0 and (row['speed_ms'] > row['cut_in'] or row['speed_ms'] < row['cut_out']):
        return 0.
    else:
        return row['energy_kwh']


def fill_gaps(data: DataFrame) -> DataFrame:
    """fill gaps in energy by interpolating based on speed and direction"""
    data['energy_kwh'] = np.where(data['energy_kwh'] > 80000, np.nan, data['energy_kwh'])
    data.set_index(['speed_ms', 'direction_deg'], inplace=True)
    data.interpolate(method='linear', inplace=True)
    data.reset_index(inplace=True)

    return data


def down_sample(df: DataFrame) -> DataFrame:
    """sub-sampling the data from 10 minute intervals to 1h"""
    df: DataFrame = df[5::6]

    return df


def etl_plant(sql_energy: str, engine: Engine) -> DataFrame:
    data: DataFrame = pd.read_sql_query(sql_energy, con=engine)
    data = data.replace('-', np.nan)
    data['speed_ms'] = data['speed_ms'].astype('float')
    data['direction_deg'] = data['direction_deg'].astype('float')
    data['energy_kwh'] = data['energy_kwh'].astype('float')
    data['date']: Series = data['date'].astype('datetime64[ns]')
    data['nan_overlap'] = data.apply(overlap, axis=1)
    # load power curve table
    td: DataFrame = pd.read_sql_query("SELECT * FROM turbine_data_sotavento", con=engine)
    td['wind_speed_ms'] = td['wind_speed_ms'].astype('float')
    td['Total_power_kW'] = td['Total_power_kW'].astype('float')
    td['Energy_kWh_10min'] = td['Total_power_kW'] / 6
    # add cut-in and cut-out based on power curve
    data['cut_in'] = 3.
    data['cut_out'] = 25.
    x: Series = data.apply(clean_row, axis=1)
    data['energy_kwh'] = x
    # interpolate NaNs
    cleaner: TsCleaner = TsCleaner(data)
    cleaner.fix_gaps()
    data: DataFrame = cleaner.df
    data.drop(['nan_overlap', 'cut_in', 'cut_out'], axis=1, inplace=True)
    # reappend cut-in and cut-out
    data['cut_in'] = 3.
    data['cut_out'] = 25.
    x: Series = data.apply(clean_row, axis=1)
    data['energy_kwh'] = x
    # fill 99999 values
    data: DataFrame = fill_gaps(data)
    # downsample
    data_reduced: DataFrame = down_sample(data)

    return data_reduced


def etl_weather(engine: Engine) -> DataFrame:
    data: DataFrame = pd.read_sql_query("SELECT * FROM pala_spain", con=engine)
    data['time']: Series = data['time'].astype('datetime64[ns]')
    data.sort_values(by='time', ascending=True, ignore_index=True, inplace=True)

    timestamp_s: Series = data['time'].map(datetime.timestamp)
    day: int = 24 * 60 * 60
    year: float = 365.2425 * day

    data['Day sin']: Series = np.sin(timestamp_s * (2 * np.pi / day))
    data['Day cos']: Series = np.cos(timestamp_s * (2 * np.pi / day))
    data['Year sin']: Series = np.sin(timestamp_s * (2 * np.pi / year))
    data['Year cos']: Series = np.cos(timestamp_s * (2 * np.pi / year))

    return data


def extract_mm(engine: Engine) -> DataFrame:
    weather_mm: DataFrame = pd.read_sql_query("SELECT *FROM meteomatics_sotavento", con=engine)
    weather_mm['date'] = weather_mm['date'].astype('datetime64[s]')
    weather_mm.rename(columns={'lon': 'long', 'date': 'time'}, inplace=True)

    timestamp_s: Series = weather_mm['time'].map(datetime.timestamp)
    day: int = 24 * 60 * 60
    year: float = 365.2425 * day

    weather_mm['Day sin']: Series = np.sin(timestamp_s * (2 * np.pi / day))
    weather_mm['Day cos']: Series = np.cos(timestamp_s * (2 * np.pi / day))
    weather_mm['Year sin']: Series = np.sin(timestamp_s * (2 * np.pi / year))
    weather_mm['Year cos']: Series = np.cos(timestamp_s * (2 * np.pi / year))

    return weather_mm
