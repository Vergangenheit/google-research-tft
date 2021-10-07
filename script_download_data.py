import numpy as np
import pandas as pd
from expt_settings.configs import ExperimentConfig
import os
import wget
import pyunpack
from pandas import DataFrame, Series, Timestamp, Index
from numpy import ndarray
from sqlalchemy.engine import Engine, Connection
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from os import getenv
from typing import Optional, List
from datetime import datetime
from etl.ETL import db_connection, group_hourly, extract_weather, etl_plant, etl_weather


# General functions for data downloading & aggregation.
def download_from_url(url: str, output_path: str):
    """Downloads a file froma url."""

    print('Pulling data from {} to {}'.format(url, output_path))
    wget.download(url, output_path)
    print('done')


def unzip(zip_path: str, output_file: str, data_folder: str):
    """Unzips files and checks successful completion."""

    print('Unzipping file: {}'.format(zip_path))
    pyunpack.Archive(zip_path).extractall(data_folder)

    # Checks if unzip was successful
    if not os.path.exists(output_file):
        raise ValueError(
            'Error in unzipping process! {} not found.'.format(output_file))


def download_and_unzip(url: str, zip_path: str, csv_path: str, data_folder: str):
    """Downloads and unzips an online csv file.
  Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data is stored.
  """

    download_from_url(url, zip_path)

    unzip(zip_path, csv_path, data_folder)

    print('Done.')


def download_electricity(config: ExperimentConfig) -> str:
    """Downloads electricity dataset from UCI repository."""

    url: str = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'

    data_folder: str = config.data_folder
    csv_path: str = os.path.join(data_folder, url.split('/')[-1].replace('.zip', ''))
    zip_path: str = csv_path + '.zip'
    if os.path.exists(csv_path):
        return csv_path
    else:
        download_and_unzip(url, zip_path, csv_path, data_folder)

        return csv_path


def preprocess_electricty(csv_path: str, config: ExperimentConfig):
    if os.path.exists(config.data_csv_path):
        print(f'File already preprocessed in {config.data_csv_path}')
    else:
        print('Aggregating to hourly data')

        df: DataFrame = pd.read_csv(csv_path, index_col=0, sep=';', decimal=',')
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Used to determine the start and end dates of a series
        output = df.resample('1h').mean().replace(0., np.nan)

        earliest_time: Timestamp = output.index.min()

        df_list = []
        for label in output:
            print('Processing {}'.format(label))
            srs: Series = output[label]

            start_date: Timestamp = min(srs.fillna(method='ffill').dropna().index)
            end_date: Timestamp = max(srs.fillna(method='bfill').dropna().index)

            active_range: ndarray = (srs.index >= start_date) & (srs.index <= end_date)
            srs: Series = srs[active_range].fillna(0.)

            tmp: DataFrame = pd.DataFrame({'power_usage': srs})
            date: Series = tmp.index
            tmp['t']: Series = (date - earliest_time).seconds / 60 / 60 + (
                    date - earliest_time).days * 24
            tmp['days_from_start']: Series = (date - earliest_time).days
            tmp['categorical_id']: Series = label
            tmp['date']: Series = date
            tmp['id']: Series = label
            tmp['hour']: Series = date.hour
            tmp['day']: Series = date.day
            tmp['day_of_week']: Series = date.dayofweek
            tmp['month']: Series = date.month

            df_list.append(tmp)

        output: DataFrame = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

        output['categorical_id']: Series = output['id'].copy()
        output['hours_from_start']: Series = output['t']
        output['categorical_day_of_week']: Series = output['day_of_week'].copy()
        output['categorical_hour']: Series = output['hour'].copy()

        # Filter to match range used by other academic papers
        output: DataFrame = output[(output['days_from_start'] >= 1096)
                                   & (output['days_from_start'] < 1346)].copy()

        output.to_csv(config.data_csv_path)
        print(f'Saved in {config.data_csv_path}')
        print('Done.')


def preprocess_erg_wind(datapath: str, config: ExperimentConfig):
    erg_raw: DataFrame = pd.read_csv(os.path.join(datapath, 'erg\erg_7farms_stacked.csv'))
    erg_raw['time'] = erg_raw['time'].astype('datetime64[s]')
    earliest_time: Timestamp = erg_raw.time.min()
    erg_raw['t']: Series = (erg_raw['time'] - earliest_time).dt.seconds / 60 / 60 + (
            erg_raw['time'] - earliest_time).dt.days * 24
    erg_raw['days_from_start']: Series = (erg_raw['time'] - earliest_time).dt.days
    erg_raw["id"] = erg_raw["UP"]
    erg_raw['hour']: Series = erg_raw["time"].dt.hour
    erg_raw['day']: Series = erg_raw["time"].dt.day
    erg_raw['day_of_week']: Series = erg_raw["time"].dt.dayofweek
    erg_raw['month']: Series = erg_raw["time"].dt.month
    erg_raw['categorical_id']: Series = erg_raw['id'].copy()
    erg_raw['hours_from_start']: Series = erg_raw['t']
    erg_raw['categorical_day_of_week']: Series = erg_raw['day_of_week'].copy()
    erg_raw['categorical_hour']: Series = erg_raw['hour'].copy()
    erg_raw.to_csv(config.data_csv_path, index=False)
    print(f'Saved in {config.data_csv_path}')
    print('Done.')


def merge_df(energy: DataFrame, weather: DataFrame, col_drop: List) -> DataFrame:
    df: DataFrame = energy.merge(weather, left_on=['time', 'plant_name_up'],
                                 right_on=['timestamp_utc', 'plant_name_up'])
    df.drop(col_drop, axis=1, inplace=True)
    df = df.sort_values(by=['plant_name_up', 'time'], ascending=True, ignore_index=True)

    return df


def preprocess_sorgenia_full(config: ExperimentConfig, get_df: bool = False) -> Optional[DataFrame]:
    """end to end etl function from database to df
    :param : weather_source (str) whether to use meteomatics or copernicus
    :param : config (ExperimentConfig) class for experiment configuration
    :param : get_df (bool) whether to return df as pandas.DataFrame or not (useful for inference demo)
    : return : df (Optional[DataFrame])"""
    engine: Engine = db_connection()
    sql_energy: str = "SELECT * FROM sorgenia_energy"
    energy_df: DataFrame = pd.read_sql_query(sql_energy, con=engine)
    energy_grouped: DataFrame = group_hourly(energy_df, 'start_date_utc', 'plant_name_up')
    # extract weather mm historical
    mm_query: str = "SELECT * FROM sorgenia_weather"
    weather_df: DataFrame = extract_weather(mm_query, engine)
    # INFER THE DATES GAP BETWEEN Energy and Weather dfs
    upper: str = weather_df['timestamp_utc'].min().strftime('%Y-%m-%d %H:%M:%S')
    lower: str = energy_grouped['time'].min().strftime('%Y-%m-%d %H:%M:%S')
    cop_sql: str = f"SELECT * FROM sorgenia_weather_copernicus WHERE timestamp_utc >= '{lower}' and timestamp_utc < '{upper}'"
    weather_remain: DataFrame = extract_weather(cop_sql, engine)
    #  STACK WEATHER df together
    weather: DataFrame = pd.concat([weather_df, weather_remain], axis=0)
    weather.sort_values(by='timestamp_utc', ascending=True, inplace=True)
    # INFER THE DATES GAP BETWEEN Energy and Weather dfs
    lower: str = weather['timestamp_utc'].max().strftime('%Y-%m-%d %H:%M:%S')
    upper: str = energy_grouped['time'].max().strftime('%Y-%m-%d %H:%M:%S')
    # extract weather meteomatics "recent"
    mm_sql: str = f"SELECT * FROM meteomatics_weather WHERE timestamp_utc >= '{lower}' and timestamp_utc < '{upper}'"
    weather_remain2: DataFrame = extract_weather(mm_sql, engine)
    weather_remain2.rename(columns={'plant_code': 'plant_name_up'}, inplace=True)
    weather_remain2 = weather_remain2[list(weather.columns)]
    weather_remain2.drop(['id'], axis=1, inplace=True)
    weather.drop(['id'], axis=1, inplace=True)
    weather: DataFrame = pd.concat([weather, weather_remain2], axis=0)
    weather.sort_values(by='timestamp_utc', ascending=True, inplace=True)
    df: DataFrame = merge_df(energy_grouped, weather, ['timestamp_utc'])
    print(df.tail())
    timestamp_s: Series = df['time'].map(datetime.timestamp)

    day: int = 24 * 60 * 60
    year: float = 365.2425 * day

    df['Day sin']: Series = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos']: Series = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin']: Series = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos']: Series = np.cos(timestamp_s * (2 * np.pi / year))

    earliest_time: Timestamp = df.time.min()
    df['t']: Series = (df['time'] - earliest_time).dt.seconds / 60 / 60 + (df['time'] - earliest_time).dt.days * 24
    df['days_from_start']: Series = (df['time'] - earliest_time).dt.days
    df["id"] = df["plant_name_up"]
    df['hour']: Series = df["time"].dt.hour
    df['day']: Series = df["time"].dt.day
    df['day_of_week']: Series = df["time"].dt.dayofweek
    df['month']: Series = df["time"].dt.month
    df['categorical_id']: Series = df['id'].copy()
    df['hours_from_start']: Series = df['t']
    df['categorical_day_of_week']: Series = df['day_of_week'].copy()
    df['categorical_hour']: Series = df['hour'].copy()

    # save df to csv file
    # df = df[df['plant_name_up'] == 'UP_MPNTLCDMRN_1']
    df.to_csv(config.data_csv_path, index=False)
    print(f'Saved in {config.data_csv_path}')
    print('Done.')
    if get_df:
        return df


def preprocess_sorgenia_cop_mm(config: ExperimentConfig, get_df: bool = False) -> Optional[DataFrame]:
    """end to end etl function from database to df
        :param : config (ExperimentConfig) class for experiment configuration
        :param : get_df (bool) whether to return df as pandas.DataFrame or not (useful for inference demo)
        : return : df (Optional[DataFrame])
    """
    engine: Engine = db_connection()
    sql_energy: str = "SELECT * FROM sorgenia_energy"
    energy_df: DataFrame = pd.read_sql_query(sql_energy, con=engine)
    energy_grouped: DataFrame = group_hourly(energy_df, 'start_date_utc', 'plant_name_up')
    # extract weather copernicus
    mm_query: str = "SELECT * FROM sorgenia_weather"
    weather_df: DataFrame = extract_weather(mm_query, engine)
    # INFER THE DATES GAP BETWEEN Energy and Weather dfs
    upper: str = weather_df['timestamp_utc'].min().strftime('%Y-%m-%d %H:%M:%S')
    lower: str = energy_grouped['time'].min().strftime('%Y-%m-%d %H:%M:%S')
    cop_sql: str = f"SELECT * FROM sorgenia_weather_copernicus WHERE timestamp_utc >= '{lower}' and timestamp_utc < '{upper}'"
    weather_remain: DataFrame = extract_weather(cop_sql, engine)
    #  STACK WEATHER df together
    weather: DataFrame = pd.concat([weather_df, weather_remain], axis=0)
    weather.sort_values(by='timestamp_utc', ascending=True, inplace=True)
    df: DataFrame = merge_df(energy_grouped, weather, ['timestamp_utc', 'id'])

    timestamp_s: Series = df['time'].map(datetime.timestamp)

    day: int = 24 * 60 * 60
    year: float = 365.2425 * day

    df['Day sin']: Series = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos']: Series = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin']: Series = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos']: Series = np.cos(timestamp_s * (2 * np.pi / year))

    earliest_time: Timestamp = df.time.min()
    df['t']: Series = (df['time'] - earliest_time).dt.seconds / 60 / 60 + (df['time'] - earliest_time).dt.days * 24
    df['days_from_start']: Series = (df['time'] - earliest_time).dt.days
    df["id"] = df["plant_name_up"]
    df['hour']: Series = df["time"].dt.hour
    df['day']: Series = df["time"].dt.day
    df['day_of_week']: Series = df["time"].dt.dayofweek
    df['month']: Series = df["time"].dt.month
    df['categorical_id']: Series = df['id'].copy()
    df['hours_from_start']: Series = df['t']
    df['categorical_day_of_week']: Series = df['day_of_week'].copy()
    df['categorical_hour']: Series = df['hour'].copy()

    # save df to csv file
    # df = df[df['plant_name_up'] == 'UP_MPNTLCDMRN_1']
    df.to_csv(config.data_csv_path, index=False)
    print(f'Saved in {config.data_csv_path}')
    print('Done.')
    if get_df:
        return df


def preprocess_sotavento(config: ExperimentConfig, get_df: bool = False) -> Optional[DataFrame]:
    engine: Engine = db_connection()
    energy_df = etl_plant("SELECT * FROM energy_sotavento", engine)
    weather_df: DataFrame = etl_weather(engine)
    weather_df.drop(['lat', 'long'], axis=1, inplace=True)
    # merge on time column
    df: DataFrame = energy_df.merge(weather_df, left_on=['date'], right_on=['time'])
    df.drop(['date', 'cut_in', 'cut_out'], axis=1, inplace=True)
    df.sort_values(by='time', ascending=True, ignore_index=True, inplace=True)

    earliest_time: Timestamp = df.time.min()
    df['t']: Series = (df['time'] - earliest_time).dt.seconds / 60 / 60 + (df['time'] - earliest_time).dt.days * 24
    df['days_from_start']: Series = (df['time'] - earliest_time).dt.days
    df["id"] = "sotavento"
    df['hour']: Series = df["time"].dt.hour
    df['day']: Series = df["time"].dt.day
    df['day_of_week']: Series = df["time"].dt.dayofweek
    df['month']: Series = df["time"].dt.month
    df['categorical_id']: Series = df['id'].copy()
    df['hours_from_start']: Series = df['t']
    df['categorical_day_of_week']: Series = df['day_of_week'].copy()
    df['categorical_hour']: Series = df['hour'].copy()

    df.to_csv(config.data_csv_path, index=False)
    print(f'Saved in {config.data_csv_path}')
    print('Done.')
    if get_df:
        return df


if __name__ == "__main__":
    # expt_config = ExperimentConfig('electricity', './outputs/data/electricity')
    # csv_path: str = download_electricity(expt_config)
    # preprocess_electricty(csv_path, expt_config)
    # expt_config = ExperimentConfig('erg_wind', './outputs/data/erg_wind')
    # preprocess_sorgenia(r'C:\Users\Lorenzo\PycharmProjects\TFT\outputs\data', expt_config)
    expt_config = ExperimentConfig('sorgenia_wind', './outputs/data/sorgenia_wind')
    preprocess_sorgenia_full(expt_config)
