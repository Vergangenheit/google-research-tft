import numpy as np
import pandas as pd
from expt_settings.configs import ExperimentConfig
import os
import wget
import pyunpack
from pandas import DataFrame, Series, Timestamp, Index
from numpy import ndarray


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


if __name__ == "__main__":
    # expt_config = ExperimentConfig('electricity', './outputs/data/electricity')
    # csv_path: str = download_electricity(expt_config)
    # preprocess_electricty(csv_path, expt_config)
    expt_config = ExperimentConfig('erg_wind', './outputs/data/erg_wind')
    preprocess_erg_wind(r'C:\Users\Lorenzo\PycharmProjects\TFT\outputs\data', expt_config)
