from data_formatters.base import GenericDataFormatter, DataTypes, InputTypes
import pandas as pd
import sklearn.preprocessing as pp
from typing import Tuple, Dict, List, Optional
from pandas import DataFrame, Series, DatetimeIndex
from libs import utils
from os import getenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, Connection


class SorgeniaFormatter(GenericDataFormatter):
    """Defines and formats data for the sorgenia wind dataset.
          Note that per-entity z-score normalization is used here, and is implemented
          across functions.
          Attributes:
            column_definition: Defines input and data type of column used in the
              experiment.
            identifiers: Entity identifiers used in experiments.
    """
    _column_definition = [
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('energy_mw', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('Wind Speed', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('2m_devpoint [C]', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('temperature [C]', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('mean_sealev_pressure [hPa]', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('surface pressure [hPa]', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('precipitation [m]', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10_wind_speed', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10_u_wind', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10_v_wind', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('instant_wind_gust [m/s]', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Day sin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Day cos', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Year sin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Year cos', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT)
    ]

    def __init__(self):
        self.db_connection()

    def db_connection(self):
        self.dbname: Optional[str] = getenv('POSTGRES_DB_NAME')
        self.host: Optional[str] = getenv('POSTGRES_HOST')
        self.user: Optional[str] = getenv('POSTGRES_USERNAME')
        self.password: Optional[str] = getenv('POSTGRES_PASSWORD')
        self.port: Optional[str] = getenv('POSTGRES_PORT')
        self.energy_table: Optional[str] = getenv('ENERGY_TABLE')
        self.weather_table_cop: Optional[str] = getenv('WEATHER_TABLE_COP')

        self.postgres_str: str = f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}'

        self.engine: Engine = create_engine(self.postgres_str)

    def query_energy(self, sql_energy: str):
        energy_df: DataFrame = pd.read_sql_query(sql_energy, con=self.engine)
        self.energy_df = self.group_hourly(energy_df)

    def group_hourly(self, df: DataFrame) -> DataFrame:
        df: DataFrame = df.copy()
        df['day']: Series = df['start_date_utc'].dt.year.astype('str') + '-' + df['start_date_utc'].dt.month.astype(
            'str') + '-' + df[
                                'start_date_utc'].dt.day.astype('str')
        df['day']: Series = pd.to_datetime(df['day'], infer_datetime_format=True)
        grouped: DataFrame = df.groupby(['plant_name_up', 'day', df.start_date_utc.dt.hour]).agg(
            {'kwh': 'mean'})
        grouped: DataFrame = grouped.reset_index(drop=False).rename(columns={'start_date_utc': 'time'})
        grouped['time'] = grouped['day'].astype('str') + ' ' + grouped['time'].astype('str') + ':00:00'
        grouped['time'] = grouped['time'].astype('datetime64[ns, UTC]')
        grouped: DataFrame = grouped.sort_values(by=['plant_name_up', 'day', 'time'], ascending=True, ignore_index=True)
        grouped.drop('day', axis=1, inplace=True)

        return grouped

    def extract_weather(self, weather_sql: str, engine: Engine):
        weather_df: DataFrame = pd.read_sql_query(weather_sql, con=engine)
        weather_df['wind_gusts_100m_1h_ms'] = weather_df['wind_gusts_100m_1h_ms'].astype('float64')
        weather_df['wind_gusts_100m_ms'] = weather_df['wind_gusts_100m_ms'].astype('float64')
        self.weather_df: DataFrame = weather_df.sort_values(by=['timestamp_utc'], ascending=True, ignore_index=True)

    def merge_df(self) -> DataFrame:
        df: DataFrame = self.energy_df.merge(self.weather_df, left_on=['time', 'plant_name_up'],
                                     right_on=['timestamp_utc', 'plant_name_up'])
        df.drop(['timestamp_utc', 'id'], axis=1, inplace=True)
        df = df.sort_values(by=['plant_name_up', 'time'], ascending=True, ignore_index=True)

        return df

