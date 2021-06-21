import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series
from typing import Union


class TsCleaner(object):
    """cleans the timeseries filling gaps in timesteps, missing wind speeds and active powers
    Args: df (pandas DataFrame)"""

    def __init__(self, df: DataFrame):
        self.df = df

    def fill_gaps_turbine(self, date_index: DatetimeIndex) -> DataFrame:
        """takes a dataframe subset by turbine, and fills the ws and ap gaps via timeinterpolation"""
        # df_1: DataFrame = self.df[self.df['Turbine'] == turbine]
        # drop duplicates in time
        self.df.drop_duplicates(subset=['date'], keep='first', inplace=True)
        df_index: DataFrame = pd.DataFrame(data=date_index, columns=['date'])
        # merge df_1 and df_index on time(0)
        self.df: DataFrame = self.df.merge(df_index, how='outer', on='date', suffixes=('_sota', '_ind'))
        self.df.sort_values(by='date', ascending=True, ignore_index=True, inplace=True)

        assert self.df.date.to_list() == date_index.to_list()

        # set time as df index to do the time interpolation
        self.df.set_index('date', inplace=True)
        self.df['speed_ms'].interpolate(method='time', inplace=True)
        self.df['energy_kwh'].interpolate(method='time', inplace=True)
        self.df['direction_deg'].interpolate(method='time', inplace=True)
        # reset index
        self.df: DataFrame = self.df.reset_index()

    def fix_gaps(self) -> DataFrame:
        """established date range and fills self.df one turbine and the time and then reconcatenates into one"""
        # create a date range
        date_index: DatetimeIndex = pd.date_range(start=self.df.date.min(), end=self.df.date.max(),
                                                  freq=pd.offsets.Minute(10))
        self.fill_gaps_turbine(date_index)

    @staticmethod
    def gaps(df: DataFrame, time_col: str) -> None:
        """utils method to check time series contiguity"""
        df: DataFrame = df.sort_values(by=time_col, ascending=True, ignore_index=True)
        df[f'{time_col}+1'] = df[time_col].shift(-1)
        df[f'{time_col}_freq'] = (df[f'{time_col}+1'] - df[time_col]).astype('timedelta64[m]')
        gaps: Union[Series, DataFrame] = df[df[f'{time_col}_freq'] > 1.0]
        print(gaps)
        print(df[f'{time_col}_freq'].unique())
        print(df[df[f'{time_col}_freq'] == df[f'{time_col}_freq'].unique()[1]])
        try:
            print(df[df[f'{time_col}_freq'] == df[f'{time_col}_freq'].unique()[2]])
        except Exception as e:
            print("no other time frequencies")

    @staticmethod
    def find_dupl(df: DataFrame, col: str):
        ids: Series = df[col]
        print(df[ids.isin(ids[ids.duplicated()])].sort_values(col))
