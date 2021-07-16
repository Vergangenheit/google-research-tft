from typing import List
import pandas as pd
from pandas import DataFrame, Series, Timestamp
from datetime import datetime
import datetime as dt
import numpy as np
import pytz
from etl.ETL import db_connection, group_hourly

from constants import columns, sorgenia_farms, preds_query_interim, query_energy
# from inference.mape import rolling_mape
from utils import shift_lags

class GetData:
    def __init__(self):
        # let's start by define date intervals
        startDate: datetime = (datetime.now(pytz.timezone('UTC')) - dt.timedelta(hours=167))
        endDate: datetime = datetime.now(pytz.timezone('UTC'))

        targetDates: List[datetime] = [endDate + dt.timedelta(hours=i) for i in range(1, 13)]

        # convert into strings and push to the start of the hour
        self.start: str = startDate.strftime("%Y-%m-%d %H:00:00")
        self.end: str = endDate.strftime("%Y-%m-%d %H:00:00")
        self.targets: List[str] = [date.strftime("%Y-%m-%d %H:00:00") for date in targetDates]

        self.farm_list: List = ['UP_PRCLCDPLRM_1',
                                'UP_PRCLCDMZRD_1',
                                'UP_PRCLCDPRZZ_1',
                                'UP_PRCLCMINEO_1',
                                'UP_PEPIZZA_1',
                                'UP_MPNTLCSMBC_1',
                                'UP_MPNTLCDMRN_1']

        self.columns = columns

    def observed_inputs1(self):
        """
        table: "meteomatics_weather" (for start to end - 1)
        :return: None
        """
        query1: str = "SELECT * FROM meteomatics_weather WHERE timestamp_utc between '{}' and '{}'"
        self.observed_df1: DataFrame = pd.read_sql_query(query1.format(self.start, self.end), con=db_connection())
        self.observed_df1.drop(['id'], axis=1, inplace=True)
        self.observed_df1.rename(columns={'timestamp_utc': 'time'}, inplace=True)

    def observed_inputs2(self):
        """
        table: "meteomatics_forecast_weather" (for end)
        :return: None
        """
        query2: str = "SELECT * FROM meteomatics_forecast_weather WHERE forecast_timestamp_utc = '{}' and timestamp_query_utc = '{}'"
        self.observed_df2: DataFrame = pd.read_sql_query(query2.format(self.end, self.end), con=db_connection())
        self.observed_df2.drop(['id', 'timestamp_query_utc'], axis=1, inplace=True)
        self.observed_df2.rename(columns={'forecast_timestamp_utc': 'time'}, inplace=True)

    def concat_observed(self):
        observed_df: DataFrame = pd.concat([self.observed_df1, self.observed_df2], axis=0, ignore_index=True)
        observed_df: DataFrame = observed_df.sort_values(by=['time', 'plant_code'], ascending=True,
                                                         ignore_index=True)
        self.observed_df: DataFrame = observed_df[observed_df['plant_code'].isin(self.farm_list)]
        # self.observed_df = observed_df[columns]

    def known_inputs(self):
        """table: "meteomatics_forecast_weather"
        """
        query_fore: str = "SELECT * FROM meteomatics_forecast_weather WHERE forecast_timestamp_utc between '{}' and '{}'"
        known_df: DataFrame = pd.read_sql_query(query_fore.format(self.targets[0], self.targets[-1]),
                                                con=db_connection())
        known_df.drop(['id'], axis=1, inplace=True)
        known_df: DataFrame = known_df.sort_values(by=['forecast_timestamp_utc', 'plant_code'], ascending=True,
                                                   ignore_index=True)
        # pay attention to the following two lines (given the nature of forecasts acquisition,
        # we need to take the latest update among the various copies
        known_df['diff'] = known_df['forecast_timestamp_utc'] - known_df['timestamp_query_utc']
        known_df = known_df.sort_values('diff', ascending=True).drop_duplicates(
            subset=['plant_code', 'forecast_timestamp_utc'], keep='first')
        known_df.drop(['timestamp_query_utc'], axis=1, inplace=True)
        known_df: DataFrame = known_df[known_df['plant_code'].isin(self.farm_list)]
        known_df.rename(columns={'forecast_timestamp_utc': 'time'}, inplace=True)
        known_df['kwh'] = np.nan
        self.known_df = known_df[columns]

    def extract_targets(self):
        query_tar: str = "SELECT * FROM sorgenia_energy WHERE start_date_utc >= '{}' and end_date_utc < '{}'"
        past_targets: DataFrame = pd.read_sql_query(query_tar.format(self.start, self.end), con=db_connection())
        past_targets: DataFrame = group_hourly(past_targets)
        past_targets: DataFrame = past_targets[past_targets['plant_name_up'].isin(self.farm_list)]
        self.observed_df = self.observed_df.merge(past_targets, how='left', left_on=['plant_code', 'time'],
                                                  right_on=['plant_name_up', 'time'])
        self.observed_df['kwh'] = self.observed_df['kwh'].fillna(method='ffill')
        self.observed_df.drop(['plant_name_up'], axis=1, inplace=True)
        self.observed_df = self.observed_df[columns]

    def generate(self) -> DataFrame:
        self.observed_inputs1()
        self.observed_inputs2()
        self.concat_observed()
        self.extract_targets()
        self.known_inputs()
        df: DataFrame = pd.concat([self.observed_df, self.known_df], axis=0, ignore_index=True)
        df = df.sort_values(['plant_code', 'time'], ascending=True, ignore_index=True)

        # add other engineered features
        timestamp_s: Series = df['time'].map(datetime.timestamp)

        day: int = 24 * 60 * 60
        year: float = 365.2425 * day

        df['Day sin']: Series = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos']: Series = np.cos(timestamp_s * (2 * np.pi / day))
        df['Year sin']: Series = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos']: Series = np.cos(timestamp_s * (2 * np.pi / year))

        earliest_time: Timestamp = df.time.min()
        df['hours_from_start']: Series = (df['time'] - earliest_time).dt.seconds / 60 / 60 + (
                    df['time'] - earliest_time).dt.days * 24
        df["id"] = df["plant_code"]
        df['hour']: Series = df["time"].dt.hour
        df['day_of_week']: Series = df["time"].dt.dayofweek
        df['categorical_id']: Series = df['id'].copy()
        df['kwh'].fillna(method='ffill', inplace=True)

        return df


class GetDataMape:
    def __init__(self, last_months: int, preds_table: str, truth_table: str, preds_query: str, query_energy: str):
        self.n = last_months
        self.preds_table = preds_table
        self.truth_table = truth_table
        self.farm_list = sorgenia_farms
        self.preds_query = preds_query
        self.query_energy = query_energy

    def extract_preds(self) -> DataFrame:
        preds: DataFrame = pd.read_sql_query(self.preds_query.format(self.preds_table, self.n - 1), con=db_connection())
        # gather min and max dates to match with energy query
        self.lower = preds['forecast_time_utc'].min().strftime("%Y-%m-%d %H:%M:%S")
        self.upper = (preds['forecast_time_utc'].max() + pd.Timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S")
        preds.sort_values(by=['identifier', 'forecast_time_utc'], ascending=True, ignore_index=True, inplace=True)

        return preds

    def extract_truths(self) -> DataFrame:
        df_energy: DataFrame = pd.read_sql_query(self.query_energy.format(self.truth_table, self.lower, self.upper),
                                                 con=db_connection())
        df_energy: DataFrame = group_hourly(df_energy)
        df_energy: DataFrame = df_energy[df_energy['plant_name_up'].isin(self.farm_list)]
        lower_en = df_energy['time'].min().strftime("%Y-%m-%d %H:%M:%S")
        upper_en = df_energy['time'].max().strftime("%Y-%m-%d %H:%M:%S")
        assert lower_en == self.lower
        assert upper_en == self.upper
        # rearrange df_energy into lagged schema
        df_energy: DataFrame = df_energy.groupby(by='plant_name_up').apply(shift_lags, 12, 'kwh')
        df_energy.index = df_energy.index.droplevel(0)
        df_energy.drop(['kwh'], axis=1, inplace=True)
        # make sure they're sorted for mape calculation
        df_energy.sort_values(by=['plant_name_up', 'time'], ascending=True, ignore_index=True, inplace=True)

        return df_energy

    def generate(self) -> (DataFrame, DataFrame):
        preds: DataFrame = self.extract_preds()
        truths: DataFrame = self.extract_truths()

        return preds, truths


# if __name__ == "__main__":
#     getdata = GetDataMape(last_months=3, preds_table='tft_testset_preds', truth_table='sorgenia_energy', preds_query=preds_query_interim, query_energy=query_energy)
#     preds, truths = getdata.generate()
#     df_mape: DataFrame = rolling_mape(truths, preds, 700, 'forecast_time_utc', 'plant_name_up')
#     print(df_mape.iloc[:, 2:].mean().mean())


