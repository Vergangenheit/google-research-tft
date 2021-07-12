from sqlalchemy.engine import Engine, Connection
from sqlalchemy import create_engine
from os import getenv
from typing import List, Dict, Optional
import pandas as pd
from pandas import DataFrame, Series, Timestamp
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
import numpy as np
from numpy import ndarray
import pytz
from etl.ETL import db_connection


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

    def observed_inputs1(self):
        """
        table: "meteomatics_weather" (for start to end - 1)
        :return: None
        """
        query1: str = "SELECT * FROM meteomatics_weather WHERE timestamp_utc between '{}' and '{}'"
        self.observed_df1: DataFrame = pd.read_sql_query(query1.format(self.start, self.end), con=db_connection())
        self.observed_df1.drop(['id'], axis=1, inplace=True)

    def observed_inputs2(self):
        """
        table: "meteomatics_forecast_weather" (for end)
        :return: None
        """
        query2: str = "SELECT * FROM meteomatics_forecast_weather WHERE forecast_timestamp_utc = '{}' and timestamp_query_utc = '{}'"
        self.observed_df2: DataFrame = pd.read_sql_query(query2.format(self.end, self.end), con=db_connection())
        self.observed_df2.drop(['id', 'timestamp_query_utc'], axis=1, inplace=True)
        self.observed_df2.rename(columns={'forecast_timestamp_utc': 'timestamp_utc'}, inplace=True)

    def concat_observed(self):
        observed_df: DataFrame = pd.concat([self.observed_df1, self.observed_df2], axis=0, ignore_index=True)
        observed_df: DataFrame = observed_df.sort_values(by=['timestamp_utc', 'plant_code'], ascending=True,
                                                         ignore_index=True)
        self.observed_df: DataFrame = observed_df[observed_df['plant_code'].isin(self.farm_list)]

    def known_inputs(self):
        """table: "meteomatics_forecast_weather"
        """
        query_fore: str = "SELECT * FROM meteomatics_forecast_weather WHERE forecast_timestamp_utc between '{}' and '{}'"
        known_df: DataFrame = pd.read_sql_query(query_fore.format(self.targets[0], self.targets[-1]), con=db_connection())
        known_df.drop(['id'], axis=1, inplace=True)
        known_df: DataFrame = known_df.sort_values(by=['forecast_timestamp_utc', 'plant_code'], ascending=True,
                                                   ignore_index=True)
        known_df['diff'] = known_df['forecast_timestamp_utc'] - known_df['timestamp_query_utc']
        known_df = known_df.sort_values('diff', ascending=True).drop_duplicates(
            subset=['plant_code', 'forecast_timestamp_utc'], keep='first')
        if known_df['timestamp_query_utc'].unique() != pd.Timestamp(self.targets[0]):
            raise AssertionError("The expected timestamp of forecasts is wrong")
        known_df.drop(['timestamp_query_utc'], axis=1, inplace=True)
        self.known_df: DataFrame = known_df[known_df['plant_code'].isin(self.farm_list)]




getdata = GetData()
getdata.observed_inputs1()
getdata.observed_inputs2()
getdata.concat_observed()
print(getdata.observed_df)


