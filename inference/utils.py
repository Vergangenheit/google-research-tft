import pandas as pd
from pandas import DataFrame, Series
from tqdm import tqdm


def test_targets(targets: DataFrame, raw_data: DataFrame) -> None:
    """test targets vs raw_data
    :param : targets (DataFrame)
    :param : raw_data (DataFrame)
    :return : None"""
    for i in tqdm(zip(targets['forecast_time'].values, targets['identifier'].values)):
        assert targets[(targets['forecast_time'] == i[0]) & (targets['identifier'] == i[1])].iloc[:,
               2:].values.tolist().sort() == raw_data[
                                                 (raw_data['date'] >= pd.Timestamp(i[0]) + pd.Timedelta(hours=1)) & (
                                                         raw_data['date'] <= pd.Timestamp(i[0]) + pd.Timedelta(
                                                     hours=24)) & (raw_data['id'] == i[1])].iloc[:,
                                             1].values.tolist().sort()


def pivot(df: DataFrame) -> DataFrame:
    """pivot the model predictions into a format congenial for mape analysis
    :param : df (DataFrame)
    :return : DataFrame"""
    df['forecast_time'] = df['forecast_time'].astype('datetime64[s]')
    df_pivot: DataFrame = df.pivot(index=['forecast_time'], columns='identifier', values=df.columns[2:].tolist())
    df_piv2: DataFrame = df_pivot.stack(level=0)
    # drop multilevel index
    df_piv2: DataFrame = df_piv2.reset_index(level=1, drop=False)
    # set level as a int (hourly timedelta)
    df_piv2['level_1'] = df_piv2['level_1'].str.replace('t+', '')
    df_piv2['level_1'] = df_piv2['level_1'].astype('int64')
    df_piv2['level_1'] = df_piv2['level_1'] + 1
    df_piv2.reset_index(drop=False, inplace=True)

    def add_hour(row: Series) -> Series:
        row['forecast_time'] = row['forecast_time'] + pd.Timedelta(hours=row['level_1'])

        return row

    df_piv2: DataFrame = df_piv2.apply(add_hour, axis=1)
    # sort by forecast_time
    df_piv2.sort_values(by=['forecast_time'], ascending=True, inplace=True)
    df_piv2.drop(['level_1'], axis=1, inplace=True)
    # drop duplicates hours
    df_piv2.drop_duplicates(subset=['forecast_time'], inplace=True)
    df_piv2.columns.name = None
    df_piv2.reset_index(drop=True, inplace=True)

    return df_piv2
