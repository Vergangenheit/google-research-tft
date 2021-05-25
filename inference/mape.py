import pandas as pd
from pandas import DataFrame, Timestamp, Series
from typing import Union, List, Dict
import numpy as np


def rolling_mape_multitarget(targets_df: DataFrame, preds_df: DataFrame, hours_mape: int) -> DataFrame:
    """calculates mape for multiple targets and returns a single df of rolling mapes
    :param : targets_df : DataFrame (dataframe of ground truths
    :param : preds_df : DataFrame (df of of predictions)
    :return : DataFrame of rolling mape values for every target"""
    count = 0
    # loop over targets as df columns skipping the first col(time)
    for mt in targets_df.columns[1:]:
        if count == 0:
            df_mape: DataFrame = pd.DataFrame(
                data={'forecast_time': preds_df['forecast_time'], 'true': targets_df[mt], 'preds': preds_df[mt]})
            df_mape['abs(Pred-true)']: Series = np.abs(df_mape['preds'] - df_mape['true'])
            d: List = []

            for i in range(0, df_mape.shape[0] - hours_mape):
                a: int = sum(df_mape['abs(Pred-true)'][i:i + hours_mape])
                b: int = sum(df_mape['true'][i:i + hours_mape])
                c: float = 100 * a / b
                d.append(c)

            # prendere la data del inizio di intervallo
            p: List = []
            for i in range(0, df_mape.shape[0] - hours_mape):
                f: Union[str, Timestamp] = df_mape['forecast_time'].iloc[i]
                p.append(f)
            assert len(p) == len(d)

            df_mape_final: DataFrame = DataFrame(data={'time': p, f'mape_{mt}': d})
            count += 1

        else:
            df_mape: DataFrame = pd.DataFrame(
                data={'forecast_time': preds_df['forecast_time'], 'true': targets_df[mt], 'preds': preds_df[mt]})
            df_mape['abs(Pred-true)']: Series = np.abs(df_mape['preds'] - df_mape['true'])
            d: List = []

            for i in range(0, df_mape.shape[0] - hours_mape):
                a: int = sum(df_mape['abs(Pred-true)'][i:i + hours_mape])
                b: int = sum(df_mape['true'][i:i + hours_mape])
                c: float = 100 * a / b
                d.append(c)

            # prendere la data del inizio di intervallo
            p: List = []
            for i in range(0, df_mape.shape[0] - hours_mape):
                f: Union[str, Timestamp] = df_mape['forecast_time'].iloc[i]
                p.append(f)
            assert len(p) == len(d)

            df_mape_interim: DataFrame = DataFrame(data={'time': p, f'mape_{mt}': d})
            df_mape_final[f'mape_{mt}'] = df_mape_interim[f'mape_{mt}']
            count += 1

    return df_mape_final
