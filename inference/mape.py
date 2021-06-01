import pandas as pd
from pandas import DataFrame, Timestamp, Series
from typing import Union, List, Dict
import numpy as np
from numpy import ndarray


def rolling_mape_multitarget(targets_df: DataFrame, preds_df: DataFrame, hours_mape: int) -> DataFrame:
    """calculates mape for multiple targets and returns a single df of rolling mapes
    :param : targets_df : DataFrame (dataframe of ground truths
    :param : preds_df : DataFrame (df of of predictions)
    :return : DataFrame of rolling mape values for every target"""
    count = 0
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


def process_mape_by_hz(targets_df: DataFrame, preds_df: DataFrame, hours_mape: int, hz: str) -> DataFrame:
    df_mape: DataFrame = pd.DataFrame(
        data={'forecast_time': preds_df['forecast_time'], 'identifier': preds_df['identifier'], 'true': targets_df[hz],
              'preds': preds_df[hz]})
    df_mape['abs(Pred-true)']: Series = np.abs(df_mape['preds'] - df_mape['true'])
    # loop over up
    count_up = 0
    for up in targets_df.identifier.unique():
        if count_up == 0:
            df_mape_up = df_mape[df_mape['identifier'] == up]
            d: List = []
            for i in range(0, df_mape_up.shape[0] - hours_mape):
                a: int = sum(df_mape_up['abs(Pred-true)'][i:i + hours_mape])
                b: int = sum(df_mape_up['true'][i:i + hours_mape])
                c: float = 100 * a / b
                d.append(c)
                # prendere la data del inizio di intervallo
            p: List = []
            for i in range(0, df_mape_up.shape[0] - hours_mape):
                f: Union[str, Timestamp] = df_mape_up['forecast_time'].iloc[i]
                p.append(f)
            #         assert len(p) == len(d)

            df_mape_final: DataFrame = DataFrame(data={'time': p, 'identifier': up, f'mape_{hz}': d})
            count_up += 1
        else:
            df_mape_up = df_mape[df_mape['identifier'] == up]
            d: List = []
            for i in range(0, df_mape_up.shape[0] - hours_mape):
                a: int = sum(df_mape_up['abs(Pred-true)'][i:i + hours_mape])
                b: int = sum(df_mape_up['true'][i:i + hours_mape])
                c: float = 100 * a / b
                d.append(c)
                # prendere la data del inizio di intervallo
            p: List = []
            for i in range(0, df_mape_up.shape[0] - hours_mape):
                f: Union[str, Timestamp] = df_mape_up['forecast_time'].iloc[i]
                p.append(f)
            #         assert len(p) == len(d)

            df_mape_final_up: DataFrame = DataFrame(data={'time': p, 'identifier': up, f'mape_{hz}': d})
            # stack
            df_mape_final = pd.concat([df_mape_final, df_mape_final_up], axis=0, ignore_index=True)
            count_up += 1

    return df_mape_final


def rolling_mape(targets_df: DataFrame, preds_df: DataFrame, hours_mape: int) -> DataFrame:
    count = 0
    for mt in targets_df.columns[2:]:
        if count == 0:
            df_mape_final = process_mape_by_hz(targets_df, preds_df, hours_mape, mt)
            count += 1
        else:
            df_mape_interim = process_mape_by_hz(targets_df, preds_df, hours_mape, mt)
            df_mape_final[f'mape_{mt}'] = df_mape_interim[f'mape_{mt}']
            count += 1

    return df_mape_final
