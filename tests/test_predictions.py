from pandas import DataFrame
import pandas as pd
import numpy as np
from numpy import allclose, ndarray
import random


def test_targets(targets: DataFrame, raw_data: DataFrame):
    """
    test if predict method returns targets who are equal in content to raw_data
    :param targets: targets DataFrame returned by predict method
    :param raw_data: original dataset
    :param timestamp: (str) random timestamp in dataset
    :param identifier: (str) random identifier
    :return: None
    """
    # pick random timestamp and identifier
    timestamp: str = random.choice(targets['forecast_time'])
    identifier: str = random.choice(targets['identifier'])
    target_data: ndarray = targets[
                               (targets['forecast_time'] == timestamp) & (targets['identifier'] == identifier)].iloc[:,
                           2:].values
    raw: ndarray = raw_data[(raw_data['date'] >= pd.Timestamp(timestamp) + pd.Timedelta(hours=1)) & (
            raw_data['date'] <= pd.Timestamp(timestamp) + pd.Timedelta(
        hours=24)) & (raw_data['id'] == identifier)].iloc[:,
                   1].values

    assert allclose(target_data, raw)
