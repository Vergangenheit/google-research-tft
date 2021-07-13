from inference import GetData
from predictor import MyPredictor

from sqlalchemy.engine import Engine, Connection
from sqlalchemy import create_engine
from os import getenv
import pandas as pd
from pandas import DataFrame, Series, Timestamp
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
import numpy as np
from numpy import ndarray
import pytz

import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow import Graph
from tensorflow.compat.v1 import Session, ConfigProto
from tensorflow.python.eager.context import PhysicalDevice
from typing import Dict, List, Union, Generator, Tuple, TypeVar, Type
import os
from numpy import load, ndarray
import time
from tensorflow.python.keras.engine.functional import Functional

from data_formatters.base import GenericDataFormatter, InputTypes, DataTypes
from data_formatters.sorgenia_wind import SorgeniaFormatter
from expt_settings.configs import ExperimentConfig
from libs.hyperparam_opt import HyperparamOptManager
from libs.tft_model import TemporalFusionTransformer
import libs.utils as utils
import json


def main(model_path: str):
    # extract inference data
    getdata = GetData()
    df: DataFrame = getdata.generate()
    # instantiate predictor
    predictor = MyPredictor.from_path(model_path)
    # testing Predictor on sample
    preds: DataFrame = predictor.predict(df)

    print(preds)


if __name__ == "__main__":
    main(r'C:\Users\Lorenzo\PycharmProjects\TFT\outputs')
