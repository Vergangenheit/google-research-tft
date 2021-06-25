import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Timestamp
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.compat.v1 import Session, ConfigProto
from tensorflow.python.eager.context import PhysicalDevice
from typing import Dict, List, Union, Generator
import os
from numpy import load
from data_formatters.base import GenericDataFormatter, InputTypes, DataTypes
from data_formatters.sorgenia_wind import SorgeniaFormatter
from data_formatters.erg_wind import ErgFormatter
from expt_settings.configs import ExperimentConfig
from libs.hyperparam_opt import HyperparamOptManager
from libs.tft_model import TemporalFusionTransformer
import libs.utils as utils
import pickle


def extract_attention():
    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    gpu: List[PhysicalDevice] = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
    # Tensorflow setup
    default_keras_session: Session = tf1.keras.backend.get_session()
    tf_config: ConfigProto = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

    # raw_data: DataFrame = pd.read_csv(file_path)
    # raw_data['time'] = raw_data['time'].astype('datetime64[s]')
    config = ExperimentConfig('sorgenia_wind', 'outputs')
    formatter: SorgeniaFormatter = config.make_data_formatter()
    data_csv_path: str = config.data_csv_path
    raw_data: DataFrame = pd.read_csv(data_csv_path)
    raw_data['time'] = raw_data['time'].astype('datetime64[s]')
    train, valid, test = formatter.split_data(raw_data)
    # Sets up default params
    fixed_params: Dict = formatter.get_experiment_params()
    params: Dict = formatter.get_default_model_params()
    params["model_folder"]: str = os.path.join(config.model_folder, "fixed")
    model_folder = os.path.join(config.model_folder, "fixed")
    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)
    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)
    print("*** Extracting attention weights ***")
    tf1.reset_default_graph()
    with tf.Graph().as_default(), tf1.Session(config=tf_config) as sess:
        tf1.keras.backend.set_session(sess)
        params: Dict = opt_manager.get_next_parameters()
        params['exp_name'] = 'sorgenia_wind'
        params['data_folder'] = os.path.abspath(os.path.join(data_csv_path, os.pardir))
        model = TemporalFusionTransformer(params, use_cudnn=False)
        params.pop('exp_name', None)
        params.pop('data_folder', None)
        # load model
        model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)
        att_weights: Dict = model.get_attention(test)

        with open(os.path.join(config.model_folder, "fixed", "attn_weights.pkl"), 'wb') as f:
            pickle.dump(att_weights, f, protocol=4)


if __name__ == "__main__":
    extract_attention()
