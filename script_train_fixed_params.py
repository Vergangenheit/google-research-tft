"""Trains TFT based on a defined set of parameters.
Uses default parameters supplied from the configs file to train a TFT model from
scratch.
Usage:
python3 script_train_fixed_params {expt_name} {output_folder}
Command line args:
  expt_name: Name of dataset/experiment to train.
  output_folder: Root folder in which experiment is saved
"""

import argparse
from argparse import ArgumentParser, Namespace
import datetime as dte
import os
from data_formatters.base import GenericDataFormatter, InputTypes, DataTypes
from data_formatters.electricity import ElectricityFormatter
from data_formatters.favorita import FavoritaFormatter
from data_formatters.traffic import TrafficFormatter
from data_formatters.volatility import VolatilityFormatter
from expt_settings.configs import ExperimentConfig
from libs.hyperparam_opt import HyperparamOptManager
from libs.tft_model import TemporalFusionTransformer
import libs.utils as utils
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.compat.v1 import Session, ConfigProto
from tensorflow.python.eager.context import PhysicalDevice
from typing import Dict, List, Union

ModelClass = TemporalFusionTransformer


def main(expt_name: str,
         use_gpu: bool,
         model_folder: str,
         data_csv_path: str,
         data_formatter: Union[ElectricityFormatter, FavoritaFormatter, TrafficFormatter, VolatilityFormatter],
         use_testing_mode: bool = False):
    """Trains tft based on defined model params.
  Args:
    expt_name: Name of experiment
    use_gpu: Whether to run tensorflow with GPU operations
    model_folder: Folder path where models are serialized
    data_csv_path: Path to csv file containing data
    data_formatter: Dataset-specific data fromatter (see
      expt_settings.dataformatter.GenericDataFormatter)
    use_testing_mode: Uses a smaller models and data sizes for testing purposes
      only -- switch to False to use original default settings
  """
    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    # gpu: List[PhysicalDevice] = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpu[0], True)
    num_repeats = 1

    if not isinstance(data_formatter, GenericDataFormatter):
        raise ValueError(
            "Data formatters should inherit from" +
            "AbstractDataFormatter! Type={}".format(type(data_formatter)))

    # Tensorflow setup
    default_keras_session: Session = tf1.keras.backend.get_session()
    if tf.test.is_gpu_available():
        gpu: List[PhysicalDevice] = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)
    # if use_gpu:
        tf_config: ConfigProto = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

    else:
        tf_config: ConfigProto = utils.get_default_tensorflow_config(tf_device="cpu")

    print("*** Training from defined parameters for {} ***".format(expt_name))

    print("Loading & splitting data...")
    print(data_csv_path)
    if expt_name != 'erg_wind':
        raw_data: DataFrame = pd.read_csv(data_csv_path, index_col=0)
    else:
        raw_data: DataFrame = pd.read_csv(data_csv_path)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration(
    )

    # Sets up default params
    fixed_params: Dict = data_formatter.get_experiment_params()
    params: Dict = data_formatter.get_default_model_params()
    params["model_folder"]: str = model_folder

    # Parameter overrides for testing only! Small sizes used to speed up script.
    if use_testing_mode:
        fixed_params["num_epochs"] = 1
        params["hidden_layer_size"] = 5
        train_samples, valid_samples = 100, 10

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)

    # Training -- one iteration only
    print("*** Running calibration ***")
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    best_loss = np.Inf
    for _ in range(num_repeats):

        tf1.reset_default_graph()
        with tf.Graph().as_default(), tf1.Session(config=tf_config) as sess:

            tf1.keras.backend.set_session(sess)

            params: Dict = opt_manager.get_next_parameters()
            params['exp_name'] = expt_name
            params['data_folder'] = os.path.abspath(os.path.join(data_csv_path, os.pardir))
            model: TemporalFusionTransformer = ModelClass(params, use_cudnn=False)
            params.pop('data_folder', None)
            params.pop('exp_name', None)
            if not os.path.exists(os.path.join(model.data_folder, 'data.npy')) and not model.training_data_cached():
                model.cache_batched_data(train, "train", num_samples=train_samples)
            if not os.path.exists(os.path.join(model.data_folder, 'val_data.npy')):
                model.cache_batched_data(valid, "valid", num_samples=valid_samples)

            sess.run(tf1.global_variables_initializer())
            model.fit()

            val_loss: Series = model.evaluate()

            if val_loss < best_loss:
                opt_manager.update_score(params, val_loss, model)
                best_loss = val_loss

            tf1.keras.backend.set_session(default_keras_session)

    print("*** Running tests ***")
    tf1.reset_default_graph()
    with tf.Graph().as_default(), tf1.Session(config=tf_config) as sess:
        tf1.keras.backend.set_session(sess)
        best_params: Dict = opt_manager.get_best_params()
        best_params['exp_name'] = expt_name
        model = ModelClass(best_params, use_cudnn=use_gpu)
        best_params.pop('exp_name', None)
        model.load(opt_manager.hyperparam_folder)

        print("Computing best validation loss")
        val_loss: Series = model.evaluate(valid)

        print("Computing test loss")
        output_map: Dict = model.predict(test, return_targets=True)
        targets: DataFrame = data_formatter.format_predictions(output_map["targets"])
        p50_forecast: DataFrame = data_formatter.format_predictions(output_map["p50"])
        p90_forecast: DataFrame = data_formatter.format_predictions(output_map["p90"])

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        p50_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(p50_forecast),
            0.5)
        p90_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(p90_forecast),
            0.9)

        tf1.keras.backend.set_session(default_keras_session)

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, " = ", best_params[k])
    print()
    print("Normalised Quantile Loss for Test Data: P50={}, P90={}".format(
        p50_loss.mean(), p90_loss.mean()))


if __name__ == "__main__":
    def get_args() -> (str, str, bool):
        """Gets settings from command line."""

        experiment_names: List[str] = ExperimentConfig.default_experiments

        parser: ArgumentParser = argparse.ArgumentParser(description="Data download configs")
        parser.add_argument(
            "expt_name",
            metavar="e",
            type=str,
            nargs="?",
            default="electricity",
            choices=experiment_names,
            help="Experiment Name. Default={}".format(",".join(experiment_names))
        )
        parser.add_argument(
            "output_folder",
            metavar="f",
            type=str,
            nargs="?",
            default=".",
            help="Path to folder for data download"
        )
        parser.add_argument(
            "use_gpu",
            metavar="g",
            type=str,
            nargs="?",
            choices=["yes", "no"],
            default="no",
            help="Whether to use gpu for training."
        )

        args: Namespace = parser.parse_known_args()[0]

        root_folder = None if args.output_folder == "." else args.output_folder

        return args.expt_name, root_folder, args.use_gpu == 'yes'


    name, output_folder, use_tensorflow_with_gpu = get_args()

    print("Using output folder {}".format(output_folder))

    config = ExperimentConfig(name, output_folder)
    formatter = config.make_data_formatter()

    # Customise inputs to main() for new datasets.
    main(expt_name=name,
         use_gpu=True,
         model_folder=os.path.join(config.model_folder, "fixed"),
         data_csv_path=config.data_csv_path,
         data_formatter=formatter,
         use_testing_mode=False)  # Change to false to use original default params
