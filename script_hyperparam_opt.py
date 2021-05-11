import argparse
import datetime as dte
import os

import data_formatters.base
from data_formatters.electricity import ElectricityFormatter
from data_formatters.favorita import FavoritaFormatter
from data_formatters.traffic import TrafficFormatter
from data_formatters.volatility import VolatilityFormatter
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from typing import Union

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer


def main(expt_name: str, use_gpu: bool, restart_opt, model_folder: str, hyperparam_iterations: int,
         data_csv_path: str,
         data_formatter: Union[ElectricityFormatter, FavoritaFormatter, TrafficFormatter, VolatilityFormatter]):
    """Runs main hyperparameter optimization routine.
  Args:
    expt_name: Name of experiment
    use_gpu: Whether to run tensorflow with GPU operations
    restart_opt: Whether to run hyperparameter optimization from scratch
    model_folder: Folder path where models are serialized
    hyperparam_iterations: Number of iterations of random search
    data_csv_path: Path to csv file containing data
    data_formatter: Dataset-specific data fromatter (see
      expt_settings.dataformatter.GenericDataFormatter)
  """

    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError(
            "Data formatters should inherit from" +
            "AbstractDataFormatter! Type={}".format(type(data_formatter)))

    default_keras_session = tf1.keras.backend.get_session()

    if use_gpu:
        tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

    else:
        tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

    print("### Running hyperparameter optimization for {} ###".format(expt_name))
    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration(
    )

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    param_ranges = ModelClass.get_hyperparm_choices()
    fixed_params["model_folder"] = model_folder

    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager(param_ranges, fixed_params, model_folder)

    success = opt_manager.load_results()
    if success and not restart_opt:
        print("Loaded results from previous training")
    else:
        print("Creating new hyperparameter optimisation")
        opt_manager.clear()

    print("*** Running calibration ***")
    while len(opt_manager.results.columns) < hyperparam_iterations:
        print("# Running hyperparam optimisation {} of {} for {}".format(
            len(opt_manager.results.columns) + 1, hyperparam_iterations, "TFT"))

        tf1.reset_default_graph()
        with tf.Graph().as_default(), tf1.Session(config=tf_config) as sess:

            tf1.keras.backend.set_session(sess)

            params = opt_manager.get_next_parameters()
            model = ModelClass(params, use_cudnn=use_gpu)

            if not model.training_data_cached():
                model.cache_batched_data(train, "train", num_samples=train_samples)
                model.cache_batched_data(valid, "valid", num_samples=valid_samples)

            sess.run(tf1.global_variables_initializer())
            model.fit()

            val_loss = model.evaluate()

            if np.allclose(val_loss, 0.) or np.isnan(val_loss):
                # Set all invalid losses to infintiy.
                # N.b. val_loss only becomes 0. when the weights are nan.
                print("Skipping bad configuration....")
                val_loss = np.inf

            opt_manager.update_score(params, val_loss, model)

            tf1.keras.backend.set_session(default_keras_session)

    print("*** Running tests ***")
    tf1.reset_default_graph()
    with tf.Graph().as_default(), tf1.Session(config=tf_config) as sess:
        tf1.keras.backend.set_session(sess)
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)

        model.load(opt_manager.hyperparam_folder)

        print("Computing best validation loss")
        val_loss = model.evaluate(valid)

        print("Computing test loss")
        output_map = model.predict(test, return_targets=True)
        targets = data_formatter.format_predictions(output_map["targets"])
        p50_forecast = data_formatter.format_predictions(output_map["p50"])
        p90_forecast = data_formatter.format_predictions(output_map["p90"])

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

    print("Hyperparam optimisation completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, " = ", best_params[k])
    print()
    print("Normalised Quantile Loss for Test Data: P50={}, P90={}".format(
        p50_loss.mean(), p90_loss.mean()))
