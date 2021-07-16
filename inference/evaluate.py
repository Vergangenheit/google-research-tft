import pandas as pd
from pandas import DataFrame
import os
from typing import Dict
from inference.mape import rolling_mape
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from libs.hyperparam_opt import HyperparamOptManager
from tensorflow.compat.v1 import Session, ConfigProto
import tensorflow.compat.v1 as tf1
import tensorflow as tf
from data_formatters.sorgenia_wind import SorgeniaFormatter
from libs.tft_model import TemporalFusionTransformer
from expt_settings.configs import ExperimentConfig
import libs.utils as utils


def compute_predictions(test: DataFrame, opt_manager: HyperparamOptManager, formatter: SorgeniaFormatter,
                        config: ExperimentConfig, tf_config: ConfigProto, default_keras_session: Session, exp_name: str):
    """function to compute predictions on testset (might work with another df with same structure)
    :param: test (input data)
    :param: opt_manager (params container object)
    :param: formatter (formatter object)
    :param: config (configuration object)
    :param: tf_config (tf graph Config object)
    :param: default_keras_session
    :param: exp_name (str) sorgenia_wind or sorgenia_wind_no_forecasts"""
    print("*** Running tests ***")
    tf1.reset_default_graph()
    with tf.Graph().as_default(), tf1.Session(config=tf_config) as sess:
        tf1.keras.backend.set_session(sess)
        params: Dict = opt_manager.get_next_parameters()
        params['exp_name'] = exp_name
        params['data_folder'] = os.path.abspath(os.path.join(config.data_csv_path, os.pardir))
        model = TemporalFusionTransformer(params, use_cudnn=False)
        params.pop('exp_name', None)
        params.pop('data_folder', None)
        # load model
        model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

        #     print("Computing best validation loss")
        #     val_loss: Series = model.evaluate(valid)

        print("Computing test loss")
        output_map: Dict = model.predict(test, return_targets=True)
        print(f"Output map returned a dict with keys {output_map.get('p50').shape}")
        targets: DataFrame = formatter.format_predictions(output_map["targets"])
        p50_forecast: DataFrame = formatter.format_predictions(output_map["p50"])
        p90_forecast: DataFrame = formatter.format_predictions(output_map["p90"])

        # save all
        print("saving predictions and targets")
        targets.to_csv(os.path.join(opt_manager.hyperparam_folder, "targets.csv"), index=False)
        p50_forecast.to_csv(os.path.join(opt_manager.hyperparam_folder, "p50.csv"), index=False)
        p90_forecast.to_csv(os.path.join(opt_manager.hyperparam_folder, "p90.csv"), index=False)

        def extract_numerical_data(data: DataFrame) -> DataFrame:
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

    print()
    print("Normalised Quantile Loss for Test Data: P50={}, P90={}".format(
        p50_loss.mean(), p90_loss.mean()))


def evaluate(predictions_path: str, model: str, window: int) -> DataFrame:
    """evaluate model performance using saved predictions made on testset
    :param : predictions_path (str) where you have the predictions and model training's output saved
    :param : model (str) if you want to tag the model that originated those saved predictions with a name
    :param : window (int) rolling mape window
    :return : df_mape (DataFrame)"""
    p50_forecast: DataFrame = pd.read_csv(os.path.join(predictions_path, "p50.csv"))
    # p90_forecast: DataFrame = pd.read_csv(os.path.join(predictions_path, "p90.csv"))
    targets: DataFrame = pd.read_csv(os.path.join(predictions_path, "targets.csv"))
    targets['forecast_time'] = targets['forecast_time'].astype('datetime64[s]')
    p50_forecast['forecast_time'] = p50_forecast['forecast_time'].astype('datetime64[s]')
    # p90_forecast['forecast_time'] = p90_forecast['forecast_time'].astype('datetime64[s]')
    # calculate rolling mape on p50
    df_mape: DataFrame = rolling_mape(targets, p50_forecast, window)
    print(f"{model} ", df_mape.iloc[:, 2:].mean().mean())

    return df_mape


def boxplotter(df: DataFrame) -> Figure:
    """plots boxplot of rolling mape across prediction horizons
    :param : df : (DataFrame) dataframe with mape per horizon in the columns (assumes time and identifier as first two columns)"""
    fig: Figure = go.Figure()
    # loop over horizons
    for hz in df.columns[2:]:
        horizon: int = int(hz.split('t')[-1])
        fig.add_trace(go.Box(y=df[hz], name=f'{str(horizon)}h'))

    fig.update_layout(width=1000, height=500, title="Mape Comparison across prediction horizons",
                      title_x=0.5, xaxis_title="Horizons", yaxis_title=f'rolling mape(%)',
                      legend_title="Models")

    return fig
