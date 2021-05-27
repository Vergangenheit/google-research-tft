import pandas as pd
from pandas import DataFrame
import os
from inference.mape import rolling_mape
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from libs.hyperparam_opt import HyperparamOptManager
from tensorflow.compat.v1 import Session, ConfigProto
from data_formatters.sorgenia_wind import SorgeniaFormatter

def compute_predictions(test: DataFrame, opt_manager: HyperparamOptManager, formatter: SorgeniaFormatter, tf_config: ConfigProto):
    """function to compute predictions on testset (might work with another df with same structure)"""


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
        horizon: int = int(hz.split('+')[-1]) + 1
        fig.add_trace(go.Box(y=df[hz], name=f'{str(horizon)}h'))

    fig.update_layout(width=1000, height=500, title="Mape Comparison across prediction horizons",
                      title_x=0.5, xaxis_title="Horizons", yaxis_title=f'rolling mape(%)',
                      legend_title="Models")

    return fig
