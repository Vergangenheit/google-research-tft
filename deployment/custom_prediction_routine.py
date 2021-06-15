from pandas import DataFrame
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.compat.v1 import Session, ConfigProto
from tensorflow.python.eager.context import PhysicalDevice
from typing import Dict, List, Tuple, TypeVar, Type
import os
import time
from data_formatters.base import GenericDataFormatter, InputTypes, DataTypes
from data_formatters.sorgenia_wind import SorgeniaFormatter
from expt_settings.configs import ExperimentConfig
from libs.hyperparam_opt import HyperparamOptManager
from libs.tft_model import TemporalFusionTransformer
import libs.utils as utils

_T = TypeVar("_T")


class MyPredictor(object):
    def __init__(self, formatter: GenericDataFormatter, params: Dict, config: ExperimentConfig):
        """

        :param formatter: Formatter for our problem
        :param params: dictionary of parameters to instantiate the model
        :param config: other useful parameters and states stored in an ad-hoc class
        """
        self.formatter = formatter
        self.params = params
        self.config = config
        print('Done.')

    def predict(self, instances: DataFrame, **kwargs: Dict) -> List[Tuple]:
        """Performs custom prediction.

        Sets up tensorflow graph and session, loads pretrained model and performs predictions

        :param instances: a DataFrame of prediction inputs
        :param kwargs: A dictionary of keyword args provided as additional
                fields on the predict request body.
        :return:
            A list of outputs containing the prediction results.
        """
        t0: float = time.perf_counter()
        if tf.test.gpu_device_name():
            print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")
        gpu: List[PhysicalDevice] = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)
        tf_config: ConfigProto = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)
        tf1.reset_default_graph()
        with tf.Graph().as_default(), tf1.Session(config=tf_config) as sess:
            tf1.keras.backend.set_session(sess)
            self.model = TemporalFusionTransformer(self.params)
            self.params.pop('exp_name', None)
            self.params.pop('data_folder', None)
            self.model.load(os.path.join(self.config.model_folder, "fixed"), use_keras_loadings=False)
            output_map: Dict = self.model.predict(instances, return_targets=False)
            # Extract predictions for each quantile into different entries
            preds: DataFrame = self.formatter.format_predictions(output_map.get("p50"))
        # convert output to a list if that's required by GCP
        t1: float = time.perf_counter()
        print("Time elapsed ", t1 - t0)

        return [tuple(x) for x in preds.to_numpy()]

    @classmethod
    def from_path(cls: Type[_T], model_dir: str) -> _T:
        """Creates an instance of MyPredictor using the given path.
        This loads artifacts that have been copied from your model directory in
        Cloud Storage. MyPredictor uses them during prediction.

         :params : folder with model checkpoints and params
         :return : an instance of the class
        """
        config = ExperimentConfig('sorgenia_wind', model_dir)
        formatter: GenericDataFormatter = config.make_data_formatter()
        print("Formatter data folder is ", formatter.data_folder)
        # Sets up default params
        fixed_params: Dict = formatter.get_experiment_params()
        params: Dict = formatter.get_default_model_params()
        params["model_folder"]: str = os.path.join(config.model_folder, "fixed")
        model_folder = os.path.join(config.model_folder, "fixed")
        # Sets up hyperparam manager
        print("*** Loading hyperparm manager ***")
        opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                           fixed_params, model_folder)
        params: Dict = opt_manager.get_next_parameters()
        params['exp_name'] = 'sorgenia_wind'
        params['data_folder'] = config.data_csv_path

        # load scalers
        formatter.load_scalers()

        return cls(formatter, params, config)

