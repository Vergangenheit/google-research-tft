from data_formatters.base import GenericDataFormatter, DataTypes, InputTypes
import pandas as pd
import sklearn.preprocessing as pp
from typing import Tuple, Dict, List, Optional
from pandas import DataFrame, Series, DatetimeIndex
from libs import utils
import os
import json
import pickle


class SotaventoFormatter(GenericDataFormatter):
    """Defines and formats data for the sorgenia wind dataset.
              Note that per-entity z-score normalization is used here, and is implemented
              across functions.
              Attributes:
                column_definition: Defines input and data type of column used in the
                  experiment.
                identifiers: Entity identifiers used in experiments.
    """
    _column_definition = [
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('speed_ms', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('direction_deg', DataTypes.DATE, InputTypes.OBSERVED_INPUT),
        ('energy_kwh', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('time', DataTypes.DATE, InputTypes.TIME),
        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('dewpoint_2m_K', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('temperature_K', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('mean_sealev_pressure_hPa', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('surface_pressure_hPa', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('precipitation_m', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('wind_speed_10_ms', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('u_wind_10_ms', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('v_wind_10_ms', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('instant_wind_gust_ms', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('post_process_wind_gust_ms', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Day sin', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('Day cos', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('Year sin', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('Year cos', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
    ]

    def __init__(self, data_folder: str, inference: bool):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self.data_folder = data_folder
        self.inference = inference
        self.save_path: str = os.path.join(self.data_folder, "fixed")
        self._time_steps = self.get_fixed_params()['total_time_steps']

    def split_data(self, df: DataFrame) -> (DataFrame, DataFrame, DataFrame):
        """Splits data frame into training-validation-test data frames.
            This also calibrates scaling object, and transforms data for each split.
            Args:
              df: Source data frame to split.
            Returns:
              Tuple of transformed (train, valid, test) data.
        """
        index: Series = df['days_from_start']
        train: DataFrame = df.loc[index < int(index.max() * 0.7)]
        valid: DataFrame = df.loc[(index >= int(index.max() * 0.7)) & (index < int(index.max() * 0.9))]
        test: DataFrame = df.loc[index >= int(index.max() * 0.9)]

        self.set_scalers(train)
        # save scalers to serialized format
        self.save_scalers()
        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df: DataFrame):
        """Calibrates scalers using the data supplied.
            Args:
              df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')
        column_definitions: List = self.get_column_definition()
        id_column: str = utils.get_single_col_by_input_type(InputTypes.ID,
                                                            column_definitions)
        target_column: str = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                                column_definitions)

        # Format real scalers
        real_inputs: List = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        # Initialise scaler caches
        self._real_scalers = {}
        self._target_scaler = {}
        identifiers = []
        for identifier, sliced in df.groupby(id_column):

            if len(sliced) >= self._time_steps:
                data = sliced[real_inputs].values
                targets = sliced[[target_column]].values
                self._real_scalers[identifier] \
                    = pp.StandardScaler().fit(data)

                self._target_scaler[identifier] \
                    = pp.StandardScaler().fit(targets)
            identifiers.append(identifier)

        # Format categorical scalers
        categorical_inputs: List = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = pp.LabelEncoder().fit(
                srs.values)
            num_classes.append(srs.nunique())
        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

        # Extract identifiers in case required
        self.identifiers = identifiers

    def transform_inputs(self, df: DataFrame) -> DataFrame:
        """Performs feature transformations.
            This includes both feature engineering, preprocessing and normalisation.
            Args:
              df: Data frame to transform.
            Returns:
              Transformed data frame.
        """
        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        # Extract relevant columns
        column_definitions: List = self.get_column_definition()
        id_col: str = utils.get_single_col_by_input_type(InputTypes.ID,
                                                         column_definitions)
        real_inputs: List = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs: List = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Transform real inputs per entity
        df_list = []
        for identifier, sliced in df.groupby(id_col):

            # Filter out any trajectories that are too short
            if len(sliced) >= self._time_steps:
                sliced_copy = sliced.copy()
                sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
                    sliced_copy[real_inputs].values)
                df_list.append(sliced_copy)

        output: DataFrame = pd.concat(df_list, axis=0)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions: DataFrame) -> DataFrame:
        """Reverts any normalisation to give predictions in original scale.
            Args:
              predictions: Dataframe of model predictions.
            Returns:
              Data frame of unnormalised predictions.
        """
        if self._target_scaler is None:
            raise ValueError('Scalers have not been set!')

        column_names = predictions.columns
        df_list = []
        for identifier, sliced in predictions.groupby('identifier'):
            sliced_copy = sliced.copy()
            target_scaler = self._target_scaler[identifier]
            for col in column_names:
                if col not in {'forecast_time', 'identifier'}:
                    sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[col])
            df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        return output

        # Default params

    def get_fixed_params(self) -> Dict:
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': (7 * 24) + 12,
            'num_encoder_steps': 7 * 24,
            'num_epochs': 100,
            'early_stopping_patience': 10,
            'multiprocessing_workers': 5
        }
        # read params from data_folder
        if self.inference:
            params_path: str = os.path.join(self.data_folder, 'params.csv')
            saved_params: DataFrame = pd.read_csv(params_path, index_col=0, header=0, names=['data'])
            fixed_params['category_counts'] = json.loads(saved_params.loc['category_counts', 'data'])

        return fixed_params

    def get_default_model_params(self) -> Dict:
        """Returns default optimised model parameters."""

        model_params = {
            'dropout_rate': 0.1,
            'hidden_layer_size': 160,
            'learning_rate': 0.001,
            'minibatch_size': 64,
            'max_gradient_norm': 0.01,
            'num_heads': 4,
            'stack_size': 1
        }

        return model_params

    def get_num_samples_for_calibration(self) -> (int, int):
        """Gets the default number of training and validation samples.
            Use to sub-sample the data for network calibration and a value of -1 uses
            all available samples.
            Returns:
              Tuple of (training samples, validation samples)
        """
        return 12280, 3508

    def save_scalers(self):
        """
        This method saves the scalers into serialized format in order to re-use them for inference without having to
        load the dataset and apply the split_data method to fit the scalers
        :return: None
        """

        if not os.path.exists(os.path.join(self.data_folder, "scalers")):
            os.makedirs(os.path.join(self.data_folder, "scalers"))
        with open(os.path.join(self.data_folder, "scalers", "real_scalers.pkl"), "wb") as real, open(
                os.path.join(self.data_folder, "scalers", "cat_scalers.pkl"), "wb") as cat, open(
            os.path.join(self.data_folder, "scalers", "target_scaler.pkl"), "wb") as tar:
            pickle.dump(self._real_scalers, real)
            pickle.dump(self._cat_scalers, cat)
            pickle.dump(self._target_scaler, tar)

    def load_scalers(self):
        """
         Loads the saved scalers for inference
        :return: None
        """

        if os.path.exists(os.path.join(self.data_folder, "scalers")):
            with open(os.path.join(self.data_folder, "scalers", "real_scalers.pkl"), "rb") as real, open(
                    os.path.join(self.data_folder, "scalers", "cat_scalers.pkl"), "rb") as cat, open(
                os.path.join(self.data_folder, "scalers", "target_scaler.pkl"), "rb") as tar:
                self._real_scalers = pickle.load(real)
                self._cat_scalers = pickle.load(cat)
                self._target_scaler = pickle.load(tar)
        else:
            raise ValueError('There are no saved scalers')
