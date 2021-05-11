from data_formatters.base import GenericDataFormatter, DataTypes, InputTypes
import pandas as pd
import sklearn.preprocessing as pp
from typing import Tuple, Dict, List
from pandas import DataFrame, Series, DatetimeIndex
from libs import utils


class ErgFormatter(GenericDataFormatter):
    """Defines and formats data for the electricity dataset.
      Note that per-entity z-score normalization is used here, and is implemented
      across functions.
      Attributes:
        column_definition: Defines input and data type of column used in the
          experiment.
        identifiers: Entity identifiers used in experiments.
      """
    _column_definition = [
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('energy_mw', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('Wind Speed', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('2m_devpoint [C]', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('temperature [C]', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('mean_sealev_pressure [hPa]', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('surface pressure [hPa]', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('precipitation [m]', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10_wind_speed', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10_u_wind', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10_v_wind', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('instant_wind_gust [m/s]', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Day sin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Day cos', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Year sin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Year cos', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT)
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self._time_steps = self.get_fixed_params()['total_time_steps']

    def split_data(self, df: DataFrame):
        """Splits data frame into training-validation-test data frames.
            This also calibrates scaling object, and transforms data for each split.
            Args:
              df: Source data frame to split.
            Returns:
              Tuple of transformed (train, valid, test) data.
        """
        # df.set_index('time', drop=True, inplace=True)
        # date_index: DatetimeIndex = pd.date_range(start=df.time.min(), end=df.time.max(),
        #                                           freq=pd.offsets.Hour(1))
        index: Series = df['days_from_start']
        train: DataFrame = df.loc[index < int(index.max()*0.7)]
        valid: DataFrame = df.loc[(index >= int(index.max()*0.7)) & (index < int(index.max()*0.9))]
        test: DataFrame = df.loc[index >= int(index.max()*0.9)]

        self.set_scalers(train)

    def set_scalers(self, df: DataFrame):
        """Calibrates scalers using the data supplied.
            Args:
              df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')
        column_definitions = self.get_column_definition()

    def get_column_definition(self):





