from data_formatters.base import GenericDataFormatter, DataTypes, InputTypes
import pandas as pd
import sklearn.preprocessing as pp
from typing import Tuple, Dict, List
from pandas import DataFrame
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
        ('', )
    ]
