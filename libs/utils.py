import os
import pathlib
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.compat.v1 import ConfigProto
from tensorflow import Tensor
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from typing import List, Set, Type, Union
from numpy import ndarray
from pandas import Series, DataFrame
from data_formatters.base import InputTypes, DataTypes


# Generic.
def get_single_col_by_input_type(input_type: InputTypes, column_definition: List) -> str:
    """Returns name of single column.
  Args:
    input_type: Input type of column to extract
    column_definition: Column definition list for experiment
  """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError('Invalid number of columns for {}'.format(input_type))

    return l[0]


def extract_cols_from_data_type(data_type: DataTypes, column_definition,
                                excluded_input_types: Set) -> List:
    """Extracts the names of columns that correspond to a define data_type.
  Args:
    data_type: DataType of columns to extract.
    column_definition: Column definition to use.
    excluded_input_types: Set of input types to exclude
  Returns:
    List of names for columns with data type specified.
  """
    return [
        tup[0]
        for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


# Loss functions.
def tensorflow_quantile_loss(y: Tensor, y_pred: Tensor, quantile: float) -> Tensor:
    """Computes quantile loss for tensorflow.
  Standard quantile loss as defined in the "Training Procedure" section of
  the main TFT paper
  Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)
  Returns:
    Tensor for quantile loss.
  """

    # Checks quantile
    if quantile < 0 or quantile > 1:
        raise ValueError(
            'Illegal quantile value={}! Values should be between 0 and 1.'.format(
                quantile))

    prediction_underflow: Tensor = y - y_pred
    q_loss: Tensor = quantile * tf.maximum(prediction_underflow, 0.) + (
            1. - quantile) * tf.maximum(-prediction_underflow, 0.)

    return tf.reduce_sum(q_loss, axis=-1)


def numpy_normalised_quantile_loss(y: DataFrame, y_pred: DataFrame, quantile: float) -> Series:
    """Computes normalised quantile loss for numpy arrays.
  Uses the q-Risk metric as defined in the "Training Procedure" section of the
  main TFT paper.
  Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)
  Returns:
    Float for normalised quantile loss.
  """
    prediction_underflow: DataFrame = y - y_pred
    weighted_errors: DataFrame = quantile * np.maximum(prediction_underflow, 0.) \
                               + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

    quantile_loss: Series = weighted_errors.mean()
    normaliser: Series = y.abs().mean()

    return 2 * quantile_loss / normaliser


# OS related functions.
def create_folder_if_not_exist(directory: Union[str, Path]):
    """Creates folder if it doesn't exist.
  Args:
    directory: Folder path to create.
  """
    # Also creates directories recursively
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# Tensorflow related functions.
def get_default_tensorflow_config(tf_device: str = 'gpu', gpu_id: int = 0) -> ConfigProto:
    """Creates tensorflow config for graphs to run on CPU or GPU.
  Specifies whether to run graph on gpu or cpu and which GPU ID to use for multi
  GPU machines.
  Args:
    tf_device: 'cpu' or 'gpu'
    gpu_id: GPU ID to use if relevant
  Returns:
    Tensorflow config.
  """

    if tf_device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # for training on cpu
        tf_config = tf1.ConfigProto(
            log_device_placement=False, device_count={'GPU': 0})

    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print('Selecting GPU ID={}'.format(gpu_id))

        tf_config: ConfigProto = tf1.ConfigProto(log_device_placement=False)
        tf_config.gpu_options.allow_growth = True

    return tf_config


def save(tf_session, model_folder, cp_name, scope=None):
    """Saves Tensorflow graph to checkpoint.
  Saves all trainiable variables under a given variable scope to checkpoint.
  Args:
    tf_session: Session containing graph
    model_folder: Folder to save models
    cp_name: Name of Tensorflow checkpoint
    scope: Variable scope containing variables to save
  """
    # Save model
    if scope is None:
        saver = tf1.train.Saver()
    else:
        var_list = tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        saver = tf1.train.Saver(var_list=var_list, max_to_keep=100000)

    save_path = saver.save(tf_session,
                           os.path.join(model_folder, '{0}.ckpt'.format(cp_name)))
    print('Model saved to: {0}'.format(save_path))


def load(tf_session, model_folder, cp_name, scope=None, verbose=False):
    """Loads Tensorflow graph from checkpoint.
  Args:
    tf_session: Session to load graph into
    model_folder: Folder containing serialised model
    cp_name: Name of Tensorflow checkpoint
    scope: Variable scope to use.
    verbose: Whether to print additional debugging information.
  """
    # Load model proper
    load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

    print('Loading model from {0}'.format(load_path))

    print_weights_in_checkpoint(model_folder, cp_name)

    initial_vars = set(
        [v.name for v in tf1.get_default_graph().as_graph_def().node])

    # Saver
    if scope is None:
        saver = tf1.train.Saver()
    else:
        var_list = tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf1.train.Saver(var_list=var_list, max_to_keep=100000)
    # Load
    saver.restore(tf_session, load_path)
    all_vars = set([v.name for v in tf1.get_default_graph().as_graph_def().node])

    if verbose:
        print('Restored {0}'.format(','.join(initial_vars.difference(all_vars))))
        print('Existing {0}'.format(','.join(all_vars.difference(initial_vars))))
        print('All {0}'.format(','.join(all_vars)))

    print('Done.')


def print_weights_in_checkpoint(model_folder: str, cp_name: str):
    """Prints all weights in Tensorflow checkpoint.
  Args:
    model_folder: Folder containing checkpoint
    cp_name: Name of checkpoint
  Returns:
  """
    load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

    print_tensors_in_checkpoint_file(
        file_name=load_path,
        tensor_name='',
        all_tensors=True,
        all_tensor_names=True)
