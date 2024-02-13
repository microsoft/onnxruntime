# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import List, Tuple, Union

import onnx

from onnxruntime.capi._pybind_state import get_model_after_loading_checkpoint as _load_checkpoint_to_model
from onnxruntime.capi._pybind_state import save_checkpoint as _save_checkpoint


def save_checkpoint(
    parameters: Tuple[List[onnx.TensorProto], List[onnx.TensorProto]],
    path_to_checkpoint: Union[str, os.PathLike],
    nominal_checkpoint: bool = False,
) -> None:
    """Saves the parameters to the checkpoint directory path_to_checkpoint.

    Args:
        parameters tuple(trainable_params, non_trainable_params): The parameters to save to the checkpoint file.
        path_to_checkpoint: The path to the checkpoint directory.
        nominal_checkpoint: If True, the checkpoint is saved as a nominal checkpoint. Default is False.
    """

    if parameters is None:
        raise RuntimeError("No checkpoint parameters provided.")

    trainable_params, non_trainable_params = parameters
    trainable_params = [param.SerializeToString() for param in trainable_params]
    non_trainable_params = [param.SerializeToString() for param in non_trainable_params]
    _save_checkpoint(trainable_params, non_trainable_params, os.fspath(path_to_checkpoint), nominal_checkpoint)


def load_checkpoint_to_model(path_to_checkpoint: Union[str, os.PathLike], model: onnx.ModelProto) -> None:
    """Loads the checkpoint to an onnx inference model.

    Args:
        path_to_checkpoint (str): The path to the checkpoint directory.
        model (onnx.ModelProto): The model to load the checkpoint to.
    """

    model.ParseFromString(_load_checkpoint_to_model(os.fspath(path_to_checkpoint), model.SerializeToString()))
