# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import importlib.util
import os
import sys
from functools import wraps  # noqa: F401

import numpy as np
import torch
from onnx import TensorProto  # noqa: F401
from packaging.version import Version


def get_device_index(device):
    if isinstance(device, str):
        # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
        device = torch.device(device)
    elif isinstance(device, int):
        return device
    return 0 if device.index is None else device.index


def get_device_index_from_input(input):
    """Returns device index from a input PyTorch Tensor"""

    if isinstance(input, (list, tuple)):
        device_index = get_device_index(input[0].device)
    else:
        device_index = get_device_index(input.device)
    return device_index


def get_device_str(device):
    if isinstance(device, str):
        # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
        if device.find(":") == -1:
            device += ":" + str(torch.cuda.current_device())
    elif isinstance(device, int):
        device = "cuda:" + str(device)
    elif isinstance(device, torch.device):
        if device.index is None:
            device = device.type + ":" + str(torch.cuda.current_device())
        else:
            device = device.type + ":" + str(device.index)
    else:
        raise RuntimeError("Unsupported device type")
    return device


def get_all_gradients_finite_name_from_session(session):
    """Find all_gradients_finite node on Session graph and return its name"""

    nodes = [x for x in session._outputs_meta if "all_gradients_finite" in x.name]
    if len(nodes) != 1:
        raise RuntimeError("'all_gradients_finite' node not found within training session")
    return nodes[0].name


def get_gradient_accumulation_name_from_session(session):
    """Find Group_Accumulated_Gradients node on Session graph and return its name"""

    nodes = [x for x in session._outputs_meta if "Group_Accumulated_Gradients" in x.name]
    if len(nodes) != 1:
        raise RuntimeError("'Group_Accumulated_Gradients' node not found within training session")
    return nodes[0].name


def dtype_torch_to_numpy(torch_dtype):
    """Converts PyTorch types to Numpy types

    Also must map to types accepted by:
        MLDataType NumpyTypeToOnnxRuntimeType(int numpy_type)

    References:
        https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
        https://pytorch.org/docs/stable/tensors.html
    """
    if torch_dtype == torch.float64 or torch_dtype == torch.double:
        return np.float64
    elif torch_dtype == torch.float32 or torch_dtype == torch.float:
        return np.float32
    elif torch_dtype == torch.float16 or torch_dtype == torch.half or torch_dtype == torch.bfloat16:
        # NOTE: numpy doesn't support bfloat16
        return np.float16
    elif torch_dtype == torch.int64 or torch_dtype == torch.long:
        return np.longlong  # np.int64 doesn't work!?
    elif torch_dtype == torch.int32 or torch_dtype == torch.int:
        return np.int32
    elif torch_dtype == torch.int16 or torch_dtype == torch.short:
        return np.int16
    elif torch_dtype == torch.int8:
        return np.int8
    elif torch_dtype == torch.uint8:
        return np.uint8
    elif torch_dtype == torch.complex64 or (
        # complex32 is missing in torch-1.11.
        (Version(torch.__version__) < Version("1.11.0") or Version(torch.__version__) >= Version("1.12.0"))
        and torch_dtype == torch.complex32
    ):
        # NOTE: numpy doesn't support complex32
        return np.complex64
    elif torch_dtype == torch.complex128 or torch_dtype == torch.cdouble:
        return np.complex128
    elif torch_dtype == torch.bool:
        return np.bool_
    else:
        raise ValueError(f"torch_dtype ({torch_dtype!s}) type is not supported by Numpy")


def dtype_onnx_to_torch(onnx_type):
    """Converts ONNX types to PyTorch types

    Reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.in.proto (enum DataType)
               https://pytorch.org/docs/stable/tensors.html
    """
    onnx_types = [
        "UNDEFINED",
        "FLOAT",
        "UINT8",
        "INT8",
        "UINT16",
        "INT16",
        "INT32",
        "INT64",
        "STRING",
        "BOOL",
        "FLOAT16",
        "DOUBLE",
        "UINT32",
        "UINT64",
        "COMPLEX64",
        "COMPLEX128",
        "BFLOAT16",
        "FLOAT8E4M3FN",
        "FLOAT8E4M3FNUZ",
        "FLOAT8E5M2",
        "FLOAT8E5M2FNUZ",
    ]

    if isinstance(onnx_type, int):
        assert onnx_type < len(onnx_types), "Invalid onnx_type integer"
    elif isinstance(onnx_type, str):
        onnx_type = onnx_type.upper()
        assert onnx_type in onnx_types, "Invalid onnx_type string"
        onnx_type = onnx_types.index(onnx_type)
    else:
        raise ValueError("'onnx_type' must be an ONNX type represented by either a string or integer")

    if onnx_type == 0:
        return None
    elif onnx_type == 1:
        return torch.float
    elif onnx_type >= 2 and onnx_type <= 3:
        # NOTE: Pytorch doesn't support uint8
        return torch.int8
    elif onnx_type >= 4 and onnx_type <= 5:
        # NOTE: Pytorch doesn't support int16
        return torch.int16
    elif onnx_type == 6 or onnx_type == 12:
        # NOTE: Pytorch doesn't support uint32
        return torch.int32
    elif onnx_type == 7 or onnx_type == 13:
        # NOTE: Pytorch doesn't support uint64
        return torch.int64
    elif onnx_type == 8:
        return str
    elif onnx_type == 9:
        return torch.bool
    elif onnx_type == 10:
        return torch.float16
    elif onnx_type == 11:
        return torch.double
    elif onnx_type == 14:
        return torch.complex64
    elif onnx_type == 15:
        return torch.complex128
    elif onnx_type == 16:
        return torch.bfloat


def static_vars(**kwargs):
    r"""Decorator to add :py:attr:`kwargs` as static vars to 'func'

    Example:

        .. code-block:: python

            >>> @static_vars(counter=0)
            ... def myfync():
            ...     myfync.counter += 1
            ...     return myfync.counter
            ...
            >>> print(myfunc())
            1
            >>> print(myfunc())
            2
            >>> print(myfunc())
            3
            >>> myfunc.counter = 100
            >>> print(myfunc())
            101
    """

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def import_module_from_file(file_path, module_name=None):
    """Import a Python module from a file into interpreter"""

    if not isinstance(file_path, str) or not os.path.exists(file_path):
        raise AssertionError(
            f"'file_path' must be a full path string with the python file to load. file_path={file_path!r}."
        )
    if module_name is not None and (not isinstance(module_name, str) or not module_name):
        raise AssertionError(
            "'module_name' must be a string with the python module name to load. module_name={module_name!r}."
        )

    if not module_name:
        module_name = os.path.basename(file_path).split(".")[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def state_dict_model_key():
    """Returns the model key name in the state dictionary"""

    return "model"


def state_dict_optimizer_key():
    """Returns the optimizer key name in the state dictionary"""

    return "optimizer"


def state_dict_partition_info_key():
    """Returns the partition info key name in the state dictionary"""

    return "partition_info"


def state_dict_trainer_options_key():
    """Returns the trainer options key name in the state dictionary"""

    return "trainer_options"


def state_dict_full_precision_key():
    """Returns the full precision key name in the state dictionary"""

    return "full_precision"


def state_dict_original_dimension_key():
    """Returns the original dimension key name in the state dictionary"""

    return "original_dim"


def state_dict_sharded_optimizer_keys():
    """Returns the optimizer key names that can be sharded in the state dictionary"""

    return {"Moment_1", "Moment_2"}


def state_dict_user_dict_key():
    """Returns the user dict key name in the state dictionary"""

    return "user_dict"


def state_dict_trainer_options_mixed_precision_key():
    """Returns the trainer options mixed precision key name in the state dictionary"""

    return "mixed_precision"


def state_dict_trainer_options_zero_stage_key():
    """Returns the trainer options zero_stage key name in the state dictionary"""

    return "zero_stage"


def state_dict_trainer_options_world_rank_key():
    """Returns the trainer options world_rank key name in the state dictionary"""

    return "world_rank"


def state_dict_trainer_options_world_size_key():
    """Returns the trainer options world_size key name in the state dictionary"""

    return "world_size"


def state_dict_trainer_options_data_parallel_size_key():
    """Returns the trainer options data_parallel_size key name in the state dictionary"""

    return "data_parallel_size"


def state_dict_trainer_options_horizontal_parallel_size_key():
    """Returns the trainer options horizontal_parallel_size key name in the state dictionary"""

    return "horizontal_parallel_size"


def state_dict_trainer_options_optimizer_name_key():
    """Returns the trainer options optimizer_name key name in the state dictionary"""

    return "optimizer_name"


def state_dict_train_step_info_key():
    """Returns the train step info key name in the state dictionary"""

    return "train_step_info"


def state_dict_train_step_info_optimization_step_key():
    """Returns the train step info optimization step key name in the state dictionary"""

    return "optimization_step"


def state_dict_train_step_info_step_key():
    """Returns the train step info step key name in the state dictionary"""

    return "step"
