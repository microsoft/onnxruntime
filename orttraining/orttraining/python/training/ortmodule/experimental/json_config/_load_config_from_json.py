# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _load_config_from_json.py

import json
import logging
import os
from functools import reduce
from types import SimpleNamespace

from onnxruntime.capi import _pybind_state as C
from onnxruntime.training.ortmodule._fallback import _FallbackPolicy
from onnxruntime.training.ortmodule.options import DebugOptions, LogLevel, _SaveOnnxOptions, _SkipCheck

from . import JSON_PATH_ENVIRONMENT_KEY

log = logging.getLogger(__name__)


def _load_data_from_json(path):
    """Loads data from the json file path provided."""

    data = None
    with open(path) as f:
        data = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    if data is None:
        raise RuntimeError(f"No data found in provided json file {path}.")

    return data


def _load_propagate_cast_ops(ortmodule_config_accessor, data):
    """Loads PropagateCastOps from json file onto ORTModule."""

    assert hasattr(data, _load_propagate_cast_ops.loading_key)
    log.info(f"Found keyword {_load_propagate_cast_ops.loading_key} in json. Loading attributes from file.")

    def _update_strategy():
        ortmodule_config_accessor._runtime_options.propagate_cast_ops_strategy = C.PropagateCastOpsStrategy.__members__[
            data.PropagateCastOps.Strategy
        ]

    def _update_level():
        ortmodule_config_accessor._runtime_options.propagate_cast_ops_level = data.PropagateCastOps.Level

    def _update_allow():
        ortmodule_config_accessor._runtime_options.propagate_cast_ops_allow = data.PropagateCastOps.Allow

    key_to_function_mapping = {"Strategy": _update_strategy, "Level": _update_level, "Allow": _update_allow}

    for key in data.PropagateCastOps.__dict__:
        key_to_function_mapping[key]()


def _load_use_external_gpu_allocator(ortmodule_config_accessor, data):
    """Loads UseExternalGPUAllocator from json file onto ORTModule."""

    assert hasattr(data, _load_use_external_gpu_allocator.loading_key)
    log.info(f"Found keyword {_load_use_external_gpu_allocator.loading_key} in json. Loading attributes from file.")

    assert isinstance(
        data.UseExternalGPUAllocator, bool
    ), f"{_load_use_external_gpu_allocator.loading_key} must be a boolean"
    ortmodule_config_accessor._runtime_options.use_external_gpu_allocator = data.UseExternalGPUAllocator


def _load_enable_custom_autograd_function(ortmodule_config_accessor, data):
    """Loads EnableCustomAutogradFunction from json file onto ORTModule."""

    assert hasattr(data, _load_enable_custom_autograd_function.loading_key)
    log.info(
        f"Found keyword {_load_enable_custom_autograd_function.loading_key} in json. Loading attributes from file."
    )

    assert isinstance(
        data.EnableCustomAutogradFunction, bool
    ), f"{_load_enable_custom_autograd_function.loading_key} must be a boolean"

    from onnxruntime.training.ortmodule._custom_autograd_function import enable_custom_autograd_support

    enable_custom_autograd_support(data.EnableCustomAutogradFunction)
    ortmodule_config_accessor._runtime_options.enable_custom_autograd_function = data.EnableCustomAutogradFunction


def _load_enable_grad_acc_optimization(ortmodule_config_accessor, data):
    """Loads EnableGradAccOptimization from json file onto ORTModule."""

    assert hasattr(data, _load_enable_grad_acc_optimization.loading_key)
    log.info(f"Found keyword {_load_enable_grad_acc_optimization.loading_key} in json. Loading attributes from file.")

    assert isinstance(
        data.EnableGradAccOptimization, bool
    ), f"{_load_enable_grad_acc_optimization.loading_key} must be a boolean"
    ortmodule_config_accessor._runtime_options.enable_grad_acc_optimization = data.EnableGradAccOptimization


def _load_run_symbolic_shape_infer(ortmodule_config_accessor, data):
    """Loads RunSymbolicShapeInference from json file onto ORTModule."""

    assert hasattr(data, _load_run_symbolic_shape_infer.loading_key)
    log.info(f"Found keyword {_load_run_symbolic_shape_infer.loading_key} in json. Loading attributes from file.")

    assert isinstance(
        data.RunSymbolicShapeInference, bool
    ), f"{_load_run_symbolic_shape_infer.loading_key} must be a boolean"
    ortmodule_config_accessor._runtime_options.run_symbolic_shape_infer = data.RunSymbolicShapeInference


def _load_use_static_shape(ortmodule_config_accessor, data):
    """Loads UseStaticShape from json file onto ORTModule."""

    assert hasattr(data, _load_use_static_shape.loading_key)
    log.info(f"Found keyword {_load_use_static_shape.loading_key} in json. Loading attributes from file.")

    assert isinstance(data.UseStaticShape, bool), f"{_load_use_static_shape.loading_key} must be a boolean"
    ortmodule_config_accessor._runtime_options.use_static_shape = data.UseStaticShape


def _load_skip_check(ortmodule_config_accessor, data):
    """Loads SkipCheck from json file onto ORTModule."""

    assert hasattr(data, _load_skip_check.loading_key)
    log.info(f"Found keyword {_load_skip_check.loading_key} in json. Loading attributes from file.")

    skip_check = reduce(lambda x, y: x | y, [_SkipCheck[name] for name in data.SkipCheck])
    if skip_check.value > 0:
        ortmodule_config_accessor._runtime_options.skip_check = skip_check


def _load_debug_options(ortmodule_config_accessor, data):
    """Loads DebugOptions from json file onto ORTModule."""

    assert hasattr(data, _load_debug_options.loading_key)
    log.info(f"Found keyword {_load_debug_options.loading_key} in json. Loading attributes from file.")

    log_level = LogLevel.WARNING

    def _update_log_level():
        nonlocal log_level
        log_level = LogLevel[data.DebugOptions.LogLevel]

    save_onnx = False

    def _update_save_onnx():
        nonlocal save_onnx
        save_onnx = data.DebugOptions.SaveONNX

    onnx_prefix = ""

    def _update_onnx_prefix():
        nonlocal onnx_prefix
        onnx_prefix = data.DebugOptions.ONNXPrefix

    def _update_onnx_path():
        os.environ[_SaveOnnxOptions._path_environment_key] = data.DebugOptions.SaveONNXPath

    key_to_function_mapping = {
        "LogLevel": _update_log_level,
        "SaveONNX": _update_save_onnx,
        "ONNXPrefix": _update_onnx_prefix,
        "SaveONNXPath": _update_onnx_path,
    }

    for key in data.DebugOptions.__dict__:
        key_to_function_mapping[key]()

    debug_options = DebugOptions(log_level=log_level, save_onnx=save_onnx, onnx_prefix=onnx_prefix)
    ortmodule_config_accessor._debug_options = debug_options


def _load_use_memory_efficient_gradient(ortmodule_config_accessor, data):
    """Loads UseMemoryEfficientGradient from json file onto ORTModule."""

    assert hasattr(data, _load_use_memory_efficient_gradient.loading_key)
    log.info(f"Found keyword {_load_use_memory_efficient_gradient.loading_key} in json. Loading attributes from file.")

    assert isinstance(
        data.UseMemoryEfficientGradient, bool
    ), f"{_load_use_memory_efficient_gradient.loading_key} must be a boolean"
    ortmodule_config_accessor._runtime_options.use_memory_efficient_gradient = data.UseMemoryEfficientGradient


def _load_fallback_policy(ortmodule_config_accessor, data):
    """Loads SkipCheck from json file onto ORTModule."""

    assert hasattr(data, _load_fallback_policy.loading_key)
    log.info(f"Found keyword {_load_fallback_policy.loading_key} in json. Loading attributes from file.")

    fallback_policy = reduce(lambda x, y: x | y, [_FallbackPolicy[name] for name in data.FallbackPolicy])
    if fallback_policy.value > 0:
        ortmodule_config_accessor._fallback_manager.policy = fallback_policy


def _load_onnx_opset_version(ortmodule_config_accessor, data):
    """Loads OnnxOpsetVersion from json file onto ORTModule."""

    assert hasattr(data, _load_onnx_opset_version.loading_key)
    log.info(f"Found keyword {_load_onnx_opset_version.loading_key} in json. Loading attributes from file.")

    assert isinstance(data.OnnxOpsetVersion, int), f"{_load_onnx_opset_version.loading_key} must be an int"
    ortmodule_config_accessor._runtime_options.onnx_opset_version = data.OnnxOpsetVersion


def _define_load_function_keys():
    """Define static key variables for each loading function"""

    _load_propagate_cast_ops.loading_key = "PropagateCastOps"
    _load_use_external_gpu_allocator.loading_key = "UseExternalGPUAllocator"
    _load_enable_custom_autograd_function.loading_key = "EnableCustomAutogradFunction"
    _load_enable_grad_acc_optimization.loading_key = "EnableGradAccOptimization"
    _load_run_symbolic_shape_infer.loading_key = "RunSymbolicShapeInference"
    _load_use_static_shape.loading_key = "UseStaticShape"
    _load_skip_check.loading_key = "SkipCheck"
    _load_debug_options.loading_key = "DebugOptions"
    _load_use_memory_efficient_gradient.loading_key = "UseMemoryEfficientGradient"
    _load_fallback_policy.loading_key = "FallbackPolicy"
    _load_onnx_opset_version.loading_key = "OnnxOpsetVersion"


def load_from_json(ortmodule, path=None):
    """Loads config from json file at given path.

    Here is the schema that the json file must adhere to:
    {
        "PropagateCastOps":
        {
            "Strategy": "FLOOD_FILL", # str representing strategy ("NONE", "FLOOD_FILL", or "INSERT_AND_REDUCE")
            "Level": 3, # propagate cast ops level as an int
            "Allow": ["ABC", "DEF"] # propagate cast ops allow as list of strs
        },
        "UseExternalGPUAllocator" : false, # bool flag
        "EnableCustomAutogradFunction": true, # bool flag
        "EnableGradAccOptimization": true, # bool flag
        "UseStaticShape": true, # bool flag
        "RunSymbolicShapeInference": false, # bool flag
        "SkipCheck": # list of strs representing `_SkipCheck`s checks to skip which will be aggregated using |
        [
            "SKIP_CHECK_DEVICE",
            "SKIP_CHECK_BUILD_GRADIENT",
            "SKIP_CHECK_EXECUTION_AGENT"
        ],
        "DebugOptions": # debug options for user facing configuration
        {
            "LogLevel": "VERBOSE",
            "SaveONNX": true,
            "ONNXPrefix": "my_model",
            "SaveONNXPath": "/path/to/onnx/directory"
        },
        "FallbackPolicy": # list of strings representing fallback policies (`_FallbackPolicy`s which can be aggregated using |
        [
            "FALLBACK_DISABLE",
            "FALLBACK_FORCE_TORCH_FORWARD",
            "FALLBACK_UNSUPPORTED_DEVICE",
            "FALLBACK_UNSUPPORTED_DATA",
            "FALLBACK_UNSUPPORTED_TORCH_MODEL",
            "FALLBACK_UNSUPPORTED_ONNX_MODEL",
            "FALLBACK_BAD_INITIALIZATION",
        ],
        "OnnxOpsetVersion": 14 # int defining the opset version to be used during export
    }

    Args:
        ortmodule (:obj:`ORTModule`): ORTModule instance that needs to be configured
        path (:obj:`str`, optional): Path to json file. Alternatively, users can set the
            environment variable ORTMODULE_JSON_CONFIG_PATH to the json config path. In case
            both path and environment variable are set, the environment variable gets precedence.
    """

    path = os.getenv(JSON_PATH_ENVIRONMENT_KEY, path)

    # figure out the json path
    if path is None:
        raise ValueError(
            "Path to json is not provided."
            f"Provide the path through function call or setting the environment variable {JSON_PATH_ENVIRONMENT_KEY}"
        )

    # load the entire json file
    data = _load_data_from_json(path)

    # define the keys for all loading functions
    _define_load_function_keys()

    # define all load functions to iterate over
    load_functions = {
        _load_propagate_cast_ops.loading_key: _load_propagate_cast_ops,
        _load_use_external_gpu_allocator.loading_key: _load_use_external_gpu_allocator,
        _load_enable_custom_autograd_function.loading_key: _load_enable_custom_autograd_function,
        _load_enable_grad_acc_optimization.loading_key: _load_enable_grad_acc_optimization,
        _load_run_symbolic_shape_infer.loading_key: _load_run_symbolic_shape_infer,
        _load_use_static_shape.loading_key: _load_use_static_shape,
        _load_skip_check.loading_key: _load_skip_check,
        _load_debug_options.loading_key: _load_debug_options,
        _load_use_memory_efficient_gradient.loading_key: _load_use_memory_efficient_gradient,
        _load_fallback_policy.loading_key: _load_fallback_policy,
        _load_onnx_opset_version.loading_key: _load_onnx_opset_version,
    }

    for training_mode in [True, False]:
        # update the debug config for both train and eval modes
        ortmodule_config_accessor = ortmodule._torch_module._execution_manager(training_mode)
        # iterate over the json data instead of checking for keys in json to catch key errors
        for key in data.__dict__:
            load_functions[key](ortmodule_config_accessor, data)
