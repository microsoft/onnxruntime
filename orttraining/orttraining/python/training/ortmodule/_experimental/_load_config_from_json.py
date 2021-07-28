# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _load_config_from_json.py

import json
import os
from types import SimpleNamespace

from onnxruntime.capi import _pybind_state as C
from functools import reduce
from .._graph_execution_manager import _SkipCheck
from ..debug_options import DebugOptions, LogLevel

def load_data_from_json(path):
    """Loads data from the json file path provided."""

    data = None
    with open(path) as f:
        data = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    if data is None:
        raise RuntimeError(f"No data found in provided json file {path}.")

    return data

def load_propagate_cast_ops(ortmodule_config_accessor, data):
    """Load PropagateCastOps from json file onto ORTModule"""

    propagate_cast_ops_str = 'PropagateCastOps'
    if not hasattr(data, propagate_cast_ops_str):
        return
    print(f"Found keyword {propagate_cast_ops_str} in json. Loading attributes from file.")

    if hasattr(data.PropagateCastOps, 'Strategy'):
        ortmodule_config_accessor._propagate_cast_ops_strategy = \
            C.PropagateCastOpsStrategy.__members__[data.PropagateCastOps.Strategy]

    if hasattr(data.PropagateCastOps, 'Level'):
        ortmodule_config_accessor._propagate_cast_ops_level = data.PropagateCastOps.Level

    if hasattr(data.PropagateCastOps, 'Allow'):
        ortmodule_config_accessor._propagate_cast_ops_allow = data.PropagateCastOps.Allow

def load_use_external_gpu_allocator(ortmodule_config_accessor, data):
    """Load UseExternalGPUAllocator from json file onto ORTModule"""

    use_external_gpu_allocator_str = 'UseExternalGPUAllocator'
    if not hasattr(data, use_external_gpu_allocator_str):
        return
    print(f"Found keyword {use_external_gpu_allocator_str} in json. Loading attributes from file.")

    assert isinstance(data.UseExternalGPUAllocator, bool), f"{use_external_gpu_allocator_str} must be a boolean"
    ortmodule_config_accessor._use_external_gpu_allocator = data.UseExternalGPUAllocator
    ortmodule_config_accessor._get_torch_gpu_allocator_function_addresses()

def load_enable_custom_autograd_function(ortmodule_config_accessor, data):
    """Load EnableCustomAutogradFunction from json file onto ORTModule"""

    enable_custom_autograd_function_str = 'EnableCustomAutogradFunction'
    if not hasattr(data, enable_custom_autograd_function_str):
        return
    print(f"Found keyword {enable_custom_autograd_function_str} in json. Loading attributes from file.")

    assert isinstance(data.EnableCustomAutogradFunction, bool), f"{enable_custom_autograd_function_str} must be a boolean"
    ortmodule_config_accessor._enable_custom_autograd_function = data.EnableCustomAutogradFunction

def load_allow_layer_norm_mod_precision(ortmodule_config_accessor, data):
    """Load AllowLayerNormModPrecision from json file onto ORTModule"""

    allow_layer_norm_mod_precision_str = 'AllowLayerNormModPrecision'
    if not hasattr(data, allow_layer_norm_mod_precision_str):
        return
    print(f"Found keyword {allow_layer_norm_mod_precision_str} in json. Loading attributes from file.")

    assert isinstance(data.AllowLayerNormModPrecision, bool), f"{allow_layer_norm_mod_precision_str} must be a boolean"
    ortmodule_config_accessor._allow_layer_norm_mod_precision = data.AllowLayerNormModPrecision

def load_enable_grad_acc_optimization(ortmodule_config_accessor, data):
    """Load EnableGradAccOptimization from json file onto ORTModule"""

    enable_grad_acc_optimization_str = 'EnableGradAccOptimization'
    if not hasattr(data, enable_grad_acc_optimization_str):
        return
    print(f"Found keyword {enable_grad_acc_optimization_str} in json. Loading attributes from file.")

    assert isinstance(data.EnableGradAccOptimization, bool), f"{enable_grad_acc_optimization_str} must be a boolean"
    ortmodule_config_accessor._enable_grad_acc_optimization = data.EnableGradAccOptimization

def load_run_symbolic_shape_infer(ortmodule_config_accessor, data):
    """Load RunSymbolicShapeInference from json file onto ORTModule"""

    run_symbolic_shape_inference_str = 'RunSymbolicShapeInference'
    if not hasattr(data, run_symbolic_shape_inference_str):
        return
    print(f"Found keyword {run_symbolic_shape_inference_str} in json. Loading attributes from file.")

    assert isinstance(data.RunSymbolicShapeInference, bool), f"{run_symbolic_shape_inference_str} must be a boolean"
    ortmodule_config_accessor._run_symbolic_shape_infer = data.RunSymbolicShapeInference

def load_use_static_shape(ortmodule_config_accessor, data):
    """Load UseStaticShape from json file onto ORTModule"""

    use_state_shape_str = 'UseStaticShape'
    if not hasattr(data, use_state_shape_str):
        return
    print(f"Found keyword {use_state_shape_str} in json. Loading attributes from file.")

    assert isinstance(data.UseStaticShape, bool), f"{use_state_shape_str} must be a boolean"
    ortmodule_config_accessor._use_static_shape = data.UseStaticShape

def load_skip_check(ortmodule_config_accessor, data):
    """Load SkipCheck from json file onto ORTModule"""

    skip_check_str = 'SkipCheck'
    if not hasattr(data, skip_check_str):
        return
    print(f"Found keyword {skip_check_str} in json. Loading attributes from file.")

    skip_check = reduce(lambda x, y: x|y, [_SkipCheck[name] for name in data.SkipCheck])
    if skip_check.value > 0:
        ortmodule_config_accessor._skip_check = skip_check

def load_debug_options(ortmodule_config_accessor, data):
    """Load DebugOptions from json file onto ORTModule"""

    debug_options_str = 'DebugOptions'
    if not hasattr(data, debug_options_str):
        return
    print(f"Found keyword {debug_options_str} in json. Loading attributes from file.")


    log_level = LogLevel.WARNING
    if hasattr(data.DebugOptions, 'LogLevel'):
        log_level = LogLevel[data.DebugOptions.LogLevel]

    save_onnx = False
    onnx_prefix = ''
    if hasattr(data.DebugOptions, 'SaveONNX'):
        save_onnx = data.DebugOptions.SaveONNX
        onnx_prefix = data.DebugOptions.ONNXPrefix

    if hasattr(data.DebugOptions, 'SaveONNXPath'):
        os.environ["ORTMODULE_SAVE_ONNX_PATH"] = data.DebugOptions.SaveONNXPath

    debug_options = DebugOptions(log_level=log_level, save_onnx=save_onnx, onnx_prefix=onnx_prefix)
    ortmodule_config_accessor._debug_options = debug_options
