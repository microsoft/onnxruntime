# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# debug.py

import os
from . import _load_config_from_json as loading_functions

def load_from_json(ortmodule, path=None):
    """Load config from json file at given path.

    Here is the schema that the json file must adhere to:
    {
        "PropagateCastOps":
        {
            "Strategy": "FLOOD_FILL", # str representing strategy (like "NONE", "FLOOD_FILL"...)
            "Level": 3, # propagate cast ops level as an int
            "Allow": ["ABC", "DEF"] # propagate cast ops allow as list of strs
        },
        "UseExternalGPUAllocator" : false, # bool flag
        "EnableCustomAutogradFunction": true, # bool flag
        "AllowLayerNormModPrecision": true, # bool flag
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
        }
    }

    Args:
        ortmodule (:obj:`ORTModule`): ORTModule instance that needs to be configured
        path (:obj:`str`, optional): Path to json file. Alternatively, users can set the
            environement variable ORTMODULE_JSON_CONFIG_PATH to the json config path. In case
            both path and environment variable are set, the environment variable gets precedence.
    """

    json_path_environment_key = "ORTMODULE_JSON_CONFIG_PATH"
    path = os.getenv(json_path_environment_key, path)

    # figure out the json path
    if path is None:
        raise ValueError(f"Path to json is not provided. Provide the path through function call or by setting the environment variable {json_path_environment_key}")

    # load the entire json file
    data = loading_functions.load_data_from_json(path)

    # define all load functions to iterate over
    load_functions = [
        loading_functions.load_propagate_cast_ops,
        loading_functions.load_use_external_gpu_allocator,
        loading_functions.load_enable_custom_autograd_function,
        loading_functions.load_allow_layer_norm_mod_precision,
        loading_functions.load_enable_grad_acc_optimization,
        loading_functions.load_run_symbolic_shape_infer,
        loading_functions.load_use_static_shape,
        loading_functions.load_skip_check,
        loading_functions.load_debug_options
    ]

    for training_mode in [True, False]:
        # update the debug config for both train and eval modes
        ortmodule_config_accessor = ortmodule._torch_module._execution_manager(training_mode)
        for load_function in load_functions:
            load_function(ortmodule_config_accessor, data)
