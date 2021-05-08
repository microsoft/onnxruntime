# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
ONNX Runtime is a performance-focused scoring engine for Open Neural Network Exchange (ONNX) models.
For more information on ONNX Runtime, please see `aka.ms/onnxruntime <https://aka.ms/onnxruntime/>`_
or the `Github project <https://github.com/microsoft/onnxruntime/>`_.
"""
__version__ = "1.7.0"
__author__ = "Microsoft"

from onnxruntime.capi._pybind_state import get_all_providers, get_available_providers, get_device, set_seed, \
    RunOptions, SessionOptions, set_default_logger_severity, enable_telemetry_events, disable_telemetry_events, \
    NodeArg, ModelMetadata, GraphOptimizationLevel, ExecutionMode, ExecutionOrder, OrtDevice, SessionIOBinding, \
    OrtAllocatorType, OrtMemType, OrtArenaCfg, OrtMemoryInfo, create_and_register_allocator

from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession, IOBinding, OrtValue
from onnxruntime.capi import onnxruntime_validation

from onnxruntime.capi.training import *  # noqa: F403

# TODO: thiagofc: Temporary experimental namespace for new PyTorch front-end
try:
    from . import experimental
except ImportError:
    pass

try:
    from onnxruntime.training.ortmodule import ORTModule
    has_ortmodule = True
except: # noqa
    has_ortmodule = False

package_name = ''
cuda_version = ''

if has_ortmodule:
    try:
        # collect onnxruntime package name, version, and cuda version
        from .build_and_package_info import package_name
        from .build_and_package_info import __version__

        cuda_version = None
        try:
            from .build_and_package_info import cuda_version
        except: # noqa
            pass

        print('onnxruntime training package info: package_name:', package_name)
        print('onnxruntime training package info: __version__:', __version__)

        if cuda_version:
            print('onnxruntime training package info: cuda_version:', cuda_version)

            # collect cuda library build info. the library info may not be available
            # when the build environment has none or multiple libraries installed
            try:
                from .build_and_package_info import cudart_version
                print('onnxruntime build info: cudart_version:', cudart_version)
            except: # noqa
                print('WARNING: failed to get cudart_version from onnxruntime build info.')
                cudart_version = None

            try:
                from .build_and_package_info import cudnn_version
                print('onnxruntime build info: cudnn_version:', cudnn_version)
            except: # noqa
                print('WARNING: failed to get cudnn_version from onnxruntime build info')
                cudnn_version = None

            # collection cuda library info from current environment.
            from onnxruntime.capi.onnxruntime_collect_build_info import find_cudart_versions, find_cudnn_versions
            local_cudart_versions = find_cudart_versions(build_env=False)
            if cudart_version and cudart_version not in local_cudart_versions:
                print('WARNING: failed to find cudart version that matches onnxruntime build info')
                print('WARNING: found cudart versions: ', local_cudart_versions)

            local_cudnn_versions = find_cudnn_versions(build_env=False)
            if cudnn_version and cudnn_version not in local_cudnn_versions:
                print('WARNING: failed to find cudnn version that matches onnxruntime build info')
                print('WARNING: found cudnn versions: ', local_cudnn_versions)
        else:
            # TODO: rcom
            pass

    except: # noqa
        print('WARNING: failed to collect onnxruntime version and build info')
        pass


onnxruntime_validation.check_distro_info()
