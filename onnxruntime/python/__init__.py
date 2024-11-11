# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

if (
    __package__ == "onnxruntime-gpu"
    # incase we rename the package name in the future
    or __package__ == "onnxruntime-cuda"
):
    from .onnxruntime_cuda_temp_env import load_nvidia_libs

    load_nvidia_libs()
