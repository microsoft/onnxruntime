# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

EP_NAME = "QNNExecutionProvider"


def get_ep_names():
    return [EP_NAME]


def get_ep_name():
    return EP_NAME


def get_library_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnxruntime_providers_qnn.dll")


def get_qnn_cpu_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "QnnCpu.dll")


def get_qnn_gpu_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "QnnGpu.dll")


def get_qnn_htp_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "QnnHtp.dll")
