# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from fusion_utils import NumpyHelper
from onnx import ModelProto
from onnx.external_data_helper import set_external_data

from onnxruntime import OrtValue


def extract_external_data_from_model(model: ModelProto):
    """
    Extract external data from model and return the external data as a list of tuples (name, value).

    Args:
        model (ModelProto): the model proto to extract external data from.
    Returns:
        (external_names, external_values): a tuple of two lists of external data names and values.
    """
    external_data = []
    for tensor in model.graph.initializer:
        name = tensor.name

        if tensor.HasField("raw_data"):
            numpy_tensor = NumpyHelper.to_array(tensor)
            ort_value = OrtValue.ortvalue_from_numpy(numpy_tensor)
            external_data.append((name, ort_value))
            # mimic set_external_data
            set_external_data(tensor, location="foo.bin")
            tensor.name = name
            tensor.ClearField("raw_data")

    return zip(*external_data)
