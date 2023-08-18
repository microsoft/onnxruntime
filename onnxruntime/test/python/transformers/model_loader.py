# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os

from onnx import ModelProto, TensorProto, external_data_helper, load_model, numpy_helper
from parity_utilities import find_transformers_source

if find_transformers_source():
    from fusion_utils import NumpyHelper
else:
    from onnxruntime.transformers.fusion_utils import NumpyHelper


def fill_zeros_for_external_data(tensor: TensorProto):
    if tensor.HasField("raw_data"):  # already loaded
        return

    value = NumpyHelper.to_array(tensor, fill_zeros=True)
    zero_tensor = numpy_helper.from_array(value, name=tensor.name)
    tensor.raw_data = zero_tensor.raw_data


def fill_zeros_for_external_data_of_model(model: ModelProto):
    """Fill zeros value for external tensors of model

    Args:
        model (ModelProto): model to set external data
        base_dir (str): directory that contains external data
    """
    # TODO: support attribute tensor, which is rare in transformers model.
    for tensor in model.graph.initializer:
        if external_data_helper.uses_external_data(tensor):
            fill_zeros_for_external_data(tensor)
            # Change the state of tensors and remove external data
            tensor.data_location = TensorProto.DEFAULT
            del tensor.external_data[:]


def load_model_with_dummy_external_data(path: str) -> ModelProto:
    """Load model and fill zeros for initializers in external data.
       It helps in testing graph fusion of onnx model without external data.

    Args:
        path (str): onnx model path

    Returns:
        ModelProto: model with external data tensors filled with zero.
    """
    model = load_model(path, load_external_data=False)
    fill_zeros_for_external_data_of_model(model)
    return model


def get_test_data_path(sub_dir: str, file: str):
    relative_path = os.path.join(os.path.dirname(__file__), "test_data", sub_dir, file)
    if os.path.exists(relative_path):
        return relative_path
    return os.path.join(".", "transformers", "test_data", sub_dir, file)


def get_fusion_test_model(file: str):
    relative_path = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "transform", "fusion", file)
    if os.path.exists(relative_path):
        return relative_path
    return os.path.join(".", "testdata", "transform", "fusion", file)
