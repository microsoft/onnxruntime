# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from pathlib import Path

import onnx

from ...calibrate import CalibrationDataReader, CalibrationMethod
from ...quant_utils import QuantType
from ...quantize import StaticQuantConfig


def get_qnn_qdq_config(
    model_input: Path,
    calibration_data_reader: CalibrationDataReader,
    calibrate_method=CalibrationMethod.MinMax,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    op_types_to_quantize=None
):
    # Parse model nodes to setup overrides.
    model = onnx.load_model(model_input)

    tensor_quant_overrides = {}

    for node in model.graph.node:
        if node.op_type == "MatMul" and activation_type == QuantType.QUInt16:
            tensor_quant_overrides[node.input[1]] = {"quant_type": QuantType.QUInt8}
        elif node.op_type == "Sigmoid":
            if activation_type == QuantType.QUInt16:
                tensor_quant_overrides[node.output[0]] = {"scale": 1.0 / 65536.0, "zero_point": 0}
            elif activation_type == QuantType.QInt16:
                tensor_quant_overrides[node.output[0]] = {"scale": 1.0 / 32768.0, "zero_point": 0}
        elif node.op_type == "Tanh":
            if activation_type == QuantType.QUInt16:
                tensor_quant_overrides[node.output[0]] = {"scale": 1.0 / 32768.0, "zero_point": 32768}
            elif activation_type == QuantType.QInt16:
                tensor_quant_overrides[node.output[0]] = {"scale": 1.0 / 32768.0, "zero_point": 0}

    return StaticQuantConfig(
        calibration_data_reader,
        calibrate_method=calibrate_method,
        activation_type=activation_type,
        weight_type=weight_type,
        # TODO: Get these from the model itself (and as arg to this function)
        op_types_to_quantize=op_types_to_quantize,
        extra_options={
            "MinimumRealRange": 0.0001,
            "DedicatedQDQPair": True,
            "TensorQuantOverrides": tensor_quant_overrides,
        },
    )
