# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import onnx

from ...calibrate import CalibrationDataReader, CalibrationMethod
from ...quant_utils import QuantType
from ...quantize import StaticQuantConfig

Q16_TYPES = {QuantType.QInt16, QuantType.QUInt16}
Q8_TYPES = {QuantType.QInt8, QuantType.QUInt8}
OP_TYPES_TO_EXCLUDE = {"Cast"}
MODEL_SIZE_THRESHOLD = 2147483648  # Quant model should use external data if >= 2GB


def get_qnn_qdq_config(
    model_input: Path,
    calibration_data_reader: CalibrationDataReader,
    calibrate_method=CalibrationMethod.MinMax,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    per_channel=False,
):
    if per_channel:
        raise ValueError("QNN EP does not yet support per-channel quantization.")

    model = onnx.load_model(model_input, load_external_data=False)

    op_types = set()
    tensor_quant_overrides = {}
    model_has_external_data = False
    name_to_initializer = {}

    # Build map of initializers (name -> initializer) and
    # check if the model has external data.
    for initializer in model.graph.initializer:
        name_to_initializer[initializer.name] = initializer
        if onnx.external_data_helper.uses_external_data(initializer):
            model_has_external_data = True

    # Setup quantization overrides for specific operator types
    for node in model.graph.node:
        op_types.add(node.op_type)

        if node.op_type == "MatMul" and activation_type in Q16_TYPES and weight_type in Q8_TYPES:
            weight_symmetric = weight_type == QuantType.QInt8

            # Override initializers to use the weight_type
            for input_name in node.input:
                if input_name in name_to_initializer:
                    tensor_quant_overrides[input_name] = [{"quant_type": weight_type, "symmetric": weight_symmetric}]
        elif node.op_type == "LayerNormalization" and activation_type in Q16_TYPES and weight_type in Q8_TYPES:
            weight_symmetric = weight_type == QuantType.QInt8

            # Override initializers to use the weight_type. Don't override the bias input.
            for i in range(2):
                input_name = node.input[i]
                if input_name in name_to_initializer:
                    tensor_quant_overrides[input_name] = [{"quant_type": weight_type, "symmetric": weight_symmetric}]
        elif node.op_type == "Sigmoid":
            if activation_type == QuantType.QUInt16:
                tensor_quant_overrides[node.output[0]] = [
                    {"scale": np.array(1.0 / 65536.0, dtype=np.float32), "zero_point": np.array(0, dtype=np.uint16)}
                ]
            elif activation_type == QuantType.QInt16:
                tensor_quant_overrides[node.output[0]] = [
                    {"scale": np.array(1.0 / 32768.0, dtype=np.float32), "zero_point": np.array(0, dtype=np.int16)}
                ]
        elif node.op_type == "Tanh":
            if activation_type == QuantType.QUInt16:
                tensor_quant_overrides[node.output[0]] = [
                    {"scale": np.array(1.0 / 32768.0, dtype=np.float32), "zero_point": np.array(32768, dtype=np.uint16)}
                ]
            elif activation_type == QuantType.QInt16:
                tensor_quant_overrides[node.output[0]] = [
                    {"scale": np.array(1.0 / 32768.0, dtype=np.float32), "zero_point": np.array(0, dtype=np.int16)}
                ]

    extra_options = {
        "MinimumRealRange": 0.0001,
        "DedicatedQDQPair": False,  # Let ORT optimizer duplicate DQ nodes
        "TensorQuantOverrides": tensor_quant_overrides,
    }

    # TODO: Remove this extra option once ORT uses an ONNX version that supports 16-bit Q/DQ ops.
    if activation_type in Q16_TYPES or weight_type in Q16_TYPES:
        extra_options["UseQDQContribOps"] = True

    return StaticQuantConfig(
        calibration_data_reader,
        calibrate_method=calibrate_method,
        activation_type=activation_type,
        weight_type=weight_type,
        op_types_to_quantize=list(op_types.difference(OP_TYPES_TO_EXCLUDE)),
        use_external_data_format=(model_has_external_data or model.ByteSize() >= MODEL_SIZE_THRESHOLD),
        extra_options=extra_options,
    )
