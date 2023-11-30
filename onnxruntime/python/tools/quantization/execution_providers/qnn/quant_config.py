# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging
from pathlib import Path

import onnx

from ...calibrate import CalibrationDataReader, CalibrationMethod
from ...onnx_model import ONNXModel
from ...quant_utils import QuantType
from ...quantize import StaticQuantConfig
from .fusion_gelu import FusionGelu
from .fusion_layernorm import FusionLayerNormalization
from .fusion_lpnorm import FusionLpNormalization

Q16_TYPES = {QuantType.QInt16, QuantType.QUInt16}
Q8_TYPES = {QuantType.QInt8, QuantType.QUInt8}
OP_TYPES_TO_EXCLUDE = {"Cast"}


def qnn_preprocess_model(model_input: Path, model_output: Path, fuse_layernorm: bool = False) -> bool:
    modified = False
    model = onnx.load_model(model_input)
    onnx_model = ONNXModel(model)

    # Fuse Erf sequence into a single Gelu
    fusion_gelu = FusionGelu(onnx_model)
    if fusion_gelu.apply():
        modified = True

    # Fuse ReduceL2 sequence into a single LpNormalization node with p == 2.
    fusion_lpnorm = FusionLpNormalization(onnx_model)
    if fusion_lpnorm.apply():
        modified = True

    # Optionally, fuse ReduceMean sequence into a single LayerNormalization node.
    if fuse_layernorm:
        onnx_opset = next(x for x in model.opset_import if x.domain == "" or x.domain == "ai.onnx")

        # Need opset >= 17 to use LayerNormalization.
        if onnx_opset.version < 17:
            logging.warning(
                "Unable to fuse ReduceMean sequence into a LayerNormalization node. "
                "ONNX model must use an opset >= 17 in order to use LayerNormalization, "
                f"but found version {onnx_opset.version}. Please use onnx.version_converter to update your model."
            )
        else:
            fusion_layernorm = FusionLayerNormalization(onnx_model)
            if fusion_layernorm.apply():
                modified = True

    if modified:
        onnx_model.topological_sort()
        onnx.save_model(model, model_output)

    return modified


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

    # Process model nodes to setup overrides.
    model = onnx.load_model(model_input)

    op_types = set()
    tensor_quant_overrides = {}

    name_to_initializer = {initializer.name: initializer for initializer in model.graph.initializer}

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
                tensor_quant_overrides[node.output[0]] = [{"scale": 1.0 / 65536.0, "zero_point": 0}]
            elif activation_type == QuantType.QInt16:
                tensor_quant_overrides[node.output[0]] = [{"scale": 1.0 / 32768.0, "zero_point": 0}]
        elif node.op_type == "Tanh":
            if activation_type == QuantType.QUInt16:
                tensor_quant_overrides[node.output[0]] = [{"scale": 1.0 / 32768.0, "zero_point": 32768}]
            elif activation_type == QuantType.QInt16:
                tensor_quant_overrides[node.output[0]] = [{"scale": 1.0 / 32768.0, "zero_point": 0}]

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
        extra_options=extra_options,
    )
