# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import onnx

from ...calibrate import CalibrationDataReader, CalibrationMethod
from ...quant_utils import QuantType
from ...quantize import StaticQuantConfig
from ...tensor_quant_overrides import TensorQuantOverridesHelper
from .mixed_precision_overrides_utils import MixedPrecisionTensorQuantOverridesFixer

Q16_TYPES = {QuantType.QInt16, QuantType.QUInt16}
Q8_TYPES = {QuantType.QInt8, QuantType.QUInt8}
OP_TYPES_TO_EXCLUDE = {"Cast"}
MODEL_SIZE_THRESHOLD = 2147483648  # Quant model should use external data if >= 2GB


def warn_unable_to_override(
    node: onnx.NodeProto,
    what_str: str,
    tensor_name: str,
    io_kind: str,
):
    logging.warning(
        f"Unable to override {what_str} for {node.op_type} node's {io_kind} "
        "because it has already been overridden! Check the initial quantization overrides provided "
        "to get_qnn_qdq_config() if the generated QDQ model does not run on QNN EP. "
        f"Node name: {node.name}, {io_kind} name: {tensor_name}"
    )


def get_qnn_qdq_config(
    model_input: Path,
    calibration_data_reader: CalibrationDataReader,
    calibrate_method=CalibrationMethod.MinMax,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    per_channel=False,
    init_overrides=None,
    add_qtype_converts=True,
    activation_symmetric=False,
    weight_symmetric=None,
):
    if per_channel:
        raise ValueError("QNN EP does not yet support per-channel quantization.")

    if weight_symmetric is None:
        weight_symmetric = weight_type in {QuantType.QInt8, QuantType.QInt16}

    model = onnx.load_model(model_input, load_external_data=False)

    op_types = set()
    model_has_external_data = False
    name_to_initializer = {}

    # Build map of initializers (name -> initializer) and
    # check if the model has external data.
    for initializer in model.graph.initializer:
        name_to_initializer[initializer.name] = initializer
        if onnx.external_data_helper.uses_external_data(initializer):
            model_has_external_data = True

    overrides_helper = TensorQuantOverridesHelper(copy.deepcopy(init_overrides) if init_overrides else {})

    if not overrides_helper.empty() and add_qtype_converts:
        overrides_fixer = MixedPrecisionTensorQuantOverridesFixer.create_from_model(
            overrides_helper, model, activation_type
        )
        overrides_fixer.apply(activation_type, activation_symmetric)

    # Setup quantization overrides for specific operator types
    for node in model.graph.node:
        if node.op_type == "MatMul" and weight_type in Q8_TYPES:
            input_16bit_act = None
            input_wgt = None

            for input_name in node.input:
                if input_name not in name_to_initializer:
                    qtype = overrides_helper.get_node_input_qtype_info(
                        input_name, node.name, activation_type
                    ).quant_type
                    if qtype in Q16_TYPES:
                        input_16bit_act = input_name
                else:
                    input_wgt = input_name

            # Override initializer to use the weight_type
            if input_16bit_act and input_wgt:
                did_update = overrides_helper.update_tensor_overrides(
                    input_wgt,
                    {"quant_type": weight_type, "symmetric": weight_symmetric},
                    overwrite=False,
                )

                if not did_update:
                    warn_unable_to_override(node, "quant_type/symmetric", input_wgt, "input weight")
        elif node.op_type == "LayerNormalization" and weight_type in Q8_TYPES:
            has_q16_activation = False
            for input_name in node.input:
                if input_name not in name_to_initializer:
                    qtype = overrides_helper.get_node_input_qtype_info(
                        input_name, node.name, activation_type
                    ).quant_type
                    if qtype in Q16_TYPES:
                        has_q16_activation = True
                        break

            # Override initializers to use the weight_type. Don't override the bias input.
            if has_q16_activation:
                for i in range(2):
                    input_name = node.input[i]
                    if input_name in name_to_initializer:
                        did_update = overrides_helper.update_tensor_overrides(
                            input_name,
                            {"quant_type": weight_type, "symmetric": weight_symmetric},
                            overwrite=False,
                        )

                        if not did_update:
                            warn_unable_to_override(node, "quant_type/symmetric", input_name, "input weight")

        elif node.op_type == "Sigmoid":
            output_type = overrides_helper.get_node_output_qtype_info(node.output[0], activation_type).quant_type

            if output_type == QuantType.QUInt16:
                did_update = overrides_helper.update_tensor_overrides(
                    node.output[0],
                    {
                        "quant_type": output_type,
                        "scale": np.array(1.0 / 65536.0, dtype=np.float32),
                        "zero_point": np.array(0, dtype=np.uint16),
                    },
                    overwrite=False,
                )
                if not did_update:
                    warn_unable_to_override(node, "quant_type/scale/zero_point", node.output[0], "output")
            elif output_type == QuantType.QInt16:
                did_update = overrides_helper.update_tensor_overrides(
                    node.output[0],
                    {
                        "quant_type": output_type,
                        "scale": np.array(1.0 / 32768.0, dtype=np.float32),
                        "zero_point": np.array(0, dtype=np.int16),
                    },
                    overwrite=False,
                )
                if not did_update:
                    warn_unable_to_override(node, "quant_type/scale/zero_point", node.output[0], "output")
        elif node.op_type == "Tanh":
            output_type = overrides_helper.get_node_output_qtype_info(node.output[0]).quant_type

            if output_type == QuantType.QUInt16:
                did_update = overrides_helper.update_tensor_overrides(
                    node.output[0],
                    {
                        "quant_type": output_type,
                        "scale": np.array(1.0 / 32768.0, dtype=np.float32),
                        "zero_point": np.array(32768, dtype=np.uint16),
                    },
                    overwrite=False,
                )
                if not did_update:
                    warn_unable_to_override(node, "quant_type/scale/zero_point", node.output[0], "output")
            elif output_type == QuantType.QInt16:
                did_update = overrides_helper.update_tensor_overrides(
                    node.output[0],
                    {
                        "quant_type": output_type,
                        "scale": np.array(1.0 / 32768.0, dtype=np.float32),
                        "zero_point": np.array(0, dtype=np.int16),
                    },
                    overwrite=False,
                )
                if not did_update:
                    warn_unable_to_override(node, "quant_type/scale/zero_point", node.output[0], "output")

    extra_options = {
        "MinimumRealRange": 0.0001,
        "DedicatedQDQPair": False,  # Let ORT optimizer duplicate DQ nodes
        "TensorQuantOverrides": overrides_helper.get_dict(),
        "ActivationSymmetric": activation_symmetric,
        "WeightSymmetric": weight_symmetric,
    }

    # TODO: Remove this extra option once ORT uses an ONNX version that supports 16-bit Q/DQ ops.
    overrides_have_int16 = any(t in Q16_TYPES for t in overrides_helper.get_quant_types())
    if activation_type in Q16_TYPES or weight_type in Q16_TYPES or overrides_have_int16:
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
