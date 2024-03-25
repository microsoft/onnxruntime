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
    model_input: str | Path | onnx.ModelProto,
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

    model = (
        model_input
        if isinstance(model_input, onnx.ModelProto)
        else onnx.load_model(model_input, load_external_data=False)
    )

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
        # Fix mixed-precision overrides.
        overrides_fixer = MixedPrecisionTensorQuantOverridesFixer.create_from_model(
            overrides_helper, model, activation_type
        )
        overrides_fixer.apply(activation_type, activation_symmetric)

    # Setup quantization overrides for specific operator types to ensure compatibility with QNN EP.
    qnn_compat = QnnCompatibilityOverrides(
        activation_type,
        weight_type,
        activation_symmetric,
        weight_symmetric,
        overrides_helper,
        name_to_initializer,
    )

    for node in model.graph.node:
        op_types.add(node.op_type)
        qnn_compat.process_node(node)

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


class QnnCompatibilityOverrides:
    """
    Helper that processes nodes to generate quantization overrides that make the resulting QDQ model
    compatible with QNN EP.
    """

    def __init__(
        self,
        default_activation_qtype: QuantType,
        default_weight_qtype: QuantType,
        activation_symmetric: bool,
        weight_symmetric: bool,
        overrides: TensorQuantOverridesHelper,
        initializers: dict[str, onnx.TensorProto],
    ):
        self.default_activation_qtype = default_activation_qtype
        self.default_weight_qtype = default_weight_qtype
        self.activation_symmetric = activation_symmetric
        self.weight_symmetric = weight_symmetric
        self.overrides = overrides
        self.initializers = initializers

        self.process_fns = {
            "MatMul": self._process_matmul,
            "LayerNormalization": self._process_layernorm,
            "Sigmoid": self._process_sigmoid,
            "Tanh": self._process_tanh,
        }

    def process_node(self, node: onnx.NodeProto):
        process_fn = self.process_fns.get(node.op_type)

        if process_fn is not None:
            process_fn(node)

    def _process_matmul(self, node: onnx.NodeProto):
        """
        Overrides MatMul's initializer input(s) to use the default weight type if:
        - The default weight type is 8-bit
        - One of the inputs is a 16-bit activation
        """
        assert node.op_type == "MatMul", f"Expected MatMul, but got {node.op_type}"
        if self.default_weight_qtype not in Q8_TYPES:
            return

        input_16bit_act = None
        input_wgt = None

        for input_name in node.input:
            if input_name and input_name not in self.initializers:
                qtype = self.overrides.get_node_input_qtype_info(
                    input_name, node.name, self.default_activation_qtype
                ).quant_type
                if qtype in Q16_TYPES:
                    input_16bit_act = input_name
            else:
                input_wgt = input_name

        # Override initializer to use the default weight type.
        if input_16bit_act and input_wgt:
            did_update = self.overrides.update_tensor_overrides(
                input_wgt,
                {"quant_type": self.default_weight_qtype, "symmetric": self.weight_symmetric},
                overwrite=False,
            )

            if not did_update:
                warn_unable_to_override(node, "quant_type/symmetric", input_wgt, "input weight")

    def _process_layernorm(self, node: onnx.NodeProto):
        """
        Overrides LayerNormalization's initializer input(s), except for bias, to use the default weight type if:
        - The default weight type is 8-bit
        - One of the inputs is a 16-bit activation
        """
        assert node.op_type == "LayerNormalization", f"Expected LayerNormalization, but got {node.op_type}"
        if self.default_weight_qtype not in Q8_TYPES:
            return

        has_q16_activation = False
        for input_name in node.input:
            if input_name and input_name not in self.initializers:
                qtype = self.overrides.get_node_input_qtype_info(
                    input_name, node.name, self.default_activation_qtype
                ).quant_type
                if qtype in Q16_TYPES:
                    has_q16_activation = True
                    break

        # Override initializers to use the self.default_weight_qtype. Don't override the bias input.
        if has_q16_activation:
            for i in range(2):
                input_name = node.input[i]
                if input_name and input_name in self.initializers:
                    did_update = self.overrides.update_tensor_overrides(
                        input_name,
                        {"quant_type": self.default_weight_qtype, "symmetric": self.weight_symmetric},
                        overwrite=False,
                    )

                    if not did_update:
                        warn_unable_to_override(node, "quant_type/symmetric", input_name, "input weight")

    def _process_sigmoid(self, node: onnx.NodeProto):
        """
        Overrides 16-bit Sigmoid's output scale and zero-point as per QNN requirements.
        """
        assert node.op_type == "Sigmoid", f"Expected Sigmoid, but got {node.op_type}"
        output_type = self.overrides.get_node_output_qtype_info(
            node.output[0], self.default_activation_qtype
        ).quant_type

        if output_type == QuantType.QUInt16:
            self.overrides.update_tensor_overrides(
                node.output[0],
                {
                    "quant_type": output_type,
                    "scale": np.array(1.0 / 65536.0, dtype=np.float32),
                    "zero_point": np.array(0, dtype=np.uint16),
                },
            )
        elif output_type == QuantType.QInt16:
            self.overrides.update_tensor_overrides(
                node.output[0],
                {
                    "quant_type": output_type,
                    "scale": np.array(1.0 / 32768.0, dtype=np.float32),
                    "zero_point": np.array(0, dtype=np.int16),
                },
            )

    def _process_tanh(self, node: onnx.NodeProto):
        """
        Overrides 16-bit Tanh's output scale and zero-point as per QNN requirements.
        """
        assert node.op_type == "Tanh", f"Expected Tanh, but got {node.op_type}"
        output_type = self.overrides.get_node_output_qtype_info(
            node.output[0], self.default_activation_qtype
        ).quant_type

        if output_type == QuantType.QUInt16:
            self.overrides.update_tensor_overrides(
                node.output[0],
                {
                    "quant_type": output_type,
                    "scale": np.array(1.0 / 32768.0, dtype=np.float32),
                    "zero_point": np.array(32768, dtype=np.uint16),
                },
            )
        elif output_type == QuantType.QInt16:
            self.overrides.update_tensor_overrides(
                node.output[0],
                {
                    "quant_type": output_type,
                    "scale": np.array(1.0 / 32768.0, dtype=np.float32),
                    "zero_point": np.array(0, dtype=np.int16),
                },
            )
