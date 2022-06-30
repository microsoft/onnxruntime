import logging

import numpy as np
import onnx
from onnx import onnx_pb as onnx_proto

from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg, find_by_name, get_mul_node, ms_domain
from .base_operator import QuantOperatorBase
from .matmul import QOpMatMul
from .qdq_base_operator import QDQOperatorBase


def is_B_transposed(gemm_node):
    transB_attribute = [attr for attr in gemm_node.attribute if attr.name == "transB"]
    if len(transB_attribute):
        return 0 < onnx.helper.get_attribute_value(transB_attribute[0])

    return False


def get_beta(gemm_node):
    beta_attribute = [attr for attr in gemm_node.attribute if attr.name == "beta"]
    if len(beta_attribute):
        return onnx.helper.get_attribute_value(beta_attribute[0])

    return 1.0


def set_default_beta(gemm_node):
    beta_attribute = [attr for attr in gemm_node.attribute if attr.name == "beta"]
    if len(beta_attribute):
        beta_attribute[0].f = 1.0

    return 1.0


class QLinearGemm(QOpMatMul):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Gemm"

        (
            data_found,
            output_scale_name,
            output_zp_name,
            _,
            _,
        ) = self.quantizer._get_quantization_params(node.output[0])

        if self.quantizer.is_input_a_weight(node.input[1]) and self.quantizer.is_per_channel():
            (
                quantized_input_names,
                zero_point_names,
                scale_names,
                nodes,
            ) = self.quantizer.quantize_inputs(node, [0], reduce_range=self.quantizer.reduce_range)
            quant_weight_tuple = self.quantizer.quantize_weight_per_channel(
                node.input[1],
                onnx_proto.TensorProto.INT8,
                0 if is_B_transposed(node) else 1,
            )
            quantized_input_names.append(quant_weight_tuple[0])
            zero_point_names.append(quant_weight_tuple[1])
            scale_names.append(quant_weight_tuple[2])
        else:
            (
                quantized_input_names,
                zero_point_names,
                scale_names,
                nodes,
            ) = self.quantizer.quantize_inputs(node, [0, 1], reduce_range=self.quantizer.reduce_range)

        if not data_found or quantized_input_names is None:
            return super().quantize()

        quantized_bias_name = ""
        if len(node.input) == 3:
            if not self.quantizer.is_input_a_weight(node.input[2]):
                return super().quantize()

            quantized_bias_name = self.quantizer.quantize_bias_static(
                node.input[2], node.input[0], node.input[1], get_beta(self.node)
            )

        qgemm_output = node.output[0] + "_quantized"
        qgemm_name = qgemm_name = node.name + "_quant" if node.name != "" else ""

        kwargs = {}
        for attribute in node.attribute:
            if attribute.name != "beta":
                kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        # generate input
        qgemm_inputs = []
        for i in range(2):
            qgemm_inputs.extend([quantized_input_names[i], scale_names[i], zero_point_names[i]])

        qgemm_inputs.extend([quantized_bias_name, output_scale_name, output_zp_name])

        qgemm_node = onnx.helper.make_node("QGemm", qgemm_inputs, [qgemm_output], qgemm_name, **kwargs)
        nodes.append(qgemm_node)

        # Create an entry for this quantized value
        q_output = QuantizedValue(
            node.output[0],
            qgemm_output,
            output_scale_name,
            output_zp_name,
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        self.quantizer.new_nodes += nodes


class QDQGemm(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Gemm"

        self.quantizer.quantize_tensor(node.input[0])
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_tensor(node.output[0])

        if self.quantizer.is_per_channel():
            self.quantizer.quantize_tensor_per_channel(node.input[1], 0 if is_B_transposed(node) else 1)
        else:
            self.quantizer.quantize_tensor(node.input[1])

        if len(node.input) == 3:
            if self.quantizer.is_input_a_weight(node.input[2]):
                self.quantizer.quantize_bias_tensor(node.input[2], node.input[0], node.input[1], get_beta(self.node))
                set_default_beta(self.node)
            else:
                logging.warning(
                    "Bias of Gemm node '{}' is not constant. Please exclude this node for better performance.".format(
                        self.node.name
                    )
                )
