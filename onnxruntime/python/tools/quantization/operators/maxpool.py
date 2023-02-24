import onnx

from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg, ms_domain
from .base_operator import QuantOperatorBase
from .direct_q8 import Direct8BitOp, QDQDirect8BitOp


class QMaxPool(Direct8BitOp):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)
        self.qa = QuantOperatorBase(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "MaxPool"
        print(f"Custom quantization code for {node.op_type}")

        nodes = []

        if node.input[0] not in self.quantizer.quantized_value_map:
            (
                quantized_input_names,
                zero_point_names,
                scale_names,
                nodes,
            ) = self.qa.quantizer.quantize_activation(node, [0])

        # Create an entry for output quantized value.
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]
        (
            data_found,
            output_scale_name_from_parameter,
            output_zp_name_from_parameter,
            _,
            _,
        ) = self.quantizer._get_quantization_params(node.output[0])

        # Just use input scale and zp if parameters for output is not specified.
        output_scale_name = output_scale_name_from_parameter if data_found else quantized_input_value.scale_name
        output_zp_name = output_zp_name_from_parameter if data_found else quantized_input_value.zp_name
        quantized_output_value = QuantizedValue(
            node.output[0],
            node.output[0] + "_q",
            output_scale_name,
            output_zp_name,
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        qnode_name = node.name + "_quant" if node.name != "" else ""

        qnode = onnx.helper.make_node(
            "MaxPool",
            [
                quantized_input_value.q_name,
            ],
            [quantized_output_value.q_name],
            qnode_name,
            **kwargs,
        )

        nodes.append(qnode)
        self.quantizer.new_nodes += nodes


class QDQMaxPool(QDQDirect8BitOp):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "MaxPool"

        # if version is less than 12, just no change
        if self.quantizer.opset_version < 12:
            return

        # Direct 8bits op
        return super().quantize()
