import onnx
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg, ms_domain
from onnx import onnx_pb as onnx_proto


class QLinearActivation(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def QuantizeClipRelu(self):
        node = self.node
        assert (node.op_type == "Relu" or node.op_type == 'Clip')

        # When mode is QLinearOps, the output quantization params are calculated based on outputs from
        # activation nodes, therefore these nodes can be removed from the graph if they follow a quantized op.
        # If input to this node is not quantized then keep this node
        if node.input[0] not in self.quantizer.quantized_value_map:
            self.quantizer.new_nodes += [node]
            return

        quantized_value = self.quantizer.quantized_value_map[node.input[0]]
        self.quantizer.quantized_value_map[node.output[0]] = quantized_value

    def quantize(self):
        node = self.node
        if node.op_type == "Relu" or node.op_type == 'Clip':
            self.QuantizeClipRelu()
            return

        nnapi_sigmoid_option = 'extra.Sigmoid.nnapi'
        sigmoid_nnapi_mode = (node.op_type == 'Sigmoid' and
                              nnapi_sigmoid_option in self.quantizer.extra_options and
                              self.quantizer.extra_options[nnapi_sigmoid_option])
        use_scale = 1 / 256.0 if sigmoid_nnapi_mode else None
        use_zeropoint = 0 if sigmoid_nnapi_mode else None

        # No assert on op_type as it is controlled by registry
        # only try to quantize when given quantization parameters for it
        data_found, output_scale_name, output_zp_name, _, _ = \
            self.quantizer._get_quantization_params(node.output[0], use_scale, use_zeropoint)
        if not data_found:
            super().quantize()
            return

        quantized_input_names, zero_point_names, scale_names, nodes = self.quantizer.quantize_inputs(node, [0])

        qlinear_activation_output = node.output[0] + "_quantized"
        qlinear_activation_name = ""
        if node.name != "":
            qlinear_activation_name = node.name + "_quant"
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qlinear_activation_inputs = [
            quantized_input_names[0], scale_names[0], zero_point_names[0], output_scale_name, output_zp_name
        ]

        qlinear_activation_node = onnx.helper.make_node("QLinear" + node.op_type, qlinear_activation_inputs,
                                                        [qlinear_activation_output], qlinear_activation_name, **kwargs)

        # Create an entry for this quantized value
        q_output = QuantizedValue(node.output[0], qlinear_activation_output, output_scale_name, output_zp_name,
                                  QuantizedValueType.Input)
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        nodes.append(qlinear_activation_node)
        self.quantizer.new_nodes += nodes


class QDQRemovableActivation(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        if self.quantizer.try_replacing_upstream_output(node.input[0], node.output[0]):
            self.quantizer.remove_node(self.node)
