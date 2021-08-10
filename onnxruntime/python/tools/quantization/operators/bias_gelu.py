import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType
from ..quant_utils import attribute_to_kwarg, ms_domain
from onnx import onnx_pb as onnx_proto
'''
    Quantize BiasGelu
'''


class BiasGeluQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "BiasGelu")

        qbias_gelu_name = node.name + "_quant" if node.name != "" else ""

        '''
        Pre-quantization BiasGelu inputs:
        [0] Input
        [1] Bias
        '''
        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [0, 1])
        '''
        TODO(kreeger): what about input scale/zp - is this something that can be
                       passed down from the matmul input?
        Quantized BiasGelu Inputs:
        [0] Input
        [1] Bias
        [2] Bias Scale
        [3] Bias ZeroPoint
        '''
        inputs = []
        inputs.extend(quantized_input_names)
        inputs.extend(scale_names)
        inputs.extend(zero_point_names)

        bias_gelu_new_output = node.output[0] + "_quantized"

        # Create an output entry for the quantized value:
        q_output = QuantizedValue(node.output[0], bias_gelu_new_output, scale_names[0], zero_point_names[0],
                                  QuantizedValueType.Input)
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qbias_gelu_node = onnx.helper.make_node("QBiasGelu", inputs, [bias_gelu_new_output], qbias_gelu_name, **kwargs)
        nodes.append(qbias_gelu_node)

        self.quantizer.new_nodes += nodes
