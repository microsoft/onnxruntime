import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import attribute_to_kwarg, ms_domain
from onnx import onnx_pb as onnx_proto
'''
    Quantize LSTM
'''


class LSTMQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        '''
            parameter node: LSTM node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized Attention node.
        '''
        node = self.node
        assert (node.op_type == "LSTM")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [1, 2])

        quant_lstm_name = "" if node.name == "" else node.name + "_quant"

        inputs = []
        input_len = len(node.input)
        inputs.extend([node.input[0]])
        inputs.extend(quantized_input_names)
        inputs.extend([node.input[3]if input_len > 3 else ""])
        inputs.extend([node.input[4]if input_len > 4 else ""])
        inputs.extend([node.input[5]if input_len > 5 else ""])
        inputs.extend([node.input[6]if input_len > 6 else ""])
        inputs.extend([node.input[7]if input_len > 7 else ""])
        inputs.extend([scale_names[0], zero_point_names[0], scale_names[1], zero_point_names[1]])

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        quant_lstm_node = onnx.helper.make_node("DynamicQuantizeLSTM", inputs, node.output, quant_lstm_name, **kwargs)
        nodes.append(quant_lstm_node)

        self.quantizer.new_nodes += nodes
