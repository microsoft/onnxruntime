import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import _find_by_name, _attribute_to_kwarg, ms_domain
from onnx import onnx_pb as onnx_proto

'''
Quantize EmbedLayerNormalization
'''
class EmbedLayerNormalizationQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "EmbedLayerNormalization")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer._quantize_inputs(node, [2, 3, 4])

        nodes.append(node)

        self.quantizer.new_nodes +=nodes