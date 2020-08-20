import onnx
from  .base_operator import QuantOperatorBase
from ..quant_utils import _find_by_name, _attribute_to_kwarg, ms_domain
from onnx import onnx_pb as onnx_proto

class QLinearActivation(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
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