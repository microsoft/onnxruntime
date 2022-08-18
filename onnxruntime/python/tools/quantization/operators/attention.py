import onnx
from onnx import onnx_pb as onnx_proto

from ..quant_utils import attribute_to_kwarg, ms_domain
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase

"""
    Quantize Attention
"""


class AttentionQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def should_quantize(self):
        return self.quantizer.should_quantize_node(self.node)

    def quantize(self):
        """
        parameter node: Attention node.
        parameter new_nodes_list: List of new nodes created before processing this node.
        return: a list of nodes in topological order that represents quantized Attention node.
        """
        node = self.node
        assert node.op_type == "Attention"

        # TODO This is a temporary fix to stop exporting QAttention with qkv_hidden_sizes
        # attribute. This needs to be removed once the QAttention for varied q,k,v sizes
        # is implemented
        for attr in node.attribute:
            if "qkv_hidden_sizes" == attr.name:
                return super().quantize()

        (
            quantized_input_names,
            zero_point_names,
            scale_names,
            nodes,
        ) = self.quantizer.quantize_inputs(node, [0, 1], reduce_range=True, op_level_per_channel=True)
        if quantized_input_names is None:
            return super().quantize()

        qattention_name = "" if node.name == "" else node.name + "_quant"

        inputs = []
        inputs.extend(quantized_input_names)
        inputs.extend([node.input[2]])
        inputs.extend(scale_names)
        inputs.extend([node.input[3] if len(node.input) > 3 else ""])
        inputs.extend(zero_point_names)
        inputs.extend([node.input[4] if len(node.input) > 4 else ""])

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        qattention_node = onnx.helper.make_node("QAttention", inputs, node.output, qattention_name, **kwargs)
        nodes.append(qattention_node)

        self.quantizer.new_nodes += nodes


class QDQAttention(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Attention"

        self.quantizer.quantize_tensor(node.input[0])
        self.quantizer.quantize_tensor(node.input[1])
        # self.quantizer.tensors_to_quantize.append(node.input[0])
        # self.quantizer.tensors_to_quantize.append(node.input[1])
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_tensor(node.output[0])
            # self.quantizer.tensors_to_quantize.append(node.output[0])

        # TODO: Test disable output
        # TODO: How it works the is_per_channel option?

        # for tensor_name in nodes_to_iterate:
        #     # only support per-channel quantization on weight
        #     if self.quantizer.is_per_channel() and find_by_name(tensor_name, self.quantizer.model.initializer()):
        #         channel_axis = self.quantizer.qdq_op_type_per_channel_support_to_axis.get(node.op_type, 1)
        #         self.quantizer.quantize_tensor_per_channel(tensor_name, channel_axis)
        #     else:
        #         self.quantizer.quantize_tensor(tensor_name)
