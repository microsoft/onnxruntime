import itertools
from abc import ABC, abstractmethod

from onnx import onnx_pb as onnx_proto


class QuantOperatorBase(ABC):
    """
    Abstract Class for Quantized Operator format
    """
    def __init__(self, onnx_quantizer, onnx_node):
        self.quantizer = onnx_quantizer
        self.node = onnx_node

    def quantize(self):
        """
        Main method to quantize an operator.
        """
        if self.should_quantize():
            self.do_quantization()
        else:
            self.fixup_quantization()

    def should_quantize(self):
        node = self.node
        quantizer = self.quantizer
        if (
            quantizer.nodes_to_quantize is not None
            and len(quantizer.nodes_to_quantize) != 0
            and node.name not in quantizer.nodes_to_quantize
        ):
            return False

        if node.op_type not in quantizer.op_types_to_quantize:
            return False

        if quantizer.nodes_to_exclude is not None and node.name in quantizer.nodes_to_exclude:
            return False

        first_input_name = self.node.input[0]
        value_infos = self.quantizer.value_infos
        if first_input_name in value_infos.keys():
            vi = value_infos[first_input_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                return True

        return False

    @abstractmethod
    def do_quantization(self):
        raise NotImplementedError("Subclass should implement do_quantization(self).")

    @abstractmethod
    def fixup_quantization(self):
        raise NotImplementedError("Subclass should implement fixup_quantization.")


class QOperatorBase(QuantOperatorBase):
    """
    Base Class for QOperator format
    """
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def do_quantization(self):
        raise NotImplementedError("Subclass should overwrite do_quantization(self)")

    def fixup_quantization(self):
        for _, node_input in enumerate(self.node.input):
            dequantize_node = self.quantizer.dequantize_value(node_input)
            if dequantize_node is not None:
                self.quantizer.new_nodes.append(dequantize_node)

        # Append the original node
        self.quantizer.new_nodes.append(self.node)


class QDQOperatorBase(QuantOperatorBase):
    """
    Base Class for QDQ format
    """
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)
        self.disable_qdq_for_node_output = (
            True if onnx_node.op_type in onnx_quantizer.op_types_to_exclude_output_quantization else False
        )

    def do_quantization(self):
        """
        Default implementation. Overwrite the function for customized implementation.
        """
        node = self.node

        if self.disable_qdq_for_node_output:
            tensors_to_quantize = node.input
        else:
            tensors_to_quantize = itertools.chain(node.input, node.output)

        for tensor_name in tensors_to_quantize:
            self.quantizer.quantize_tensor(tensor_name)

    def fixup_quantization(self):
        pass
