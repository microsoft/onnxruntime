import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg

class QPad(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Pad")

        # Only after version 11, it has the optional constant_value
        if (self.quantizer.opset_version < 11) or (node.input[0] not in self.quantizer.quantized_value_map):
            super().quantize()
            return

        kwargs = {}
        for attribute in node.attribute:
            kv = attribute_to_kwarg(attribute)
            kwargs.update(kv)

        constant_value = None
        if 'mode' not in kwargs or kwargs['mode'] == 'constant':
            tensor = self.quantizer.model.get_initializer(quantized_input_value.zp_name)
            if tensor is None:
                super().quantize()
                return
            tensor_array = onnx.numpy_helper.to_array(tensor)
            constant_value = tensor_array.item() if tensor_array.ndim == 0 else tensor_array[0]
            kwargs.update({'const_value' : constant_value})

        # Create an entry for output quantized value
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]
        quantized_output_value = QuantizedValue(node.output[0], node.output[0] + "_quantized",
                                                quantized_input_value.scale_name, quantized_input_value.zp_name,
                                                QuantizedValueType.Input)
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        if constant_value is not None:
            qnode = onnx.helper.make_node("Pad", [quantized_input_value.q_name], [quantized_output_value.q_name], node.name, **kwargs)
            self.quantizer.new_nodes += [qnode]
        else:
            node.input[0] = quantized_input_value.q_name
            node.output[0] = quantized_output_value.q_name
            self.quantizer.new_nodes += [node]
