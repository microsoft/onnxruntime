import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import attribute_to_kwarg, ms_domain
from onnx import onnx_pb as onnx_proto

'''
TODO(kreeger): Doc me.
'''
class EmbedLayerNormBiasGeluQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "EmbedLayerNormBiasGelu")

        '''
        Pre-quantization EmbedLayerNorm inputs:
        [0] SLN Input 0 (input)
        [1] SLN Input 1 (skip)
        [2] SLN Input 2 (gamma)
        [3] SLN Input 3 (beta)
        [4] SLN Input 4 (bias)
        [5] MatMul #1 Input 1
        [6] BiasGelu Input 1
        [7] MatMul #2 Input 1
        '''
        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [2, 3, 4, 5, 6, 7])

        name = "" if node.name == "" else node.name + "_quant"

        '''
        Quantized Input Tensor List
        [0] SLN Input 0 (input)
        [1] SLN Input 1 (skip)
        [2] SLN Input 2 (gamma)
        [3] SLN Input 3 (beta)
        [4] SLN Input 4 (bias)
        [5] MatMul #1 Input 1
        [6] BiasGelu Input 1
        [7] MatMul #2 Input 1
	    [8] SLN Input 2 (gamma) scale (float)
	    [9] SLN Input 3 (beta) scale (float)
	    [10] SLN Input 4 (bias) scale (float)
	    [11] MatMul #1 Input 1 scale (float)
	    [12] BiasGelu Input 1 scale (float)
	    [13] MatMul #2 Input 1 scale (float)
	    [14] SLN Input 2 (gamma) zero point (uint8)
	    [15] SLN Input 3 (beta) zero point (uint8)
	    [16] SLN Input 4 (bias) zero point (uint8)
	    [17] MatMul #1 Input 1 zero point (uint8)
	    [18] BiasGelu Input 1 zero point (uint8)
	    [19] MatMul #2 Input 1 zero point (uint8)
        '''
        inputs = []
        # 'SLN Input 0 (input)'
        inputs.extend([node.input[0]])
        # 'SLN Input 1 (skip)'
        inputs.extend([node.input[1]])
        # 'SLN Input 2 (gamma)'
        inputs.extend([quantized_input_names[0]])
        # 'SLN Input 3 (beta)'
        inputs.extend([quantized_input_names[1]])
        # 'SLN Input 4 (bias)'
        inputs.extend([quantized_input_names[2]])
        # 'MatMul #1 Input 1'
        inputs.extend([quantized_input_names[3]])
        # 'BiasGelu Input 1'
        inputs.extend([quantized_input_names[4]])
        # 'MatMul #2 Input 1'
        inputs.extend([quantized_input_names[4]])

        # Add all scales:
        inputs.extend([scale_names[0]])
        inputs.extend([scale_names[1]])
        inputs.extend([scale_names[2]])
        inputs.extend([scale_names[3]])
        inputs.extend([scale_names[4]])
        inputs.extend([scale_names[5]])

        # Add all zero points:
        inputs.extend([zero_point_names[0]])
        inputs.extend([zero_point_names[1]])
        inputs.extend([zero_point_names[2]])
        inputs.extend([zero_point_names[3]])
        inputs.extend([zero_point_names[4]])
        inputs.extend([zero_point_names[5]])

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        quantized_node = onnx.helper.make_node("QEmbedLayerNormBiasGelu", inputs, node.output, name, **kwargs)
        nodes.append(quantized_node)

        self.quantizer.new_nodes += nodes
