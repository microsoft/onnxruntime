#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from onnx import helper
from onnx_model import OnnxModel
from fusion_base import Fusion

# TODO(kreeger): Add ASCII art to document fusion process:
class FusionEmbedLayerNormBiasGelu(Fusion):
    """
    TODO(kreeger): Add some documentation here.
    """
    def __init__(self, model: OnnxModel):
        super().__init__(model, "EmbedLayerNormBiasGelu", "SkipLayerNormalization")

    def fuse(self, skip_layer_norm_node, input_name_to_nodes, output_name_to_node):
        if len(skip_layer_norm_node.input) != 5:
            return

        # Walk down through children:
        # 'MatMul' #1
        if skip_layer_norm_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[skip_layer_norm_node.output[0]]
        # SkipLayerNorm has 2 consumers of output - a 'MatMul' and another 'SkipLayerNormalization':
        if len(children) != 2 or children[0].op_type != 'MatMul':
            return
        matmul_1_node = children[0]

        # 'BiasGelu'
        if matmul_1_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[matmul_1_node.output[0]]
        if len(children) != 1 or children[0].op_type != 'BiasGelu':
            return
        bias_gelu_node = children[0]

        # 'MatMul' #2
        if bias_gelu_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[bias_gelu_node.output[0]]
        if len(children) != 1 or children[0].op_type != 'MatMul':
            return
        matmul_2_node = children[0]

        # Ensure next output is another SkipLayerNorm:
        if matmul_2_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[matmul_2_node.output[0]]
        if len(children) != 1 or children[0].op_type != 'SkipLayerNormalization':
            return

        # Build the new list of inputs for the 'EmbedLayerNormBiasGelu' node:
        # 0: SLN Input 0 (input)
        # 1: SLN Input 1 (skip)
        # 2: SLN Input 2 (gamma)
        # 3: SLN Input 3 (beta)
        # 4: SLN Input 4 (bias)
        # 5: MatMul #1 Input 1
        # 6: BiasGelu Input 1
        # 7: MatMul #2 Input 1
        inputs = [
            skip_layer_norm_node.input[0],
            skip_layer_norm_node.input[1],
            skip_layer_norm_node.input[2],
            skip_layer_norm_node.input[3],
            skip_layer_norm_node.input[4],
            matmul_1_node.input[1],
            bias_gelu_node.input[1],
            matmul_2_node.input[1]
        ]

        # Build the new list of outputs for the 'EmbedLayerNormBiasGelu' node:
        # 0: SLN Output 0
        # 1: MatMul #2 Output 0
        outputs = [
            skip_layer_norm_node.output[0],
            matmul_2_node.output[0]
        ]

        # TODO(kreeger): Need to carry over attributes!

        subgraph_nodes = [skip_layer_norm_node, matmul_1_node, bias_gelu_node, matmul_2_node]
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, outputs, input_name_to_nodes, output_name_to_node):
            return

        self.nodes_to_remove.extend(subgraph_nodes)

        # Construct the new 'EmbedLayerNormBiasGelu' node:
        fused_node = helper.make_node(
            'EmbedLayerNormBiasGelu',
            inputs=inputs,
            outputs=outputs,
            name=self.model.create_node_name('EmbedLayerNormBiasGelu'))
        fused_node.domain = 'com.microsoft'
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
