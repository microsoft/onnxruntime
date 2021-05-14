#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from logging import getLogger
from onnx import helper
from onnx_model import OnnxModel
from fusion_base import Fusion
from fusion_utils import NumpyHelper

logger = getLogger(__name__)


class FusionSkipLayerNormalization(Fusion):
    """
    Fuse Add + LayerNormalization into one node: SkipLayerNormalization
    Note: This fusion does not check the input shape of Add and LayerNormalization.
    """
    def __init__(self, model: OnnxModel):
        super().__init__(model, "SkipLayerNormalization", "LayerNormalization")
        self.shape_infer_helper = self.model.infer_runtime_shape({"batch_size": 4, "seq_len": 7})

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        add = self.model.get_parent(node, 0, output_name_to_node)

        # In some models there is input_ids->gather->add->LayerNorm and one of input of the
        # add node is initializer with fixed shape which should not be fused into SkipLayerNorm
        if add is None:
            return

        for add_input in add.input:
            if self.model.get_initializer(add_input) != None:
                return

        # The number of input node of add should be 2
        parents = self.model.get_parents(add)
        if len(parents) != 2:
            return

        # work around for fluency-bart
        if parents[0].op_type == 'Mul' and parents[1].op_type == 'Slice':
            return

        gather_node = self.model.match_parent_path(add, ['Gather'], [None])
        if gather_node is not None:
            return

        if self.shape_infer_helper is not None:
            if not self.shape_infer_helper.compare_shape(add.input[0], add.input[1]):
                return
        else:
            # shape_infer_helper can not handle subgraphs. Current work around is to disable skiplayernorm fusion
            # longterm todo: support subgraph in symbolic_shape_infer or support add broadcasting in skiplayernorm op
            logger.warning(
                "symbolic shape infer failed. it's safe to ignore this message if there is no issue with optimized model"
            )

        gather_path = self.model.match_parent_path(add, ['Gather'], [None])
        if gather_path is not None and self.model.find_graph_input(gather_path[0].input[1]) is None:
            if self.model.match_parent_path(gather_path[0], ['ConstantOfShape'], [1]) is None:
                return

        if add is not None and add.op_type == 'Add' and self.model.is_safe_to_fuse_nodes(
            [add, node], node.output, input_name_to_nodes, output_name_to_node):
            self.nodes_to_remove.extend([add, node])

            inputs = [add.input[0], add.input[1], node.input[1], node.input[2]]
            normalize_node = helper.make_node("SkipLayerNormalization",
                                              inputs=inputs,
                                              outputs=[node.output[0]],
                                              name=self.model.create_node_name("SkipLayerNormalization",
                                                                               name_prefix="SkipLayerNorm"))
            normalize_node.domain = "com.microsoft"

            # Pass attribute "epsilon" from layernorm node to SkipLayerNormalization
            for att in node.attribute:
                if att.name == 'epsilon':
                    normalize_node.attribute.extend([att])

            # Set default epsilon if no epsilon exists from layernorm
            if len(normalize_node.attribute) == 0:
                normalize_node.attribute.extend([helper.make_attribute("epsilon", 1.0E-12)])

            self.nodes_to_add.append(normalize_node)
            self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name


class FusionBiasSkipLayerNormalization(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "SkipLayerNormalization", "SkipLayerNormalization", "add bias")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if len(node.input) != 4:
            return

        return_indice = []
        nodes = self.model.match_parent_path(node, ['Add', 'MatMul'], [None, None], None, return_indice)
        if nodes is None:
            return
        assert len(return_indice) == 2
        add_input_index = return_indice[0]
        if add_input_index >= 2:
            return

        (add, matmul) = nodes

        # bias should be one dimension
        bias_index = -1
        for i, input in enumerate(add.input):
            initializer = self.model.get_initializer(input)
            if initializer is None:
                continue
            bias_index = i
            bias_weight = NumpyHelper.to_array(initializer)
            break
        if bias_weight is None:
            logger.debug(f"Bias weight not found")
            return
        if len(bias_weight.shape) != 1:
            logger.debug(f"Bias weight is not 1D")
            return

        subgraph_nodes = [node, add]
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, [node.output[0]], input_name_to_nodes,
                                                output_name_to_node):
            logger.debug(f"Skip fusing SkipLayerNormalization with Bias since it is not safe")
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        inputs = [
            node.input[1 - add_input_index], matmul.output[0], node.input[2], node.input[3], add.input[bias_index]
        ]
        new_node = helper.make_node("SkipLayerNormalization",
                                    inputs=inputs,
                                    outputs=node.output,
                                    name=self.model.create_node_name("SkipLayerNormalization",
                                                                     "SkipLayerNorm_AddBias_"))
        new_node.domain = "com.microsoft"

        # Pass attribute "epsilon" from skiplayernorm node to skiplayernorm(add bias)
        for att in node.attribute:
            if att.name == 'epsilon':
                new_node.attribute.extend([att])

        # Set default epsilon if no epsilon exists from skiplayernorm
        if len(new_node.attribute) == 0:
            new_node.attribute.extend([helper.make_attribute("epsilon", 1.0E-12)])

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
