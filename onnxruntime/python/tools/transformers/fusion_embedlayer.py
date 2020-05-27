#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from typing import Dict
from logging import getLogger
from onnx import helper
from OnnxModel import OnnxModel
from fusion_base import Fusion
from fusion_utils import FusionUtils

logger = getLogger(__name__)


class FusionEmbedLayerNoMask(Fusion):
    """
     Embed Layer Normalization will fuse embeddings and mask processing into one node.
     The embeddings before conversion:

     (input_ids) -------->  Gather ----------+       (segment_ids)
        |                                    |            |
        |                                    v            v
        +--> Shape --> Expand -> Gather---->Add         Gather
        |                ^                   |            |
        |                |                   v            v
        +---(optional graph)               SkipLayerNormalization

      Optional graph is used to generate position list (0, 1, ...) per batch. It can be a constant in some model.

      (input_ids) --> Gather -----+           Slice
                                  |            |
                                  v            v
     (segment_ids)--> Gather --->Add        Reshape
                                  |            |
                                  v            v
                              SkipLayerNormalization
    """
    def __init__(self,
                 model: OnnxModel,
                 name: str = "EmbedLayerNormalization(no mask)",
                 search_op_types="SkipLayerNormalization"):
        super().__init__(model, name, search_op_types)
        self.utils = FusionUtils(model)

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # already fused. Assumes that only one mebedding layer in a transformer model.
        if self.nodes_to_add:
            return

        if self.model.match_parent_path(node, ['Add', 'Gather'], [0, 0]) is None:
            return

        if self.model.find_first_child_by_type(node, 'Attention', input_name_to_nodes, recursive=False) is None:
            # In case user disables attention fusion, check whether subgraph looks like Attention.
            if node.output[0] not in input_name_to_nodes:
                return
            children = input_name_to_nodes[node.output[0]]
            children_types = sorted([child.op_type for child in children])
            if children_types != ['MatMul', 'MatMul', 'MatMul', 'SkipLayerNormalization']:
                return

        # Assume the order of embeddings are word_embedding + position_embedding + segment_embedding
        normalize_node = node
        word_embedding_path = self.model.match_parent_path(normalize_node, ['Add', 'Gather'], [0, 0])
        if word_embedding_path is None:
            logger.info("Failed to find word embedding")
            return
        add_node, word_embedding_gather = word_embedding_path
        input_ids = word_embedding_gather.input[1]

        position_embedding_expand = None
        position_embedding_shape = None

        position_embedding_path = self.model.match_parent_path(normalize_node, ['Reshape', 'Slice'], [1, 0])
        if position_embedding_path is not None:
            _, position_embedding_weight_node = position_embedding_path
        else:
            position_embedding_path = self.model.match_parent_path(add_node, ['Gather', 'Expand', 'Shape'], [1, 1, 1])
            if position_embedding_path is not None:
                position_embedding_weight_node, position_embedding_expand, position_embedding_shape = position_embedding_path
            else:
                position_embedding_path = self.model.match_parent_path(
                    add_node, ['Gather', 'Expand', 'Concat', 'Unsqueeze', 'Gather', 'Shape'], [1, 1, 1, 1, 0, 0])
                if position_embedding_path is not None:
                    position_embedding_weight_node, position_embedding_expand, _, _, _, position_embedding_shape = position_embedding_path
                else:
                    # Here we will not try to get exact match. Instead, we only try identify position embedding weights.
                    position_embedding_path = self.model.match_parent_path(add_node, ['Gather', 'Expand'], [1, 1])
                    if position_embedding_path is not None:
                        position_embedding_weight_node, position_embedding_expand = position_embedding_path
                    else:
                        logger.info("Failed to find position embedding")
                        return

            if position_embedding_shape is not None and position_embedding_shape.input[0] != input_ids:
                logger.info("position and word embedding is expected to be applied on same input")
                return

        segment_embedding_path = self.model.match_parent_path(normalize_node, ['Gather'], [1])
        if segment_embedding_path is None:
            segment_embedding_path = self.model.match_parent_path(normalize_node, ['Add', 'Gather'], [0, 1])
            if segment_embedding_path is None:
                logger.info("Failed to find segment embedding")
                return
            _, segment_embedding_gather = segment_embedding_path
        else:
            segment_embedding_gather = segment_embedding_path[0]

        segment_ids = segment_embedding_gather.input[1]

        if position_embedding_expand and position_embedding_shape:
            input_parent = self.model.get_parent(position_embedding_shape, 0, output_name_to_node)
            subgraph_nodes = self.model.get_parent_subgraph_nodes(position_embedding_expand,
                                                                  [input_parent] if input_parent else [],
                                                                  output_name_to_node)
            self.nodes_to_remove.extend(subgraph_nodes)

        self.nodes_to_remove.extend(word_embedding_path)
        self.nodes_to_remove.extend(position_embedding_path)
        self.nodes_to_remove.extend(segment_embedding_path)

        self.nodes_to_remove.extend([normalize_node])

        # store inputs for further processing
        if self.model.find_graph_input(input_ids):
            self.model.bert_inputs = [input_ids, segment_ids
                                      ] if self.model.find_graph_input(segment_ids) else [input_ids]

        # Cast input_ids and segment_ids to int32.
        input_ids_cast_node = None
        if self.model.find_graph_input(input_ids):
            casted, input_ids = self.utils.cast_graph_input_to_int32(input_ids)
        else:
            input_ids, input_ids_cast_node = self.utils.cast_input_to_int32(input_ids)

        if self.model.find_graph_input(segment_ids):
            casted, segment_ids = self.utils.cast_graph_input_to_int32(segment_ids)
        else:
            segment_ids, segment_ids_cast_node = self.utils.cast_input_to_int32(segment_ids)

            # Cast might be removed by OnnxRuntime.
            _, segment_id_path, _ = self.model.match_parent_paths(
                segment_ids_cast_node, 
                [(['ConstantOfShape', 'Concat', 'Unsqueeze', 'Gather', 'Shape', 'Cast'], [0, 0, 1, 0, 0, 0]),
                 (['ConstantOfShape', 'Concat', 'Unsqueeze', 'Gather', 'Shape'], [0, 0, 1, 0, 0])],
                output_name_to_node)

            if segment_id_path and input_ids_cast_node and input_ids_cast_node.input[0] == segment_id_path[-1].input[0]:
                logger.debug("Simplify semgent id path...")
                self.model.add_node(
                    helper.make_node('Shape', inputs=[input_ids_cast_node.input[0]], outputs=["input_shape"]))
                self.model.add_node(
                    helper.make_node('ConstantOfShape',
                                     inputs=["input_shape"],
                                     outputs=["zeros_for_input_shape"],
                                     value=helper.make_tensor("value", onnx.TensorProto.INT32, [1], [1])))
                segment_ids = "zeros_for_input_shape"

        embed_node = helper.make_node(
            'EmbedLayerNormalization',
            inputs=[
                input_ids,
                segment_ids,
                word_embedding_gather.input[0],
                position_embedding_weight_node.input[0],
                segment_embedding_gather.input[0],
                normalize_node.input[2],
                normalize_node.input[3]  # gamma and beta
            ],
            outputs=["embed_output", "dummy_mask_index"],
            name="EmbedLayer")

        embed_node.domain = "com.microsoft"

        # Pass attribute "epsilon" from normalize node to EmbedLayerNormalization.
        for att in normalize_node.attribute:
            if att.name == 'epsilon':
                embed_node.attribute.extend([att])
        # Set default value to 1e-12 if no attribute is found.
        if len(embed_node.attribute) == 0:
            embed_node.attribute.extend([onnx.helper.make_attribute("epsilon", 1.0E-12)])

        self.model.replace_input_of_all_nodes(normalize_node.output[0], 'embed_output')
        self.nodes_to_add.append(embed_node)


class FusionEmbedLayerNormalization(FusionEmbedLayerNoMask):
    def __init__(self, model: OnnxModel, mask_indice: Dict, mask_casted: Dict):
        super().__init__(model, "EmbedLayerNormalization(with mask)", "SkipLayerNormalization")
        self.mask_indice: Dict = mask_indice
        self.mask_casted: Dict = mask_casted
        self.mask_input_name = None

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # already fused. Assumes that only one mebedding layer in a transformer model.
        if self.nodes_to_add:
            return

        super().fuse(node, input_name_to_nodes, output_name_to_node)
        if not self.nodes_to_add:
            return

        if len(self.nodes_to_add[0].input) != 7:
            return

        assert len(self.mask_indice) <= 1, "Unexpected: there are multiple mask inputs found!"
        if len(self.mask_indice) != 1:
            logger.info("Fused EmbedLayerNormalization (no mask) count: 1")
        else:
            embed_node = self.nodes_to_add.pop()
            mask_input_name = next(iter(self.mask_indice))
            mask_output_name = self.mask_indice[mask_input_name]
            mask_node = output_name_to_node[mask_output_name]

            self.nodes_to_remove.extend([mask_node])

            # store inputs for further processing
            self.mask_input_name = mask_input_name

            # When mask has been casted to int32, use that casted one as input of embed layer norm.
            if mask_input_name in self.mask_casted:
                mask_input_name = self.mask_casted[mask_input_name]

            embed_node.input.append(mask_input_name)
            embed_node.output[1] = mask_output_name
            self.nodes_to_add.append(embed_node)
            self.prune_graph = True
