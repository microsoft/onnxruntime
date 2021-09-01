#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from typing import Dict
from logging import getLogger
from onnx import helper, TensorProto
from onnx_model import OnnxModel
from fusion_base import Fusion
from fusion_utils import FusionUtils

logger = getLogger(__name__)


class FusionEmbedLayerNoMask(Fusion):
    """
     Fuse embedding layer into one node (EmbedLayerNormalization).
     It supports the following model types: BERT, DistilBert, Roberta, ALBert.
    """
    def __init__(self, model: OnnxModel, description='no mask'):
        super().__init__(model, "EmbedLayerNormalization", ["LayerNormalization", "SkipLayerNormalization"],
                         description)
        self.utils = FusionUtils(model)
        self.shape_infer_helper = self.model.infer_runtime_shape({}, update=True)
        # The following will be reset in each fuse call of FusionEmbedLayerNormalization
        self.attention = None
        self.embed_node = None

    def match_two_gather(self, add):
        gather_0_path = self.model.match_parent_path(add, ['Gather'], [0])
        if gather_0_path is None:
            return None

        gather_1_path = self.model.match_parent_path(add, ['Gather'], [1])
        if gather_1_path is None:
            return None

        return gather_0_path[0], gather_1_path[0]

    def check_attention_subgraph(self, layernorm, input_name_to_nodes, is_distil_bert):
        """Check that LayerNormalization has a child of Attention node or subgraph like Attention.

        Args:
            layernorm (NodeProto): LayerNormalization node
            input_name_to_nodes (Dict[str, List[NodeProto]]): map from input name to nodes
            is_distil_bert (bool): whether it is DistilBert or not

        Returns:
            bool: whether there is Attention node or subgraph like Attention
        """
        self.attention = self.model.find_first_child_by_type(layernorm,
                                                             'Attention',
                                                             input_name_to_nodes,
                                                             recursive=False)
        if self.attention is None:
            # In case user disables attention fusion, check whether subgraph looks like Attention.
            if layernorm.output[0] not in input_name_to_nodes:
                return False
            children = input_name_to_nodes[layernorm.output[0]]

            # For Albert, there is MatMul+Add after embedding layer before attention.
            if len(children) == 1 and children[0].op_type == "MatMul" and children[0].output[0] in input_name_to_nodes:
                grandchildren = input_name_to_nodes[children[0].output[0]]
                if len(grandchildren) == 1 and grandchildren[0].op_type == "Add" and grandchildren[0].output[
                        0] in input_name_to_nodes:
                    nodes = input_name_to_nodes[grandchildren[0].output[0]]
                    for node in nodes:
                        if node.op_type == "Attention":
                            self.attention = node
                            return True
                    children_types = sorted([child.op_type for child in nodes])
            else:
                children_types = sorted([child.op_type for child in children])

            # Two Shape nodes might be merged by ORT
            if is_distil_bert:
                # SkipLayerNormailization might exist when model has been optimized by ORT first.
                if children_types != ['MatMul', 'MatMul', 'MatMul', 'Shape', 'SkipLayerNormalization'] and \
                   children_types != ['Add', 'MatMul', 'MatMul', 'MatMul', 'Shape', 'Shape'] and \
                   children_types != ['Add', 'MatMul', 'MatMul', 'MatMul', 'Shape']:
                    logger.debug("No Attention like subgraph in children of LayerNormalization")
                    return False
            else:
                if children_types != ['Add', 'MatMul', 'MatMul', 'MatMul'] and \
                   children_types != ['MatMul', 'MatMul', 'MatMul', 'SkipLayerNormalization']:
                    logger.debug("No Attention like subgraph in children of LayerNormalization")
                    return False
        return True

    def process_segment(self, segment_embedding_gather, output_name_to_node, input_ids_cast_node, nodes_to_remove,
                        nodes_to_add):
        segment_ids = segment_embedding_gather.input[1]

        nodes_to_remove.append(segment_embedding_gather)

        # Cast segment_ids to int32.
        segment_ids, segment_ids_cast_node = self.cast_to_int32(segment_ids)

        return segment_ids, segment_embedding_gather

    def create_fused_node(self, input_ids, layernorm, word_embedding_gather, position_embedding_weight_node,
                          has_segment_embedding, segment_embedding_gather, output_name_to_node, nodes_to_remove):
        nodes_to_add = []
        # Cast input_ids and segment_ids to int32.
        input_ids, input_ids_cast_node = self.cast_to_int32(input_ids)

        node_name = self.model.create_node_name('EmbedLayerNormalization')
        output_name = node_name + "_output"

        if layernorm.op_type == "LayerNormalization":
            gamma = layernorm.input[1]
            beta = layernorm.input[2]
        else:  # SkipLayerNormalization
            gamma = layernorm.input[2]
            beta = layernorm.input[3]

        embed_node_inputs = None
        if has_segment_embedding:
            segment_ids, segment_embedding_gather = self.process_segment(segment_embedding_gather, output_name_to_node,
                                                                         input_ids_cast_node, nodes_to_remove,
                                                                         nodes_to_add)

            embed_node_inputs = [
                input_ids, segment_ids, word_embedding_gather.input[0], position_embedding_weight_node.input[0],
                segment_embedding_gather.input[0], gamma, beta
            ]
        else:
            embed_node_inputs = [
                input_ids, '', word_embedding_gather.input[0], position_embedding_weight_node.input[0], '', gamma, beta
            ]

        embed_node = helper.make_node('EmbedLayerNormalization',
                                      embed_node_inputs,
                                      outputs=[node_name + "_output", node_name + "_dummy_mask_index"],
                                      name=node_name)

        embed_node.domain = "com.microsoft"

        # Pass attribute "epsilon" from normalize node to EmbedLayerNormalization.
        for att in layernorm.attribute:
            if att.name == 'epsilon':
                embed_node.attribute.extend([att])
        # Set default value to 1e-12 if no attribute is found.
        # OnnxRuntime 1.2.0 or older has no epsilon attribute. The optimized model can only work for 1.3.0 or later.
        if len(embed_node.attribute) == 0:
            embed_node.attribute.extend([helper.make_attribute("epsilon", 1.0E-12)])

        # Make sure new EmbedLayerNormalization node is the last one in self.nodes_to_add.
        nodes_to_add.append(embed_node)
        for node in nodes_to_add:
            self.node_name_to_graph_name[node.name] = self.this_graph_name
        self.nodes_to_add.extend(nodes_to_add)

        self.embed_node = embed_node
        return embed_node

    def finish_fusion(self, layernorm, embed_node, nodes_to_remove):
        self.model.replace_input_of_all_nodes(layernorm.output[0], embed_node.output[0])
        self.nodes_to_remove.extend(nodes_to_remove)
        # use prune graph to clean up postion embedding subgraph.
        self.prune_graph = True

    def match_position_embedding_distilbert(self, position_embedding_gather, input_ids, output_name_to_node):
        """  Match position embedding path from input_ids to Gather for DistilBert.

        DistilBert has word and position embeddings, subgraph pattern is like
                input_ids
                |      \
                |     Shape
                |       |   \
                |       |    Gather (indices=1)
                |       |       |
                |       |      Cast (optional)
                |       |       |
                |       |      Range (start=0, end=*, delta=1)
                |       |       |
                |       |    Unsqueeze
                |       |    /
                |      Expand
                |        |
             Gather    Gather
                  \   /
                   Add
                    |
            LayerNormalization
        """
        path1 = self.model.match_parent_path(position_embedding_gather, ['Expand', 'Shape'], [1, 1])
        if path1 is None:
            return False

        expand, shape = path1
        if shape.input[0] != input_ids:
            return False

        _, path2, _ = self.model.match_parent_paths(expand, [(['Unsqueeze', 'Range', 'Cast', 'Gather', 'Shape'], [0, 0, 1, 0, 0]), \
                                                             (['Unsqueeze', 'Range', 'Gather', 'Shape'], [0, 0, 1, 0])], output_name_to_node)
        if path2 is None:
            return False

        range_node = path2[1]
        if not (self.utils.check_node_input_value(range_node, 0, 0)
                and self.utils.check_node_input_value(range_node, 2, 1)):
            return False

        gather_node = path2[-2]
        if not (self.utils.check_node_input_value(gather_node, 1, 1)):
            return False

        shape_node = path2[-1]
        if shape_node.input[0] != input_ids:
            return False

        return True

    def match_position_embedding_roberta(self, position_embedding_gather, input_ids, output_name_to_node):
        """  Match position embedding path from input_ids to Gather for Roberta.

        Roberta Embedding Layer Pattern:       
                   (input_ids) -- Equal(B=0) -- Not
                          |                     |
                          |                   Cast (to=6)
                          |                     |  \
                          |                     |   CumSum(axis=1)
                          |                     |   /
                          |                     Mul
                          |                     |
                          |                    Cast (to=7)
                          |                     |
                        Gather (segment_ids)  Add (B=1)
                           \        |           |
                            \     Gather      Cast (to=7, optional)
                              \    /            |
                                Add          Gather
                                   \       /
                                      Add
                                       |
                                LayerNormalization
        """
        path = self.model.match_parent_path(position_embedding_gather,
                                            ['Cast', 'Add', 'Cast', 'Mul', 'CumSum', 'Cast', 'Not', 'Equal'],
                                            [1, 0, 0, 0, 0, 0, 0, 0], output_name_to_node)
        if path is not None:
            # constant input of Add shall be 1.
            i, value = self.model.get_constant_input(path[1])
            if value != 1:
                return False

            # constant input of Equal shall be 0. In distilroberta, the constant is 1
            i, value = self.model.get_constant_input(path[-1])
            if value != 0:
                return False

            return input_ids == path[-1].input[0]

        # Deal with optional Cast
        path = self.model.match_parent_path(position_embedding_gather,
                                            ['Add', 'Cast', 'Mul', 'CumSum', 'Cast', 'Not', 'Equal'],
                                            [1, 0, 0, 0, 0, 0, 0], output_name_to_node)
        if path is not None:
            return input_ids == path[-1].input[0]

        return False

    def match_position_embedding_bert(self, position_embedding_gather, input_ids, output_name_to_node):
        """  Match position embedding path from input_ids to Gather for BERT.

        BERT Embedding Layer Pattern:       
                                    (input_ids)
                                   /         \
                                 /          Shape
                                /              |
                              /              Gather (indices=1)
                            /                  |
                        Gather (segment_ids) Unsqueeze (axes=0)
                           \        |           |
                            \     Gather      Slice (data[1,512], starts=0, ends=*, axes=1, steps=1)
                              \    /            |
                                Add          Gather 
                                   \       /
                                      Add
                                       |
                                LayerNormalization
        """
        path = self.model.match_parent_path(position_embedding_gather, ['Slice', 'Unsqueeze', 'Gather', 'Shape'],
                                            [1, 2, 0, 0], output_name_to_node)
        if path is not None:
            slice, unsqueeze, gather, shape = path
            if not (self.utils.check_node_input_value(gather, 1, 1)):
                return False
            if not (self.utils.check_node_attribute(unsqueeze, "axes", [0], default_value=[0])):
                return False
            if not (self.utils.check_node_input_value(slice, 1, [0])):
                return False
            return input_ids == shape.input[0]

        return False

    def match_position_embedding(self, position_embedding_gather, input_ids, output_name_to_node):
        if self.match_position_embedding_bert(position_embedding_gather, input_ids, output_name_to_node):
            return True

        if self.match_position_embedding_roberta(position_embedding_gather, input_ids, output_name_to_node):
            return True

        if self.match_position_embedding_distilbert(position_embedding_gather, input_ids, output_name_to_node):
            return True

        return False

    def fuse_distilbert(self, layernorm, add_before_layernorm, input_name_to_nodes, output_name_to_node):
        """Fuse embedding layer for DistilBert
        Args:
            layernorm (NodeProto): node of LayerNormalization or SkipLayerNormalization
            add_before_layernorm (NodeProto): the Add node before LayerNormalization, or the SkipLayerNormalization itself
            input_name_to_nodes (Dict[str, List[NodeProto]]): map from input name to nodes
            output_name_to_node (Dict[str, List[NodeProto]]): map from output name to nodes
        """

        # DistilBert has no segment embedding, subgraph pattern is like
        #       input_ids
        #        |      \
        #        |     (position_embedding_subgraph)
        #        |        |
        #     Gather    Gather
        #          \   /
        #           Add
        #            |
        #    LayerNormalization
        two_gather = self.match_two_gather(add_before_layernorm)
        if two_gather is None:
            return False

        word_embedding_gather, position_embedding_gather = two_gather
        input_ids = word_embedding_gather.input[1]

        if not self.check_attention_subgraph(layernorm, input_name_to_nodes, is_distil_bert=True):
            return False

        if not self.match_position_embedding(position_embedding_gather, input_ids, output_name_to_node):
            return False

        if not self.check_embedding(word_embedding_gather, None, position_embedding_gather):
            return False

        nodes_to_remove = [add_before_layernorm, word_embedding_gather, position_embedding_gather
                           ] + ([] if layernorm == add_before_layernorm else [layernorm])
        embed_node = self.create_fused_node(input_ids, layernorm, word_embedding_gather, position_embedding_gather,
                                            False, None, output_name_to_node, nodes_to_remove)
        self.finish_fusion(layernorm, embed_node, nodes_to_remove)
        return True

    def check_embedding(self, word_embedding_gather, segment_embedding_gather, position_embedding_gather):
        """Sanity check of embedding weights, and match hidden_size of weights and shape of inputs.
        """
        input_ids = word_embedding_gather.input[1]
        segment_ids = segment_embedding_gather.input[1] if segment_embedding_gather else None
        position_ids = position_embedding_gather.input[1]

        if self.shape_infer_helper is not None:
            input_ids_shape = self.shape_infer_helper.get_edge_shape(input_ids)
            position_ids_shape = self.shape_infer_helper.get_edge_shape(position_ids)
            assert input_ids_shape and position_ids_shape
            if not (len(input_ids_shape) == 2 and len(position_ids_shape) == 2
                    and input_ids_shape[1] == position_ids_shape[1]):
                logger.info(
                    "Cannot fuse EmbedLayerNormalization: input_ids and position_ids not matched in 2nd dimension: {} vs {}"
                    .format(input_ids_shape, position_ids_shape))
                return False

            if segment_ids and not self.shape_infer_helper.compare_shape(input_ids, segment_ids):
                logger.info(
                    "Cannot fuse EmbedLayerNormalization: input_ids and segment_ids does not have same shape: {} != {}".
                    format(input_ids_shape, self.shape_infer_helper.get_edge_shape(segment_ids)))
                return False

        word_embedding_table = self.model.get_constant_value(word_embedding_gather.input[0])
        if word_embedding_table is None or len(word_embedding_table.shape) != 2:
            logger.info("Cannot fuse EmbedLayerNormalization: word embedding table is not expected")
            return False

        position_embedding_table = self.model.get_constant_value(position_embedding_gather.input[0])
        if position_embedding_table is None or len(position_embedding_table.shape) != 2 or (
                word_embedding_table.shape[1] != position_embedding_table.shape[1]):
            logger.info("Cannot fuse EmbedLayerNormalization: position embedding table is not expected")
            return False

        if segment_ids:
            segment_embedding_table = self.model.get_constant_value(segment_embedding_gather.input[0])
            if segment_embedding_table is None or len(segment_embedding_table.shape) != 2 or (
                    word_embedding_table.shape[1] != segment_embedding_table.shape[1]):
                logger.info("Cannot fuse EmbedLayerNormalization: segment embedding table is not expected")
                return False

        # In normal case, word embeding table is the largest, and segment embedding table is the smallest, while postion embedding table is in between.
        # TODO: use other information (like initializer names) to identify different embedding weights automatically.
        if word_embedding_table.shape[0] <= position_embedding_table.shape[0]:
            logger.warn(
                f"word_embedding_table ({word_embedding_gather.input[0]}) size {word_embedding_table.shape[0]} <= position_embedding_table ({position_embedding_gather.input[0]}) size {position_embedding_table.shape[0]}"
            )

        if segment_ids:
            if word_embedding_table.shape[0] <= segment_embedding_table.shape[0]:
                logger.warn(
                    f"word_embedding_table ({word_embedding_gather.input[0]}) size {word_embedding_table.shape[0]} <= segment_embedding_table ({segment_embedding_gather.input[0]}) size {segment_embedding_table.shape[0]}"
                )

            if position_embedding_table.shape[0] <= segment_embedding_table.shape[0]:
                logger.warn(
                    f"position_embedding_table ({position_embedding_gather.input[0]}) size {position_embedding_table.shape[0]} <= segment_embedding_table ({segment_embedding_gather.input[0]}) size {segment_embedding_table.shape[0]}"
                )

        return True

    def cast_graph_input_to_int32(self, input_name: str):
        graph_input = self.model.find_graph_input(input_name)
        if graph_input is not None and graph_input.type.tensor_type.elem_type != TensorProto.INT32:
            cast_output, cast_node = self.cast_input_to_int32(input_name)

    def cast_to_int32(self, input):
        input_cast_node = None
        graph_input = self.model.find_graph_input(input)
        if graph_input is not None:
            if graph_input.type.tensor_type.elem_type != TensorProto.INT32:
                int32_output, input_cast_node = self.utils.cast_input_to_int32(input)
            else:
                int32_output = input
        else:
            int32_output, input_cast_node = self.utils.cast_input_to_int32(input)

        return int32_output, input_cast_node

    def fuse_bert(self, layernorm, add_before_layernorm, input_name_to_nodes, output_name_to_node):
        """Fuse embedding layer for Bert
        Args:
            layernorm (NodeProto): node of LayerNormalization or SkipLayerNormalization
            add_before_layernorm (NodeProto): the Add node before LayerNormalization, or the SkipLayerNormalization itself
            input_name_to_nodes (Dict[str, List[NodeProto]]): map from input name to nodes
            output_name_to_node (Dict[str, List[NodeProto]]): map from output name to nodes
        """

        add_2_gather = self.model.match_parent_path(add_before_layernorm, ['Add'], [0])
        if add_2_gather is None:
            return False

        two_gather = self.match_two_gather(add_2_gather[0])
        if two_gather is None:
            return False

        word_embedding_gather, segment_embedding_gather = two_gather

        input_ids = word_embedding_gather.input[1]

        if not self.check_attention_subgraph(layernorm, input_name_to_nodes, is_distil_bert=False):
            return False

        position_embedding_path = self.model.match_parent_path(add_before_layernorm, ['Gather'], [1])
        if position_embedding_path is None:
            return False

        position_embedding_gather = position_embedding_path[0]
        if not self.match_position_embedding(position_embedding_gather, input_ids, output_name_to_node):
            if not self.match_position_embedding(segment_embedding_gather, input_ids, output_name_to_node):
                return False
            # position and segment are switched
            temp = segment_embedding_gather
            segment_embedding_gather = position_embedding_gather
            position_embedding_gather = temp

        if not self.check_embedding(word_embedding_gather, segment_embedding_gather, position_embedding_gather):
            return False

        nodes_to_remove = [add_before_layernorm] + ([] if layernorm == add_before_layernorm else [layernorm])
        embed_node = self.create_fused_node(input_ids, layernorm, word_embedding_gather, position_embedding_gather,
                                            True, segment_embedding_gather, output_name_to_node, nodes_to_remove)
        self.finish_fusion(layernorm, embed_node, nodes_to_remove)
        return True

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if node.op_type == "LayerNormalization":
            first_add_path = self.model.match_parent_path(node, ['Add'], [0])
            if first_add_path is None:
                return
            add_before_layernorm = first_add_path[0]
        else:  # SkipLayerNormalization
            add_before_layernorm = node  # Add is fused into SkipLayerNormalization

        if self.fuse_distilbert(node, add_before_layernorm, input_name_to_nodes, output_name_to_node):
            return

        if self.fuse_bert(node, add_before_layernorm, input_name_to_nodes, output_name_to_node):
            return


class FusionEmbedLayerNormalization(FusionEmbedLayerNoMask):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "with mask")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # Reset attention and embed_node so that we know fusion is successful when they are not None.
        self.attention = None
        self.embed_node = None
        super().fuse(node, input_name_to_nodes, output_name_to_node)

        if self.attention and self.embed_node:
            mask_index = self.attention.input[3]
            if mask_index in output_name_to_node:
                node = output_name_to_node[mask_index]
                if node.op_type == "ReduceSum":
                    embed_node = self.embed_node
                    mask_input_name = node.input[0]
                    self.nodes_to_remove.extend([node])
                    embed_node.input.append(mask_input_name)
                    embed_node.output[1] = mask_index
