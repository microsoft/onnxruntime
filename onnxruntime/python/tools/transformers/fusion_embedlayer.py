#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from typing import Dict, List, Tuple, Union
from logging import getLogger
from onnx import helper, TensorProto, NodeProto
from onnx_model import OnnxModel
from fusion_base import Fusion
from fusion_utils import FusionUtils

logger = getLogger(__name__)


class FusionEmbedLayerNoMask(Fusion):
    """
     Fuse embedding layer into one node (EmbedLayerNormalization).
     It supports the following model types: BERT, DistilBert, ALBert.
    """
    def __init__(self, model: OnnxModel, description: str = 'no mask'):
        super().__init__(model, "EmbedLayerNormalization", ["LayerNormalization", "SkipLayerNormalization"],
                         description)
        self.utils = FusionUtils(model)
        self.shape_infer_helper = self.model.infer_runtime_shape({}, update=True)
        # The following will be reset in each fuse call of FusionEmbedLayerNormalization
        self.attention = None
        self.embed_node = None

    def match_two_gather(self, add: NodeProto) -> Union[None, Tuple[NodeProto, NodeProto]]:
        gather_0_path = self.model.match_parent_path(add, ['Gather'], [0])
        if gather_0_path is None:
            return None

        gather_1_path = self.model.match_parent_path(add, ['Gather'], [1])
        if gather_1_path is None:
            return None

        return gather_0_path[0], gather_1_path[0]

    def check_attention_subgraph(self, layernorm: NodeProto, input_name_to_nodes: Dict[str, List[NodeProto]],
                                 is_distil_bert: bool) -> bool:
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

    def match_position_embedding_distilbert(self, position_embedding_gather, input_ids, output_name_to_node):
        """  Match position embedding path from input_ids to Gather for DistilBert.

        Pattern is like the following:
                 (input_ids)
                      |
                     Shape
                       |   \
                       |    Gather (indices=1)
                       |       |
                       |      Cast (optional)
                       |       |
                       |      Range (start=0, end=*, delta=1)
                       |       |
                       |    Unsqueeze
                       |    /
                      Expand
                        |
                      Gather
        """
        # remove after tests pass
        path1 = self.model.match_parent_path(position_embedding_gather, ['Expand', 'Shape'], [1, 1])
        if path1 is None:
            path1 = self.model.match_parent_path(position_embedding_gather, ['Expand', 'Where', 'Reshape', 'Shape'], [1, 1, 2, 0])
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

        Roberta Embedding Layer Pattern (* is optional since it might be removed by ORT, ? is the padding word id):       
          (input_ids) --> Equal(B=?) -- Not -- Cast(to=6) -- CumSum(axis=1) -- Mul -- Cast(to=7) -- Add(B=1) -- Cast(to=7)* --> Gather
                                                |                              ^
                                                V                              |
                                                +------------------------------+  

        Roberta new pattern from transformers v4.9:
           (input_ids) --> Equal(B=?) -- Not -- Cast(to=6) -- CumSum(axis=1) -- Add(B=0) -- Mul -- Cast(to=7) -- Add(B=1) --> Gather
                                                |                                           ^
                                                V                                           |
                                                +-------------------------------------------+  

        start_node = position_embedding_gather
        start_index = 1

        # match optional Cast node.
        parent = self.model.get_parent(start_node, start_index, output_name_to_node)
        if parent is None:
            return
        if parent.op_type == "Cast":
            if OnnxModel.get_node_attribute(parent, "to") != 7:
                return
            start_node = parent
            start_index = 0

        i, path, return_indices = self.model.match_parent_paths(
            start_node,
            [ (['Add', 'Cast', 'Mul', 'CumSum', 'Cast', 'Not', 'Equal'], [start_index, 0, 0, 0, 0, 0, 0]),
              (['Add', 'Cast', 'Mul', 'Add', 'CumSum', 'Cast', 'Not', 'Equal'], [start_index, 0, 0, 0, 0, 0, 0, 0])],
            output_name_to_node)

        if path is not None:
            # constant input of Add shall be 1.
            i, value = self.model.get_constant_input(path[0])
            if value != 1:
                return False

            _, self.padding_word_id = self.model.get_constant_input(path[-1])

            return input_ids == path[-1].input[0]
        """

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
                            /                  Add (optional, B=0)
                           /                    |
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
        path = self.model.match_parent_path(position_embedding_gather, ['Slice', 'Unsqueeze'], [1, 2],
                                            output_name_to_node)
        if path is None:
            return False

        slice, unsqueeze = path
        slice_weight = self.model.get_constant_value(slice.input[0])
        if not (slice_weight is not None  and len(slice_weight.shape) == 2 and slice_weight.shape[0] == 1 \
                and self.utils.check_node_input_value(slice, 1, [0]) \
                and self.utils.check_node_input_value(slice, 3, [1]) \
                and (len(slice.input) == 4 or self.utils.check_node_input_value(slice, 4, [1]))):
            return False

        opset_version = self.model.get_opset_version()
        if opset_version < 13:
            if not FusionUtils.check_node_attribute(unsqueeze, 'axes', [0]):
                return False
        else:
            if not self.utils.check_node_input_value(unsqueeze, 1, [0]):
                return False

        node = self.model.get_parent(unsqueeze, 0, output_name_to_node)
        if node is None:
            return False
        if node.op_type == "Add":
            if not self.utils.check_node_input_value(node, 1, 0):
                return False
            gather = self.model.get_parent(node, 0, output_name_to_node)
        else:
            gather = node

        if gather is None or gather.op_type != "Gather":
            return False
        if not (self.utils.check_node_input_value(gather, 1, 1)):
            return False

        shape = self.model.get_parent(gather, 0, output_name_to_node)
        if shape is None or shape.op_type != "Shape":
            return False

        return input_ids == shape.input[0]

    def match_position_embedding(self, position_embedding_gather, input_ids, output_name_to_node):
        if self.match_position_embedding_bert(position_embedding_gather, input_ids, output_name_to_node):
            return True

        # TODO: Support roberta (position starts from 2 instead of 0) in EmbedLayerNormalization kernel
        #       related: https://github.com/huggingface/transformers/issues/10736
        #if self.match_position_embedding_roberta(position_embedding_gather, input_ids, output_name_to_node):
        #    return True

        if self.match_position_embedding_distilbert(position_embedding_gather, input_ids, output_name_to_node):
            return True

        return False

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
            logger.warning(
                f"word_embedding_table ({word_embedding_gather.input[0]}) size {word_embedding_table.shape[0]} <= position_embedding_table ({position_embedding_gather.input[0]}) size {position_embedding_table.shape[0]}"
            )

        if segment_ids:
            if word_embedding_table.shape[0] <= segment_embedding_table.shape[0]:
                logger.warning(
                    f"word_embedding_table ({word_embedding_gather.input[0]}) size {word_embedding_table.shape[0]} <= segment_embedding_table ({segment_embedding_gather.input[0]}) size {segment_embedding_table.shape[0]}"
                )

            if position_embedding_table.shape[0] <= segment_embedding_table.shape[0]:
                logger.warning(
                    f"position_embedding_table ({position_embedding_gather.input[0]}) size {position_embedding_table.shape[0]} <= segment_embedding_table ({segment_embedding_gather.input[0]}) size {segment_embedding_table.shape[0]}"
                )

        return True

    def cast_to_int32(self, input_name: str) -> Tuple[str, Union[None, NodeProto]]:
        """Cast a graph input or node input to int32.

        Args:
            input_name (str): name of graph input or node input

        Returns:
            A tuple of casted input name and the cast node.
            int32_output (str): If input is int32, it is the input name, Otherwise it is output name of Cast node.
            input_cast_node (Union[None, NodeProto]): Cast node. It could be None if input is int32.
        """
        input_cast_node = None
        graph_input = self.model.find_graph_input(input_name)
        if graph_input is not None:
            if graph_input.type.tensor_type.elem_type != TensorProto.INT32:
                int32_output, input_cast_node = self.utils.cast_input_to_int32(input_name)
            else:
                int32_output = input_name
        else:
            int32_output, input_cast_node = self.utils.cast_input_to_int32(input_name)

        return int32_output, input_cast_node

    def create_fused_node(self, input_ids: str, layernorm: NodeProto, word_embedding_gather: NodeProto,
                          position_embedding_gather: NodeProto, segment_embedding_gather: Union[None, NodeProto],
                          position_ids: str = None, embedding_sum_output = False):
        """Create an EmbedLayerNormalization node. Note that segment embedding is optional.

        Args:
            input_ids (str): input_ids for word embeddings
            layernorm (NodeProto): LayerNormalization or SkipLayerNormalization node.
            word_embedding_gather (NodeProto): the Gather node for word embedding
            position_embedding_gather (NodeProto): the Gather node for position embedding
            segment_embedding_gather (Union[None, NodeProto]): the Gather node for segment embedding, or None.

        Returns:
            NodeProto: the EmbedLayerNormalization node created.
        """
        nodes_to_add = []
        input_ids, _ = self.cast_to_int32(input_ids)

        node_name = self.model.create_node_name('EmbedLayerNormalization')

        if layernorm.op_type == "LayerNormalization":
            gamma = layernorm.input[1]
            beta = layernorm.input[2]
        else:  # SkipLayerNormalization
            gamma = layernorm.input[2]
            beta = layernorm.input[3]

        embed_node_inputs = None
        if segment_embedding_gather is not None:
            segment_ids, _ = self.cast_to_int32(segment_embedding_gather.input[1])

            embed_node_inputs = [
                input_ids, segment_ids, word_embedding_gather.input[0], position_embedding_gather.input[0],
                segment_embedding_gather.input[0], gamma, beta
            ]
        else:  # no segment embedding
            embed_node_inputs = [
                input_ids, '', word_embedding_gather.input[0], position_embedding_gather.input[0], '', gamma, beta
            ]

        if position_ids is not None:
            #Adding an empty input for mask before position_ids
            embed_node_inputs.append('')
            position_ids, _ = self.cast_to_int32(position_ids)
            embed_node_inputs.append(position_ids)

        embed_node_outputs = [node_name + "_output", node_name + "_dummy_mask_index"]
        if embedding_sum_output:
            embed_node_outputs.append(node_name + "_embedding_sum")

        embed_node = helper.make_node('EmbedLayerNormalization',
                                      embed_node_inputs,
                                      outputs=embed_node_outputs,
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

    def finish_fusion(self, layernorm, embed_node):
        self.model.replace_input_of_all_nodes(layernorm.output[0], embed_node.output[0])
        # use prune graph to remove nodes that is not needed
        self.prune_graph = True

    def is_embedding_sum_needed(self, add_before_layer_norm):
        """Check that Add before layer norm has an output to add before next layernorm

        Args:
            add_before_layer_norm (NodeProto): Add before any LayerNormalization node in topological order of graph

        Returns:
            bool: whether there is an extra output needed out of embed layer norm node
        """

        nodes = self.model.get_children(add_before_layer_norm)

        return len(nodes) > 1

    def fuse_gpt2(self, layernorm, add_before_layernorm, input_name_to_nodes, output_name_to_node):
        #graph checks
        # gpt2 has no segment embedding, subgraph pattern is like
        #     input_ids  position_ids
        #        |        |
        #     Gather    Gather
        #          \   /
        #           Add _ _ _ _ _
        #            |           |
        #    LayerNormalization  |
        #            |           |
        #         Attention      |
        #            |           |
        #          Matmul        |
        #            |          /
        #           Add        /
        #             \       /
        #                Add
        two_gather = self.match_two_gather(add_before_layernorm)
        if two_gather is None:
            return False

        add_output = add_before_layernorm.output[0]

        word_embedding_gather, position_embedding_gather = two_gather
        input_ids = word_embedding_gather.input[1]
        position_ids = position_embedding_gather.input[1]

        if not self.check_attention_subgraph(layernorm, input_name_to_nodes, is_distil_bert=False):
            return False

        if not self.check_embedding(word_embedding_gather, None, position_embedding_gather):
            return False

        optional_embedding_sum_output = False
        if self.is_embedding_sum_needed(add_before_layernorm):
            optional_embedding_sum_output = True

        # make the fused node
        embed_node = self.create_fused_node(input_ids, layernorm, word_embedding_gather, position_embedding_gather,
                                           None, position_ids, optional_embedding_sum_output)

        # direct the output to another add too
        self.model.replace_input_of_all_nodes(layernorm.output[0], embed_node.output[0])
        if optional_embedding_sum_output:
            self.model.replace_input_of_all_nodes(add_output, embed_node.output[2])

        return True

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

        embed_node = self.create_fused_node(input_ids, layernorm, word_embedding_gather, position_embedding_gather,
                                            None)
        self.finish_fusion(layernorm, embed_node)
        return True

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

        embed_node = self.create_fused_node(input_ids, layernorm, word_embedding_gather, position_embedding_gather,
                                            segment_embedding_gather)
        self.finish_fusion(layernorm, embed_node)
        return True

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if node.op_type == "LayerNormalization":
            first_add_path = self.model.match_parent_path(node, ['Add'], [0])
            if first_add_path is None:
                return
            add_before_layernorm = first_add_path[0]
        else:  # SkipLayerNormalization
            add_before_layernorm = node  # Add is fused into SkipLayerNormalization

        if self.fuse_gpt2(node, add_before_layernorm, input_name_to_nodes, output_name_to_node):
            return

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
