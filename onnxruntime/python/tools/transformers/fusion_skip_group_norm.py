# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from typing import List

from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionSkipGroupNorm(Fusion):
    """
    Fuse Add + GroupNorm into one node: SkipGroupNorm
    """

    def __init__(self, model: OnnxModel):
        super().__init__(model, "SkipGroupNorm", "GroupNorm")
        # Update shape inference is needed since other fusions might add new edge which does not have shape info yet.
        self.shape_infer_helper = self.model.infer_runtime_shape(update=True)

        if self.shape_infer_helper is None:
            logger.warning("SkipGroupNorm fusion will be skipped since symbolic shape inference disabled or failed.")

    def create_transpose_node(self, input_name: str, perm: List[int], output_name=None):
        """Append a Transpose node after an input"""
        node_name = self.model.create_node_name("Transpose")
        if output_name is None:
            output_name = node_name + "_out" + "-" + input_name
        transpose_node = helper.make_node("Transpose", inputs=[input_name], outputs=[output_name], name=node_name)
        transpose_node.attribute.extend([helper.make_attribute("perm", perm)])
        return transpose_node

    def get_skip_index(self, add, is_channel_last: bool):
        skip = -1
        broadcast = False

        assert self.shape_infer_helper is not None
        shape_a = self.shape_infer_helper.get_edge_shape(add.input[0])
        shape_b = self.shape_infer_helper.get_edge_shape(add.input[1])
        assert shape_a is not None and shape_b is not None

        if len(shape_a) == 4 and len(shape_b) == 4:
            if shape_a == shape_b:
                skip = 1
            else:
                c = 3 if is_channel_last else 1
                h = 1 if is_channel_last else 2
                w = 2 if is_channel_last else 3
                if shape_a[0] == shape_b[0] and shape_a[c] == shape_b[c]:
                    if shape_b[h] == 1 and shape_b[w] == 1:
                        skip = 1
                        broadcast = True
                    elif shape_a[h] == 1 and shape_a[w] == 1:
                        skip = 0
                        broadcast = True

        if skip < 0:
            logger.debug(
                "skip SkipGroupNorm fusion since shape of Add inputs (%s, %s) are not expected",
                add.input[0],
                add.input[1],
            )
        return skip, broadcast

    def has_multiple_consumers(self, output_name, input_name_to_nodes):
        return (
            self.model.find_graph_output(output_name) is not None
            or (output_name in input_name_to_nodes)
            and (len(input_name_to_nodes[output_name]) > 1)
        )

    def remove_if_safe(self, node, input_name_to_nodes):
        # Remove a node if it is safe (only one children, and not graph output)
        if not self.has_multiple_consumers(node.output[0], input_name_to_nodes):
            self.nodes_to_remove.extend([node])

    def is_bias_1d(self, bias_name: str):
        initializer = self.model.get_initializer(bias_name)
        if initializer is None:
            return False

        bias_weight = NumpyHelper.to_array(initializer)
        if bias_weight is None:
            logger.debug("Bias weight not found")
            return False

        if len(bias_weight.shape) != 1:
            logger.debug("Bias weight is not 1D")
            return False
        return True

    def match_bias_path(self, node, input_name_to_nodes, output_name_to_node):
        add_input_index = []
        bias_nodes = self.model.match_parent_path(
            node, ["Reshape", "Add", "MatMul"], [0, 0, None], output_name_to_node, add_input_index
        )
        if bias_nodes is None:
            return None

        (reshape, add_bias, matmul) = bias_nodes
        bias = bias_nodes[1].input[1 - add_input_index[0]]
        if not self.is_bias_1d(bias):
            return None

        reshape.input[0] = matmul.output[0]
        self.remove_if_safe(add_bias, input_name_to_nodes)

        return bias

    def match_transpose_from_nhwc(self, output_name, input_name_to_nodes, output_name_to_node):
        parent = output_name_to_node[output_name] if output_name in output_name_to_node else None
        if parent is not None and parent.op_type == "Transpose":
            permutation = OnnxModel.get_node_attribute(parent, "perm")
            if permutation == [0, 3, 1, 2]:
                self.remove_if_safe(parent, input_name_to_nodes)
                return parent
        return None

    # Before Fusion:
    #     (a)  (b)
    #       \  /
    #       Add
    #       /\
    #   (c)   Transpose([0,2,3,1])
    #            \
    #          GroupNorm
    #             |
    #            (d)
    #
    # After Fusion:
    #           (a)              (b)
    #             \              /
    #   Transpose([0,2,3,1])   Transpose([0,2,3,1])
    #                \        /
    #              SkipGroupNorm
    #                  /    \
    #                 /     Transpose([0, 3, 1, 2])
    #                /        \
    #               (d)       (c)
    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if self.shape_infer_helper is None:
            return

        nodes = self.model.match_parent_path(node, ["Transpose", "Add"], [0, 0], output_name_to_node)
        if nodes is None:
            return

        (transpose, add) = nodes
        if transpose in self.nodes_to_remove or add in self.nodes_to_remove:
            return

        if self.has_multiple_consumers(transpose.output[0], input_name_to_nodes):
            return

        permutation = OnnxModel.get_node_attribute(transpose, "perm")
        assert isinstance(permutation, list)
        if permutation != [0, 2, 3, 1]:
            return

        inputs = []
        bias = None
        for i in range(2):
            matched_transpose = self.match_transpose_from_nhwc(add.input[i], input_name_to_nodes, output_name_to_node)
            if matched_transpose:
                inputs.append(matched_transpose.input[0])
                bias = self.match_bias_path(matched_transpose, input_name_to_nodes, output_name_to_node) or bias
            else:
                new_transpose = self.create_transpose_node(add.input[i], [0, 2, 3, 1])
                self.model.add_node(new_transpose, self.this_graph_name)
                inputs.append(new_transpose.output[0])

        skip, broadcast = self.get_skip_index(add, is_channel_last=False)
        if skip < 0:
            return

        inputs = [inputs[1 - skip], node.input[1], node.input[2], inputs[skip]]
        if bias:
            inputs = [*inputs, bias]

        outputs = node.output

        new_node_name = self.model.create_node_name(self.fused_op_type, name_prefix="SkipGroupNorm")
        if self.has_multiple_consumers(add.output[0], input_name_to_nodes):
            add_out_name = new_node_name + "_add_out"
            outputs.append(add_out_name)

            add_out_transpose = self.create_transpose_node(add_out_name, [0, 3, 1, 2], add.output[0])
            self.model.add_node(add_out_transpose, self.this_graph_name)

        skip_group_norm = helper.make_node(
            self.fused_op_type,
            inputs=inputs,
            outputs=outputs,
            name=new_node_name,
        )
        skip_group_norm.domain = "com.microsoft"

        self.increase_counter(
            f"SkipGroupNorm(add_out={len(outputs) > 1} bias={bias is not None} broadcast={broadcast})"
        )

        # Pass attributes from GroupNorm node to SkipGroupNorm
        for att in node.attribute:
            skip_group_norm.attribute.extend([att])

        self.nodes_to_remove.extend([add, transpose, node])
        self.nodes_to_add.append(skip_group_norm)
        self.node_name_to_graph_name[skip_group_norm.name] = self.this_graph_name
        self.prune_graph = True
