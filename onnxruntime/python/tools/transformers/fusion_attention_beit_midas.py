# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from typing import Tuple, Union

import numpy as np
from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import NodeProto, TensorProto, helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionAttentionBeitMidas(Fusion):
    """
    Fuse Attention subgraph of Beit (from MiDaS) into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__(model, "MultiHeadAttention", ["SkipLayerNormalization"])
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads_and_hidden_size(
        self, reshape_before_split: NodeProto, add_before_split: NodeProto
    ) -> Tuple[int, int]:
        """Detect num_heads and hidden_size for ONNX model from MiDaS

        Args:
            reshape_before_split (NodeProto): reshape node before Split
            add_before_split (NodeProto): add node before Split

        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """

        concat = self.model.match_parent(reshape_before_split, "Concat", 1)
        if concat is None or len(concat.input) != 5:
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        num_head_value = self.model.get_constant_value(concat.input[3])
        if num_head_value is None:
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        if len(num_head_value) != 1 or num_head_value[0] <= 0:
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        num_heads = num_head_value[0]

        i, bias = self.model.get_constant_input(add_before_split)
        if bias is None or bias.shape[0] % 3 != 0:
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        hidden_size = bias.shape[0] // 3

        if self.num_heads > 0 and num_heads != self.num_heads:
            if self.num_heads_warning:
                logger.warning(f"--num_heads is {self.num_heads}. Detected value is {num_heads}. Using detected value.")
                self.num_heads_warning = False  # Do not show the warning more than once

        if self.hidden_size > 0 and hidden_size != self.hidden_size:
            if self.hidden_size_warning:
                logger.warning(
                    f"--hidden_size is {self.hidden_size}. Detected value is {hidden_size}. Using detected value."
                )
                self.hidden_size_warning = False  # Do not show the warning more than once

        return num_heads, hidden_size

    def create_attention_node(
        self,
        matmul_before_split: NodeProto,
        add_before_split: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        relative_position_bias_input: str,
        output: str,
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            matmul_before_split (NodeProto): MatMul node for input projection of Q/K/V
            add_before_split (NodeProto): Add node in input projection of Q/K/V
            num_heads (int): number of attention heads.
            hidden_size (int): hidden dimension.
            input (str): input name
            output (str): output name
        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        weight = self.model.get_initializer(matmul_before_split.input[1])
        if not weight:
            return None

        if weight.data_type == 10:
            logger.debug("weights are in fp16. Please run fp16 conversion after optimization")
            return None

        w = NumpyHelper.to_array(weight)
        if len(w.shape) != 2:
            logger.debug("weights shall be 2D")
            return None
        logger.debug(f"w={w.shape} hidden_size={hidden_size}")

        w_in_size = w.shape[0]
        if hidden_size > 0 and hidden_size != w_in_size:
            raise ValueError(
                f"Input hidden size ({hidden_size}) is not same as weight dimension of q,k,v ({w_in_size}). "
                "Please provide a correct input hidden size or pass in 0"
            )

        qw, kw, vw = np.split(w, 3, axis=1)

        attention_node_name = self.model.create_node_name("MultiHeadAttention")

        c = w_in_size
        n = num_heads
        h = qw.shape[1] // num_heads

        # Concat and interleave weights so that the output of fused KV GEMM has [B, S_kv, N, 3, H] shape
        qkv_weight = np.dstack([qw.reshape(c, n, h), kw.reshape(c, n, h), vw.reshape(c, n, h)]).reshape(c, n * 3 * h)

        matmul_node_name = self.model.create_node_name("MatMul", name_prefix="MatMul_QKV")
        weight = helper.make_tensor(
            name=matmul_node_name + "_weight",
            data_type=TensorProto.FLOAT,
            dims=[qkv_weight.shape[0], qkv_weight.shape[1]],
            vals=qkv_weight.flatten().tolist(),
        )

        self.model.add_initializer(weight, self.this_graph_name)

        matmul_node = helper.make_node(
            "MatMul",
            inputs=[input, matmul_node_name + "_weight"],
            outputs=[matmul_node_name + "_out"],
            name=matmul_node_name,
        )
        self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name

        add_node_name = self.model.create_node_name("Add", name_prefix="Add_QKV")
        add_node = helper.make_node(
            "Add",
            inputs=[matmul_node_name + "_out", add_before_split.input[0]],
            outputs=[add_node_name + "_out"],
            name=add_node_name,
        )
        self.node_name_to_graph_name[add_node.name] = self.this_graph_name

        shape_tensor = helper.make_tensor(
            name=matmul_node_name + "_reshape_shape",
            data_type=TensorProto.INT64,
            dims=[5],
            vals=[0, 0, n, 3, h],
        )
        self.model.add_initializer(shape_tensor, self.this_graph_name)

        reshape_node = helper.make_node(
            "Reshape",
            inputs=[add_node_name + "_out", matmul_node_name + "_reshape_shape"],
            outputs=[attention_node_name + "_input"],
            name=matmul_node_name + "_reshape",
        )
        self.node_name_to_graph_name[reshape_node.name] = self.this_graph_name
        self.nodes_to_add.extend([matmul_node, add_node, reshape_node])
        self.nodes_to_remove.extend([matmul_before_split, add_before_split])

        attention_inputs = [attention_node_name + "_input", "", "", "", "", relative_position_bias_input]

        attention_node = helper.make_node(
            "MultiHeadAttention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        node_before_layernorm = self.model.match_parent(normalize_node, "SkipLayerNormalization", 0)
        if node_before_layernorm is not None:
            root_input = node_before_layernorm.output[0]
        else:
            node_before_layernorm = self.model.match_parent(normalize_node, "Concat", 0)
            if node_before_layernorm is None:
                return
            child = self.model.find_first_child_by_type(
                node_before_layernorm, "LayerNormalization", input_name_to_nodes, False
            )
            if child is None:
                return
            root_input = child.output[0]

        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["Mul", "Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [1, None, None, 0, 0, 0],
        )
        if qkv_nodes is None:
            return

        (_, _, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

        v_nodes = self.model.match_parent_path(
            matmul_qkv, ["Squeeze", "Transpose", "Split", "Reshape", "Add", "MatMul"], [1, 0, 0, 0, 0, None]
        )
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (_, _, split_qkv, reshape_before_split, add_before_split, matmul_before_split) = v_nodes
        if len(split_qkv.output) != 3:
            logger.debug("fuse_attention: split_qkv shall have 3 outputs")
            return
        if root_input not in matmul_before_split.input:
            logger.debug("fuse_attention: matmul_before_split shall be root input")
            return

        qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "Add", "MatMul"], [0, 0, 0])
        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return

        (_softmax_qk, add_bias_qk, matmul_qk) = qk_nodes

        q_nodes = self.model.match_parent_path(matmul_qk, ["Mul", "Squeeze", "Transpose"], [0, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        (_, _squeeze_q, transpose_q) = q_nodes
        if transpose_q.input[0] != split_qkv.output[0]:
            logger.debug("fuse_attention: transpose_q input shall be Split output")
            return

        k_nodes = self.model.match_parent_path(matmul_qk, ["Transpose", "Squeeze"], [1, 0])
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return

        (_, squeeze_k) = k_nodes
        if squeeze_k.input[0] != split_qkv.output[1]:
            logger.debug("fuse_attention: transpose_q input shall be Split output")
            return

        attention_last_node = reshape_qkv

        num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_before_split, add_before_split)
        if num_heads <= 0:
            logger.debug("fuse_attention: failed to detect num_heads")
            return
        attention_last_node = reshape_qkv

        # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
        new_node = self.create_attention_node(
            matmul_before_split,
            add_before_split,
            num_heads,
            hidden_size,
            input=root_input,
            relative_position_bias_input=add_bias_qk.input[1],
            output=attention_last_node.output[0],
        )
        if new_node is None:
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([attention_last_node, transpose_qkv])

        # Use prune graph to remove nodes since they are shared by all attention nodes.
        self.prune_graph = True
