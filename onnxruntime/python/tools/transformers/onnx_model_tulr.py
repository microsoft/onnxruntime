# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Union

import numpy as np
from fusion_attention import AttentionMask, FusionAttention
from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)


class FusionTulrAttention(Fusion):
    """
    Fuse TNLR Attention subgraph into one Attention node.
    TNLR Attention has extra addtion after qk nodes and adopts [S, B, NH] as I/O shape.
    """

    def __init__(self, model: OnnxModel):
        super().__init__(model, "MultiHeadAttention", "MultiHeadAttention")

    def update_attention_node(
        self, mha: NodeProto, q_matmul: NodeProto, k_matmul: NodeProto, v_matmul: NodeProto, num_heads=64
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for K
            v_matmul (NodeProto): MatMul node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        if q_matmul.input[0] != k_matmul.input[0] or v_matmul.input[0] != k_matmul.input[0]:
            logger.debug(
                "For self attention, input hidden state for q and k/v shall be same. Got %s, %s, %s",
                q_matmul.input[0],
                k_matmul.input[0],
                v_matmul.input[0],
            )
            return None

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])
        if not (q_weight and k_weight and v_weight):
            return None

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)
        logger.debug(f"qw={qw.shape} kw={kw.shape} vw={vw.shape}")

        # assert q and k have same shape as expected
        if qw.shape != kw.shape or qw.shape != vw.shape:
            return None

        qw_in_size = qw.shape[0]

        # All the matrices can have the same shape or q, k matrics can have the same shape with v being different
        # For 2d weights, the shapes would be [in_size, out_size].
        # For 3d weights, shape would be [in_size, a, b] where a*b = out_size
        qw_out_size = int(np.prod(qw.shape[1:]))

        attention_node_name = self.model.create_node_name("MultiHeadAttention")

        c = qw_in_size
        n = num_heads
        h = qw_out_size // num_heads

        # Concat and interleave weights so that the output of fused KV GEMM has [B, S_kv, N, 3, H] shape
        qkv_weight = np.dstack([qw.reshape(c, n, h), kw.reshape(c, n, h), vw.reshape(c, n, h)]).reshape(c, n * 3 * h)

        matmul_node_name = self.model.create_node_name("MatMul", name_prefix="MatMul_QKV")
        weight = helper.make_tensor(
            name=matmul_node_name + "_weight",
            data_type=TensorProto.FLOAT16,  # TODO: get data type from q weights
            dims=[qkv_weight.shape[0], qkv_weight.shape[1]],
            vals=qkv_weight.flatten().tolist(),
        )

        self.model.add_initializer(weight, self.this_graph_name)

        matmul_node = helper.make_node(
            "MatMul",
            inputs=[k_matmul.input[0], matmul_node_name + "_weight"],
            outputs=[matmul_node_name + "_out"],
            name=matmul_node_name,
        )
        self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name

        shape_tensor = helper.make_tensor(
            name=matmul_node_name + "_reshape_shape",
            data_type=TensorProto.INT64,
            dims=[5],
            vals=[0, 0, n, 3, h],
        )
        self.model.add_initializer(shape_tensor, self.this_graph_name)

        reshape_node = helper.make_node(
            "Reshape",
            inputs=[matmul_node_name + "_out", matmul_node_name + "_reshape_shape"],
            outputs=[attention_node_name + "_input"],
            name=matmul_node_name + "_reshape",
        )
        self.node_name_to_graph_name[reshape_node.name] = self.this_graph_name
        self.nodes_to_add.extend([matmul_node, reshape_node])
        # self.nodes_to_remove.extend([q_matmul, k_matmul, v_matmul])

        mha.input[0] = attention_node_name + "_input"
        mha.input[1] = ""
        mha.input[2] = ""

        counter_name = "MultiHeadAttention ({})".format("self attention with packed qkv")
        self.increase_counter(counter_name)
        return mha

    def fuse(self, att_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        q_matmul = self.model.get_parent(att_node, 0)
        k_matmul = self.model.get_parent(att_node, 1)
        v_matmul = self.model.get_parent(att_node, 2)

        if (
            q_matmul
            and q_matmul.op_type == "MatMul"
            and k_matmul
            and k_matmul.op_type == "MatMul"
            and v_matmul
            and v_matmul.op_type == "MatMul"
        ):
            if self.update_attention_node(att_node, q_matmul, k_matmul, v_matmul):
                # Use prune graph to remove mask nodes since they are shared by all attention nodes.
                # self.nodes_to_remove.extend(mask_nodes)
                self.prune_graph = True


class TulrOnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        self.attention_fusion = FusionTulrAttention(self)

    def fuse_attention(self):
        self.attention_fusion.apply()
