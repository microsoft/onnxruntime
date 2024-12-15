# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from typing import Optional

import numpy as np
from fusion_base import Fusion
from fusion_utils import FusionUtils
from onnx import NodeProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionMultiHeadAttentionMMDit(Fusion):
    """
    Fuse MultiHeadAttention for Multimodal Diffusion Transformer (MMDiT).
    """

    def __init__(self, model: OnnxModel):
        super().__init__(model, fused_op_type="MultiHeadAttention", search_op_types=["Softmax"])

    def get_num_heads(self, node: NodeProto, output_name_to_node) -> int:
        """
        Detect num_heads and hidden_size from Concat node in the following subgraph:
                              MatMul      .. [-1] [24] ..
                                 |         |  |  /   /
                              Add<1536>  Concat
                                 |      /
                                 Reshape
                                  |
                               Transpose(perm=0,2,1,3)
                                  |
                       SimplifiedLayerNorm -- scale<64>
                                  |
                               (node)

        The num_heads can be read directly from the third input of Concat node.

        Here we deduce num_heads=hidden_size/head_size from the following two nodes:
        The hidden_size can be read from the bias input of Add node
        The head_size can be read from the scale input of SimplifiedLayerNormalization node
        """
        k_proj_nodes = self.model.match_parent_path(
            node,
            ["SimplifiedLayerNormalization", "Transpose", "Reshape", "Add"],
            [0, 0, 0, 0],
            output_name_to_node=output_name_to_node,
        )

        num_heads = 0
        if k_proj_nodes:
            simplified_layernorm, _transpose, _reshape, add = k_proj_nodes
            _i, bias = self.model.get_constant_input(add)

            hidden_size = 0
            if isinstance(bias, np.ndarray) and len(bias.shape) == 1:
                hidden_size = bias.shape[0]

            weight = self.model.get_constant_value(simplified_layernorm.input[1])
            if isinstance(weight, np.ndarray) and len(weight.shape) == 1:
                head_size = weight.shape[0]
                if (hidden_size % head_size) == 0:
                    num_heads = hidden_size // head_size

        return num_heads

    def reshape_to_3d(self, input_name: str, output_name: str) -> str:
        # Add a shape to convert 4D BxSxNxH to 3D BxSxD, which is required by MHA operator.
        new_dims_name = "bsnh_to_bsd_reshape_dims"
        new_dims = self.model.get_initializer(new_dims_name)
        if new_dims is None:
            new_dims = numpy_helper.from_array(np.array([0, 0, -1], dtype="int64"), name=new_dims_name)
            self.model.add_initializer(new_dims, self.this_graph_name)
        reshape_q = helper.make_node(
            "Reshape",
            inputs=[input_name, new_dims_name],
            outputs=[output_name],
            name=self.model.create_node_name("Reshape"),
        )
        self.nodes_to_add.append(reshape_q)
        self.node_name_to_graph_name[reshape_q.name] = self.this_graph_name
        return reshape_q.output[0]

    def get_num_heads_with_concat(self, transpose_k: NodeProto, output_name_to_node) -> int:
        """
        Detect num_heads and hidden_size from Concat node in the following subgraph:

                        /        |
                     MatMul    MatMul     .. [-1] [24] ..
                        |        |         |  |  /   /
                       Add      Add<1536>  Concat
                        |          |      /
                     Reshape       Reshape
                        |           |
                    Transpose  Transpose(perm=0,2,1,3)
                        |           |
        SimplifiedLayerNorm  SimplifiedLayerNorm -- scale<64>
                        |     /
                       Concat
                         |
                    Transpose(perm=0,1,3,2)
        """
        nodes = self.model.match_parent_path(transpose_k, ["Concat"], [0], output_name_to_node=output_name_to_node)

        return self.get_num_heads(nodes[0], output_name_to_node) if nodes else 0

    def adjust_query_from_bnsh_to_bsd_no_concat(self, mul_q: NodeProto, output_name_to_node) -> Optional[str]:
        """
        Before:
                               MatMul      .. [-1] [24] ..
                                 |         |  |  /   /
                               Add       Concat
                                 |      /
                                 Reshape
                                  |
                               Transpose(perm=0,2,1,3)
                                  |
                       SimplifiedLayerNorm
                                  |
                                 Mul

        After:
                               MatMul    .. [-1] [24] ..
                                 |       |  |  /   /
                               Add       Concat
                                 |      /
                                 Reshape
                                   |
                           SimplifiedLayerNorm
                                   |
                        Reshape (shape=[0, 0, -1])
        """

        path = self.model.match_parent_path(
            mul_q,
            ["SimplifiedLayerNormalization", "Transpose"],
            [0, 0],
        )
        if path is None:
            return None
        sln_a, transpose_a = path

        if not FusionUtils.check_node_attribute(transpose_a, "perm", [0, 2, 1, 3]):
            return None

        # Update the graph
        sln_a.input[0] = transpose_a.input[0]
        sln_output = sln_a.output[0]
        sln_a.output[0] = sln_output + "_BSNH"

        return self.reshape_to_3d(sln_a.output[0], sln_output + "_BSD")

    def adjust_query_from_bnsh_to_bsd(self, mul_q: NodeProto, output_name_to_node) -> Optional[str]:
        """
            Before:
                      MatMul      MatMul    .. [-1] [24] ..
                        |            |       |  |  /   /
                        Add Concat  Add    Concat
                         |    /      |      /
                         Reshape     Reshape
                            |           |
        Transpose(perm=0,2,1,3)      Transpose(perm=0,2,1,3)
                            |           |
            SimplifiedLayerNorm  SimplifiedLayerNorm
                            |     /
                            Concat(axis=2)
                             |
                            Mul

            After:
                      MatMul      MatMul    .. [-1] [24] ..
                        |            |       |  |  /   /
                        Add Concat Add       Concat
                         |    /      |      /
                         Reshape     Reshape
                            |           |
                SimplifiedLayerNorm  SimplifiedLayerNorm
                            |         /
                        Concat(axis=1)
                             |
                           Reshape (shape=[0, 0, -1])
        """

        path = self.model.match_parent_path(
            mul_q,
            ["Concat", "SimplifiedLayerNormalization", "Transpose"],
            [0, 0, 0],
        )
        if path is None:
            return None
        concat, sln_a, transpose_a = path

        if len(concat.input) != 2:
            return None

        path = self.model.match_parent_path(
            concat,
            ["SimplifiedLayerNormalization", "Transpose"],
            [1, 0],
        )
        if path is None:
            return None
        sln_b, transpose_b = path

        if not FusionUtils.check_node_attribute(transpose_a, "perm", [0, 2, 1, 3]):
            return None

        if not FusionUtils.check_node_attribute(transpose_b, "perm", [0, 2, 1, 3]):
            return None

        # Update the graph
        sln_a.input[0] = transpose_a.input[0]
        sln_b.input[0] = transpose_b.input[0]

        new_concat_node = helper.make_node(
            "Concat",
            inputs=[sln_a.output[0], sln_b.output[0]],
            outputs=[concat.output[0] + "_BSNH"],
            name=self.model.create_node_name("Concat"),
            axis=1,
        )
        self.nodes_to_add.append(new_concat_node)
        self.node_name_to_graph_name[new_concat_node.name] = self.this_graph_name

        return self.reshape_to_3d(new_concat_node.output[0], concat.output[0] + "_BSD")

    def create_multihead_attention_node(
        self,
        q: str,
        k: str,
        v: str,
        output: str,
        num_heads: int,
    ) -> NodeProto:
        """
        Create a MultiHeadAttention node.

        Args:
            q (str): name of q
            k (str): name of k
            v (str): name of v
            output (str): output name of MHA
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.

        Returns:
            NodeProto: the node created.
        """

        assert num_heads > 0

        # Add inputs for MHA: Query, Key, Value (Proj_Bias, Mask, Attention_Bias, Past_K, Past_V are optional)
        mha_inputs = [q, k, v]

        # Add outputs for MHA (Present_K, Present_V are optional)
        mha_outputs = [output]

        mha_node = helper.make_node(
            "MultiHeadAttention",
            inputs=mha_inputs,
            outputs=mha_outputs,
            name=self.model.create_node_name("MultiHeadAttention"),
        )

        mha_node.domain = "com.microsoft"
        mha_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        # No mask is used in MMDit model, so we need not set the optional mask_filter_value attribute.
        return mha_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        assert node.op_type == "Softmax"
        softmax = node

        # Softmax output shall not be graph output.
        if self.model.find_graph_output(softmax.output[0]):
            return

        nodes = self.model.match_child_path(
            softmax, ["MatMul", "Transpose", "Reshape"], [(0, 0), (0, 0), (0, 0)], input_name_to_nodes
        )
        if nodes is None:
            return

        matmul_s_v, transpose_out, reshape_out = nodes
        if not FusionUtils.check_node_attribute(transpose_out, "perm", [0, 2, 1, 3]):
            return

        q_nodes = self.model.match_parent_path(
            softmax,
            ["MatMul", "Mul", "Sqrt", "Div", "Sqrt", "Cast", "Slice", "Shape"],
            [0, 0, 1, 0, 1, 0, 0, 0],
        )

        if q_nodes is None:
            return

        matmul_qk, mul_q, sqrt_q_2, div_q, sqrt_q, _, _, shape_q = q_nodes

        if mul_q.input[0] != shape_q.input[0]:
            return

        k_nodes = self.model.match_parent_path(matmul_qk, ["Mul", "Transpose"], [1, 0])
        if k_nodes is None:
            return

        mul_k, transpose_k = k_nodes
        k = transpose_k.input[0]
        if not FusionUtils.check_node_attribute(transpose_k, "perm", [0, 1, 3, 2]):
            return

        k_scale_nodes = self.model.match_parent_path(mul_k, ["Sqrt", "Div"], [1, 0])
        if k_scale_nodes is None:
            return
        if k_scale_nodes[0].input[0] != sqrt_q_2.input[0]:
            return

        v = matmul_s_v.input[1]

        # Here we sanity check the v path to make sure it is in the expected BNSH format.
        concat = self.model.match_parent(matmul_s_v, "Concat", input_index=1, output_name_to_node=output_name_to_node)
        if concat is not None:
            # Match v path like:
            #   -- Transpose (perm=[0,2,1,3]) ----+
            #                                     |
            #                                     v
            #   -- Transpose (perm=[0,2,1,3]) -> Concat -> (v)
            transpose_1 = self.model.match_parent(
                concat, "Transpose", input_index=0, output_name_to_node=output_name_to_node
            )
            if transpose_1 is None:
                return
            if not FusionUtils.check_node_attribute(transpose_1, "perm", [0, 2, 1, 3]):
                return

            transpose_2 = self.model.match_parent(
                concat, "Transpose", input_index=1, output_name_to_node=output_name_to_node
            )
            if transpose_2 is None:
                return
            if not FusionUtils.check_node_attribute(transpose_2, "perm", [0, 2, 1, 3]):
                return
        else:
            # Match v path like:
            #   -- Transpose (perm=[0,2,1,3]) -> (v)
            transpose_1 = self.model.match_parent(
                matmul_s_v, "Transpose", input_index=1, output_name_to_node=output_name_to_node
            )
            if transpose_1 is None:
                return
            if not FusionUtils.check_node_attribute(transpose_1, "perm", [0, 2, 1, 3]):
                return

        if concat is not None:
            num_heads = self.get_num_heads_with_concat(transpose_k, output_name_to_node)
        else:
            num_heads = self.get_num_heads(transpose_k, output_name_to_node)
        if num_heads <= 0:
            return

        # Q is in BNSH format, we need to adjust it to BSD format due to limitation of MHA op.
        if concat is not None:
            query = self.adjust_query_from_bnsh_to_bsd(mul_q, output_name_to_node)
        else:
            query = self.adjust_query_from_bnsh_to_bsd_no_concat(mul_q, output_name_to_node)

        if query is None:
            return

        new_node = self.create_multihead_attention_node(
            q=query,
            k=k,
            v=v,
            output=reshape_out.output[0],
            num_heads=num_heads,
        )
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([matmul_s_v, transpose_out, reshape_out])

        # Use prune graph to remove nodes
        self.prune_graph = True
