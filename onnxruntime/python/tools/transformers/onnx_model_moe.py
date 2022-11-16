# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Union

from fusion_attention import AttentionMask, FusionAttention
from fusion_utils import NumpyHelper
from fusion_base import Fusion
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)

#python optimizer.py --input your_path_to_input_model --output your_path_to_output_model --model_type moe --hidden_size 768 --num_heads 12 --float16 --use_external_data_format --opt_level 0


class FusionMoEAttention(FusionAttention):
    """
    Fuse MOE Attention subgraph into one Attention node.
    """
    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
    ):
        super().__init__(model, hidden_size, num_heads, attention_mask)

    def create_attention_node(
        self,
        mask_index: str,
        matmul: NodeProto,
        add: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
        add_qk_str: str,
    ) -> Union[NodeProto, None]:

        assert num_heads > 0
        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(
                f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}"
            )
            return None

        weight = self.model.get_initializer(matmul.input[1])
        bias = self.model.get_initializer(
            add.input[1]) or self.model.get_initializer(add.input[0])

        if weight is None or bias is None:
            return None

        qkv_weight = NumpyHelper.to_array(weight)
        qkv_bias = NumpyHelper.to_array(bias)

        attention_node_name = self.model.create_node_name("Attention")

        weight = helper.make_tensor(
            name=attention_node_name + "_qkv_weight",
            data_type=TensorProto.FLOAT,
            dims=[hidden_size, 3 * hidden_size],
            vals=qkv_weight.flatten().tolist(),
        )

        # Sometimes weights and bias are stored in fp16
        if weight.data_type == 10:
            weight.CopyFrom(
                numpy_helper.from_array(
                    NumpyHelper.to_array(weight).astype(np.float16),
                    weight.name))
        self.model.add_initializer(weight, self.this_graph_name)

        bias = helper.make_tensor(
            name=attention_node_name + "_qkv_bias",
            data_type=TensorProto.FLOAT,
            dims=[3 * hidden_size],
            vals=qkv_bias.flatten().tolist(),
        )
        if bias.data_type == 10:
            bias.CopyFrom(
                numpy_helper.from_array(
                    NumpyHelper.to_array(bias).astype(np.float16), bias.name))
        self.model.add_initializer(bias, self.this_graph_name)

        attention_inputs = [
            input,
            attention_node_name + "_qkv_weight",
            attention_node_name + "_qkv_bias",
        ]
        if mask_index is not None:
            attention_inputs.append(mask_index)
        else:
            attention_inputs.append("")

        if add_qk_str is not None:
            attention_inputs.append("")
            attention_inputs.append(add_qk_str)

        attention_node = helper.make_node(
            "Attention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend(
            [helper.make_attribute("num_heads", num_heads)])

        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = normalize_node
        if normalize_node.op_type != "LayerNormalization":
            return

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
            start_node,
            ["Add", "Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [0, 1, 1, 0, 0, 0],
        )
        if qkv_nodes is not None:
            (_, _, matmul_below, reshape_qkv, transpose_qkv,
             matmul_qkv) = qkv_nodes
        else:
            return

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Gather", "Transpose"],
            [1, 0],
        )
        if v_nodes is None:
            return
        (_, transpose) = v_nodes

        q_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Softmax", "MatMul", "Mul", "Gather", "Transpose"],
            [0, 0, 0, 0, 0],
        )
        if q_nodes is None:
            return

        k_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Softmax", "MatMul", "Transpose", "Gather", "Transpose"],
            [0, 0, 1, 0, 0],
        )
        if k_nodes is None:
            return

        main_matmul_nodes = self.model.match_parent_path(
            transpose,
            ["Reshape", "Add", "MatMul"],
            [0, 0, 1],
        )
        if main_matmul_nodes is None:
            return
        (_, add, matmul) = main_matmul_nodes

        mask_index = None
        attention_last_node = reshape_qkv
        new_node = self.create_attention_node(
            mask_index,
            matmul,
            add,
            self.num_heads,
            self.hidden_size,
            matmul.input[0],
            attention_last_node.output[0],
            None,
        )
        if new_node is None:
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend(
            [attention_last_node, transpose_qkv, matmul_qkv])
        self.nodes_to_remove.extend(main_matmul_nodes)
        self.nodes_to_remove.extend(q_nodes)
        self.nodes_to_remove.extend(k_nodes)
        self.nodes_to_remove.extend(v_nodes)

        # Use prune graph to remove mask nodes since they are shared by all attention nodes.
        # self.nodes_to_remove.extend(mask_nodes)
        self.prune_graph = True


class FusionMoEBlock(Fusion):
    def __init__(self, model: OnnxModel, hidden_size: int):
        super().__init__(model, "MoEBlock", "Add")
        self.hidden_size = hidden_size

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if node.op_type != "Add":
            return

        mul_nodes = self.model.match_parent_path(
            node,
            ["Reshape", "Mul"],
            [1, 0],
        )
        if mul_nodes is None:
            return
        reshape_1, mul_0 = mul_nodes

        gate_top1_nodes = self.model.match_parent_path(
            mul_0,
            [
                "Unsqueeze", "Unsqueeze", "ReduceMax", "Softmax", "MatMul",
                "Reshape"
            ],
            [1, 0, 0, 0, 0, 0],
        )
        if gate_top1_nodes is None:
            return
        matmul_0 = gate_top1_nodes[-2]
        reshape_0 = gate_top1_nodes[-1]

        ln_node = self.model.match_parent(
            reshape_0,
            "LayerNormalization",
            0,
        )
        if ln_node is None:
            return

        matmul_2_nodes = self.model.match_parent_path(
            mul_0,
            ["MatMul"],
            [0],
        )
        if matmul_2_nodes is None:
            return
        matmul_2 = matmul_2_nodes[0]

        expert_0_nodes = self.model.match_parent_path(
            matmul_2,
            ["Gelu", "MatMul", "Gather"],
            [0, 0, 1],
        )
        if expert_0_nodes is None:
            return
        _, matmul_1, gather_0 = expert_0_nodes

        shape_nodes = self.model.match_parent_path(
            matmul_1,
            ["Unsqueeze"],
            [0],
        )
        if shape_nodes is None:
            return

        expert_1_nodes = self.model.match_parent_path(
            matmul_2,
            ["Gather", "ArgMax"],
            [1, 1],
        )
        if expert_1_nodes is None:
            return
        gather_1 = expert_1_nodes[0]

        self.nodes_to_remove.extend(mul_nodes)
        self.nodes_to_remove.extend(gate_top1_nodes)
        self.nodes_to_remove.extend(matmul_2_nodes)
        self.nodes_to_remove.extend(expert_0_nodes)
        self.nodes_to_remove.extend(shape_nodes)
        self.nodes_to_remove.extend(expert_1_nodes)

        fused_node = helper.make_node(
            "MoEBlock",
            [
                reshape_0.input[0], matmul_0.input[1], gather_0.input[0],
                gather_1.input[0]
            ],
            [reshape_1.output[0]],
            name=self.model.create_node_name("MoEBlock"),
        )
        fused_node.domain = "com.microsoft"

        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name


class MOEOnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionMoEAttention(self, self.hidden_size,
                                                   self.num_heads,
                                                   self.attention_mask)
        self.moe_fusion = FusionMoEBlock(self, hidden_size)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def postprocess(self):
        self.moe_fusion.apply()
        self.clean_graph()
        self.prune_graph()
