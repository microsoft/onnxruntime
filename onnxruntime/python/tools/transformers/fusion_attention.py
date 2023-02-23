# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum
from logging import getLogger
from os import name
from sys import path
from typing import Tuple, Union

import numpy as np
from fusion_base import Fusion
from fusion_options import AttentionMaskFormat
from fusion_utils import FusionUtils, NumpyHelper
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel
from shape_infer_helper import SymbolicShapeInferenceHelper, get_shape_from_type_proto

logger = getLogger(__name__)


class AttentionMask:
    """
    Fuse Attention subgraph into one Attention node.
    """

    def __init__(self, model: OnnxModel):
        self.model = model
        # A lookup table with mask input as key, and mask index output as value
        self.mask_indice = {}
        # A lookup table with mask input as key, and cast (to int32) output as value
        self.mask_casted = {}
        self.utils = FusionUtils(model)
        self.mask_format = AttentionMaskFormat.MaskIndexEnd

    def set_mask_format(self, mask_format: AttentionMaskFormat):
        self.mask_format = mask_format

    def set_mask_indice(self, mask, mask_index):
        if mask in self.mask_indice:
            assert mask_index == self.mask_indice[mask]
        self.mask_indice[mask] = mask_index

    def get_first_mask(self):
        assert len(self.mask_indice) > 0
        return next(iter(self.mask_indice))

    def process_mask(self, input: str) -> str:
        if self.mask_format == AttentionMaskFormat.NoMask:
            return None

        if input in self.mask_indice:
            return self.mask_indice[input]

        # Add cast to convert int64 to int32
        if self.model.find_graph_input(input):
            casted, input_name = self.utils.cast_graph_input_to_int32(input)
        else:
            input_name, cast_node = self.utils.cast_input_to_int32(input)
            casted = True

        if casted:
            self.mask_casted[input] = input_name

        # Attention supports int32 attention mask (2D) since 1.4.0
        if self.mask_format == AttentionMaskFormat.AttentionMask:
            self.mask_indice[input] = input_name
            return input_name

        # Add a mask processing node to convert attention mask to mask index (1D)
        output_name = self.model.create_node_name("mask_index")
        mask_index_node = helper.make_node(
            "ReduceSum",
            inputs=[input_name],
            outputs=[output_name],
            name=self.model.create_node_name("ReduceSum", "MaskReduceSum"),
        )
        mask_index_node.attribute.extend([helper.make_attribute("axes", [1]), helper.make_attribute("keepdims", 0)])
        self.model.add_node(mask_index_node)

        self.mask_indice[input] = output_name
        return output_name


class FusionAttention(Fusion):
    """
    Fuse Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
        use_multi_head_attention: bool = False,
    ):
        attention_op_name = "MultiHeadAttention" if use_multi_head_attention else "Attention"
        super().__init__(model, attention_op_name, ["SkipLayerNormalization", "LayerNormalization"])
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_mask = attention_mask
        self.use_multi_head_attention = use_multi_head_attention
        self.mask_filter_value = None

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads_and_hidden_size_from_concat(self, concat: NodeProto) -> Tuple[int, int]:
        """
        Detect num_heads and hidden_size from Concat node in the following subgraph:

        SkipLayerNormalization or EmbedLayerNormalization
                        /        |
                     MatMul    Shape
                        |        |
                       Add     Gather(indices=0)
                        |        |
                        |      Unsqueeze
                        |        |
                        |     Concat (*, -1, 12, 64)
                        |     /
                       Reshape
                          |
                       Transpose
        """
        if len(concat.input) == 4:
            num_heads = self.model.get_constant_value(concat.input[2])
            head_size = self.model.get_constant_value(concat.input[3])
            if (
                isinstance(num_heads, np.ndarray)
                and num_heads.size == 1
                and isinstance(head_size, np.ndarray)
                and head_size.size == 1
            ):
                return num_heads[0], num_heads[0] * head_size[0]

        return self.num_heads, self.hidden_size

    def get_num_heads_and_hidden_size(self, reshape_q: NodeProto) -> Tuple[int, int]:
        """Detect num_heads and hidden_size from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q

        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """
        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        q_shape = self.model.get_initializer(reshape_q.input[1])
        if q_shape is None:
            concat = self.model.get_parent(reshape_q, 1)
            if concat is not None and concat.op_type == "Concat":
                return self.get_num_heads_and_hidden_size_from_concat(concat)
            logger.debug(f"{reshape_q.input[1]} is not initializer.")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        q_shape_value = NumpyHelper.to_array(q_shape)
        if len(q_shape_value) != 4 or (q_shape_value[2] <= 0 or q_shape_value[3] <= 0):
            logger.debug(f"q_shape_value={q_shape_value}. Expected value are like [0, 0, num_heads, head_size].")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        num_heads = q_shape_value[2]
        head_size = q_shape_value[3]
        hidden_size = num_heads * head_size

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

    def get_add_qk_str(self, add_qk: NodeProto):
        shape_infer = self.model.infer_runtime_shape(update=True)
        if shape_infer is None:
            return

        input_0_shape = shape_infer.get_edge_shape(add_qk.input[0])
        input_1_shape = shape_infer.get_edge_shape(add_qk.input[1])

        if input_0_shape is None or input_1_shape is None:
            logger.debug(f"one of the inputs of {add_qk} is None")
            return None

        if input_0_shape != input_1_shape:
            logger.debug(f"the shape of two inputs of {add_qk} is not same")
            return None

        return add_qk.input[1]

    def create_attention_node(
        self,
        mask_index: str,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        q_add: NodeProto,
        k_add: NodeProto,
        v_add: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
        add_qk_str: str,
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            mask_index (str): mask input
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for  K
            v_matmul (NodeProto): MatMul node in fully connection for  V
            q_add (NodeProto): Add bias node in fully connection for Q
            k_add (NodeProto): Add bias node in fully connection for K
            v_add (NodeProto): Add bias node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])
        q_bias = self.model.get_initializer(q_add.input[1]) or self.model.get_initializer(q_add.input[0])
        k_bias = self.model.get_initializer(k_add.input[1]) or self.model.get_initializer(k_add.input[0])
        v_bias = self.model.get_initializer(v_add.input[1]) or self.model.get_initializer(v_add.input[0])

        if q_weight is None:
            print(
                f"{q_matmul.input[1]} is not an initializer. "
                "Please set do_constant_folding=True in torch.onnx.export to unblock attention fusion"
            )
            return None
        if not (k_weight and v_weight and q_bias and k_bias):
            return None

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)

        # assert q and k have same shape as expected
        assert qw.shape == kw.shape

        qw_in_size = qw.shape[0]
        kw_in_size = kw.shape[0]
        vw_in_size = vw.shape[0]

        assert qw_in_size == kw_in_size == vw_in_size

        if hidden_size > 0 and hidden_size != qw_in_size:
            logger.warning(
                f"Input hidden size ({hidden_size}) is not same as weight matrix dimension of q,k,v ({qw_in_size}). "
                "Please provide a correct input hidden size or pass in 0"
            )

        is_qkv_diff_dims = False
        if qw.shape != vw.shape:
            is_qkv_diff_dims = True

        # All the matrices can have the same shape or q, k matrics can have the same shape with v being different
        # For 2d weights, the shapes would be [in_size, out_size].
        # For 3d weights, shape would be [in_size, a, b] where a*b = out_size
        qw_out_size = np.prod(qw.shape[1:])
        kw_out_size = np.prod(kw.shape[1:])
        vw_out_size = np.prod(vw.shape[1:])

        qkv_weight_dim = 0
        if is_qkv_diff_dims:
            qkv_weight = np.concatenate((qw, kw, vw), axis=1)
            qkv_weight_dim = qw_out_size + kw_out_size + vw_out_size
        else:
            qkv_weight = np.stack((qw, kw, vw), axis=1)
            qkv_weight_dim = 3 * qw_out_size

        qb = NumpyHelper.to_array(q_bias)
        kb = NumpyHelper.to_array(k_bias)
        vb = NumpyHelper.to_array(v_bias)

        q_bias_shape = np.prod(qb.shape)
        k_bias_shape = np.prod(kb.shape)
        v_bias_shape = np.prod(vb.shape)

        assert q_bias_shape == k_bias_shape == qw_out_size
        assert v_bias_shape == vw_out_size

        qkv_bias_dim = 0
        if is_qkv_diff_dims:
            qkv_bias = np.concatenate((qb, kb, vb), axis=0)
            qkv_bias_dim = q_bias_shape + k_bias_shape + v_bias_shape
        else:
            qkv_bias = np.stack((qb, kb, vb), axis=0)
            qkv_bias_dim = 3 * q_bias_shape

        attention_node_name = self.model.create_node_name("Attention")

        if not self.use_multi_head_attention:
            weight = helper.make_tensor(
                name=attention_node_name + "_qkv_weight",
                data_type=TensorProto.FLOAT,
                dims=[qw_in_size, qkv_weight_dim],
                vals=qkv_weight.flatten().tolist(),
            )

            # Sometimes weights and bias are stored in fp16
            if q_weight.data_type == 10:
                weight.CopyFrom(numpy_helper.from_array(NumpyHelper.to_array(weight).astype(np.float16), weight.name))
            self.model.add_initializer(weight, self.this_graph_name)

        bias = helper.make_tensor(
            name=attention_node_name + "_qkv_bias",
            data_type=TensorProto.FLOAT,
            dims=[qkv_bias_dim],
            vals=qkv_bias.flatten().tolist(),
        )
        if q_bias.data_type == 10:
            bias.CopyFrom(numpy_helper.from_array(NumpyHelper.to_array(bias).astype(np.float16), bias.name))
        self.model.add_initializer(bias, self.this_graph_name)

        # For MultiHeadAttention operator, use separated inputs for query, key and value, and no weights.
        if self.use_multi_head_attention:
            if add_qk_str is not None:
                logger.debug("MultiHeadAttention does not support relative_position_bias: cannot fuse the attention.")
                return None

            attention_inputs = [
                q_matmul.output[0],
                k_matmul.output[0],
                v_matmul.output[0],
                attention_node_name + "_qkv_bias",
            ]
            if mask_index is not None:
                attention_inputs.append(mask_index)

            attention_node = helper.make_node(
                "MultiHeadAttention",
                inputs=attention_inputs,
                outputs=[output],
                name=attention_node_name,
            )
        else:
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
                attention_inputs.append("")  # no past
                attention_inputs.append(add_qk_str)

            attention_node = helper.make_node(
                "Attention",
                inputs=attention_inputs,
                outputs=[output],
                name=attention_node_name,
            )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        if is_qkv_diff_dims:
            attention_node.attribute.extend(
                [helper.make_attribute("qkv_hidden_sizes", [qw_out_size, kw_out_size, vw_out_size])]
            )

        if self.mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = normalize_node
        if normalize_node.op_type == "LayerNormalization":
            add_before_layernorm = self.model.match_parent(normalize_node, "Add", 0)
            if add_before_layernorm is not None:
                start_node = add_before_layernorm
            else:
                return

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
            start_node,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [None, None, 0, 0, 0],
        )
        einsum_node = None
        if qkv_nodes is not None:
            (_, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes
        else:
            # Match Albert
            qkv_nodes = self.model.match_parent_path(
                start_node, ["Add", "Einsum", "Transpose", "MatMul"], [1, None, 0, 0]
            )
            if qkv_nodes is not None:
                (_, einsum_node, transpose_qkv, matmul_qkv) = qkv_nodes
            else:
                return

        other_inputs = []
        for i, input in enumerate(start_node.input):
            if input not in output_name_to_node:
                continue

            if input == qkv_nodes[0].output[0]:
                continue
            other_inputs.append(input)
        if len(other_inputs) != 1:
            return

        root_input = other_inputs[0]
        """
        Match flaubert                     Mask
                                            |
        Mul --> LayerNormalization -->  Attention --> MatMul --> Add
         |                                                        |
         |                                                        |
         +---------------------------------------------------------
        """
        mul_before_layernorm = self.model.match_parent(start_node, "Mul", 0)
        if mul_before_layernorm is not None:
            mul_children = input_name_to_nodes[mul_before_layernorm.output[0]]
            if mul_children is not None and len(mul_children) == 2:
                layernorm_node = mul_children[1]
                if layernorm_node.op_type == "LayerNormalization":
                    root_input = layernorm_node.output[0]
                else:
                    return
            elif mul_children is not None and len(mul_children) == 5:
                root_input = mul_before_layernorm.output[0]
            else:
                return
        elif normalize_node.op_type == "LayerNormalization":
            children = input_name_to_nodes[root_input]
            for child in children:
                if child.op_type == "LayerNormalization":
                    root_input = child.output[0]

        children = input_name_to_nodes[root_input]
        children_types = [child.op_type for child in children]
        if children_types.count("MatMul") != 3:
            return

        v_nodes = self.model.match_parent_path(matmul_qkv, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (_, _, add_v, matmul_v) = v_nodes

        is_distill = False
        is_distill_add = False
        qk_paths = {
            "path1": (["Softmax", "Add", "Div", "MatMul"], [0, 0, None, 0]),
            "path2": (["Softmax", "Add", "Mul", "MatMul"], [0, 0, None, 0]),
            "path3": (["Softmax", "Where", "MatMul", "Div"], [0, 0, 2, 0]),
            "path4": (["Softmax", "Add", "Where", "MatMul"], [0, 0, 0, 2]),
        }

        qk_nodes = None
        for k, v in qk_paths.items():
            qk_nodes = self.model.match_parent_path(matmul_qkv, v[0], v[1])
            if qk_nodes is None:
                continue
            if k == "path3":
                is_distill = True
            if k == "path4":
                is_distill_add = True
            break

        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return

        add_qk = None
        matmul_qk = None
        where_qk = None
        if is_distill:
            (_, where_qk, matmul_qk, _) = qk_nodes
        elif is_distill_add:
            (_, add_qk, where_qk, matmul_qk) = qk_nodes
        else:
            (_, add_qk, _, matmul_qk) = qk_nodes

        q_nodes = self.model.match_parent_path(matmul_qk, ["Transpose", "Reshape", "Add", "MatMul"], [0, 0, 0, None])
        if q_nodes is None:
            q_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Div", "Transpose", "Reshape", "Add", "MatMul"],
                [0, 0, 0, 0, None],
            )
            if q_nodes is None:
                logger.debug("fuse_attention: failed to match q path")
                return
        reshape_q = q_nodes[-3]
        add_q = q_nodes[-2]
        matmul_q = q_nodes[-1]

        k_nodes = self.model.match_parent_path(matmul_qk, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None])
        if k_nodes is None:
            k_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Transpose", "Transpose", "Reshape", "Add", "MatMul"],
                [1, 0, 0, 0, None],
            )
            if k_nodes is None:
                logger.debug("fuse_attention: failed to match k path")
                return
        add_k = k_nodes[-2]
        matmul_k = k_nodes[-1]

        # Note that Cast might be removed by OnnxRuntime so we match two patterns here.
        mask_nodes = None
        add_qk_str = None
        if is_distill:
            _, mask_nodes, _ = self.model.match_parent_paths(
                where_qk,
                [
                    (["Expand", "Reshape", "Equal"], [0, 0, 0]),
                    (["Equal", "Unsqueeze", "Unsqueeze"], [0, 0, 0]),
                    (["Cast", "Expand", "Reshape", "Equal"], [0, 0, 0, 0]),
                ],
                output_name_to_node,
            )
        elif is_distill_add:
            _, mask_nodes, _ = self.model.match_parent_paths(
                where_qk,
                [
                    (["Cast", "Equal", "Unsqueeze", "Unsqueeze"], [0, 0, 0, 0]),
                    (["Equal", "Unsqueeze", "Unsqueeze"], [0, 0, 0]),
                ],
                output_name_to_node,
            )
            if add_qk is not None:
                add_qk_str = self.get_add_qk_str(add_qk)
                if add_qk_str is None:
                    logger.debug(f"fuse_attention: failed to verify shape inference of {add_qk}")
                    return
        else:
            _, mask_nodes, _ = self.model.match_parent_paths(
                add_qk,
                [
                    (
                        ["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
                        [None, 0, 1, 0, 0],
                    ),
                    (["Mul", "Sub", "Unsqueeze", "Unsqueeze"], [None, 0, 1, 0]),
                ],
                output_name_to_node,
            )
        if mask_nodes is None:
            logger.debug("fuse_attention: failed to match mask path")
            return

        if len(mask_nodes) > 1 and mask_nodes[0].op_type == "Mul":
            _, mul_val = self.model.get_constant_input(mask_nodes[0])
            if mul_val != -10000:
                self.mask_filter_value = mul_val

        if matmul_v.input[0] == root_input and matmul_q.input[0] == root_input and matmul_k.input[0] == root_input:
            mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])

            attention_last_node = reshape_qkv if einsum_node is None else transpose_qkv

            q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q)
            # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
            # the input_hidden_size represents the input hidden size, this is used as needed but hidden sizes for Q, K are extracted appropriately
            new_node = self.create_attention_node(
                mask_index,
                matmul_q,
                matmul_k,
                matmul_v,
                add_q,
                add_k,
                add_v,
                q_num_heads,
                q_hidden_size,
                root_input,
                attention_last_node.output[0],
                add_qk_str,
            )
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            if einsum_node is not None:
                unique_index = einsum_node.input[0]
                new_edge = "edge_modified_" + unique_index
                shape_tensor = helper.make_tensor(
                    name="shape_modified_tensor" + unique_index,
                    data_type=TensorProto.INT64,
                    dims=[4],
                    vals=np.int64([0, 0, q_num_heads, int(q_hidden_size / q_num_heads)]).tobytes(),
                    raw=True,
                )
                self.model.add_initializer(shape_tensor, self.this_graph_name)
                self.model.add_node(
                    helper.make_node(
                        "Reshape",
                        [attention_last_node.output[0], shape_tensor.name],
                        [new_edge],
                        "reshape_modified_" + unique_index,
                    ),
                    self.this_graph_name,
                )
                einsum_node.input[0] = new_edge

            self.nodes_to_remove.extend([attention_last_node, transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)

            # For MultiHeadAttention operator, MatMul nodes for Q/K/V projection shall not be fused.
            self.nodes_to_remove.extend(q_nodes if not self.use_multi_head_attention else q_nodes[:-1])
            self.nodes_to_remove.extend(k_nodes if not self.use_multi_head_attention else k_nodes[:-1])
            self.nodes_to_remove.extend(v_nodes if not self.use_multi_head_attention else v_nodes[:-1])

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            self.prune_graph = True
