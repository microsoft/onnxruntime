# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Optional, Union

from fusion_attention import FusionAttention
from fusion_base import Fusion
from fusion_simplified_layernorm import FusionSimplifiedLayerNormalization, FusionSkipSimplifiedLayerNormalization
from onnx import FunctionProto, NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = logging.getLogger(__name__)


class FusionRotaryAttention(FusionAttention):
    """
    Fuse Attention subgraph with rotary positional embeddings into one MultiHeadAttention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__(
            model,
            hidden_size,
            num_heads,
            use_multi_head_attention=True,
            search_op_types=["SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "Add"],
        )

    def create_mha_node(
        self,
        input: str,
        output: str,
        q_rotary: NodeProto,
        k_rotary: NodeProto,
        v_matmul: NodeProto,
        attn_mask: str,
        past_k: str = "",
        past_v: str = "",
        present_k: str = "",
        present_v: str = "",
        scale: Optional[float] = None,
    ) -> Union[NodeProto, None]:
        assert self.num_heads > 0

        if self.hidden_size > 0 and (self.hidden_size % self.num_heads) != 0:
            logger.debug(
                f"fuse_rotary_attention: input hidden size {self.hidden_size} is not a multiple of num of heads {self.num_heads}"
            )
            return None

        mha_node_name = self.model.create_node_name("MultiHeadAttention")
        mha_inputs = [
            q_rotary.output[0],
            k_rotary.output[0],
            v_matmul.output[0],
            "",  # bias
            attn_mask,  # key_padding_mask
            "",  # relative_position_bias
            past_k,
            past_v,
        ]

        mha_outputs = [output]
        if present_k and present_v:
            mha_outputs.extend([present_k, present_v])

        mha_node = helper.make_node(
            "MultiHeadAttention",
            inputs=mha_inputs,
            outputs=mha_outputs,
            name=mha_node_name,
        )

        mha_node.domain = "com.microsoft"
        mha_node.attribute.extend([helper.make_attribute("num_heads", self.num_heads)])
        if scale is not None:
            mha_node.attribute.extend([helper.make_attribute("scale", scale)])
        if self.mask_filter_value is not None:
            mha_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        self.increase_counter("MultiHeadAttention")
        return mha_node

    def check_runtime_shape_paths(
        self,
        reshape_qkv_2,  # Reshape after Transpose
        reshape_qkv_1,  # Reshape before Transpose
        reshape_q_2,  # Reshape after RotaryEmbedding
        reshape_k_2,  # Reshape after RotaryEmbedding
        reshape_v_2,  # Reshape after Transpose
        reshape_v_1,  # Reshape before Transpose
        add_qk,  # Add before Softmax
        root_input,  # Root input to attention subgraph
    ):
        # Check #1: check paths for qkv nodes
        concat_qkv_2_path = self.model.match_parent_path(reshape_qkv_2, ["Concat"], [1])
        concat_qkv_1_path = self.model.match_parent_path(reshape_qkv_1, ["Concat"], [1])
        if concat_qkv_2_path is None or concat_qkv_1_path is None:
            return False
        concat_qkv_2, concat_qkv_1 = concat_qkv_2_path[0], concat_qkv_1_path[0]

        reshape_qkv_2_path_1 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_qkv_2_path_2 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        reshape_qkv_1_path_1 = self.model.match_parent_path(concat_qkv_1, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_qkv_1_path_2 = self.model.match_parent_path(concat_qkv_1, ["Unsqueeze", "Gather", "Shape"], [2, 0, 0])
        if (
            reshape_qkv_2_path_1 is None
            or reshape_qkv_2_path_2 is None
            or reshape_qkv_1_path_1 is None
            or reshape_qkv_1_path_2 is None
        ):
            return False

        _, gather_1, shape_1 = reshape_qkv_2_path_1
        _, gather_2, shape_2 = reshape_qkv_2_path_2

        # Check root_input --> Shape --> Gather connection
        if shape_1.input[0] != root_input or shape_2.input[0] != root_input:
            return False

        # Check Gather --> Unsqueeze --> Concat --> Reshape connection for reshape_qkv_1_path_1 and reshape_qkv_1_path_2
        if reshape_qkv_1_path_1[1].name != gather_1.name or reshape_qkv_1_path_2[1].name != gather_2.name:
            return False

        # Check #2: check paths for v nodes
        concat_v_2_path = self.model.match_parent_path(reshape_v_2, ["Concat"], [1])
        concat_v_1_path = self.model.match_parent_path(reshape_v_1, ["Concat"], [1])
        if concat_v_2_path is None or concat_v_1_path is None:
            return False
        concat_v_2, concat_v_1 = concat_v_2_path[0], concat_v_1_path[0]

        reshape_v_2_path_1 = self.model.match_parent_path(
            concat_v_2, ["Unsqueeze", "Mul", "Gather", "Shape"], [0, 0, 0, 0]
        )
        reshape_v_2_path_2 = self.model.match_parent_path(
            concat_v_2, ["Unsqueeze", "Add", "Gather", "Shape"], [1, 0, 0, 0]
        )
        reshape_v_1_path_1 = self.model.match_parent_path(concat_v_1, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_v_1_path_2 = self.model.match_parent_path(concat_v_1, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        if (
            reshape_v_2_path_1 is None
            or reshape_v_2_path_2 is None
            or reshape_v_1_path_1 is None
            or reshape_v_1_path_2 is None
        ):
            return False

        # Check Gather --> Mul --> Unsqueeze --> Concat --> Reshape connection for reshape_v_2_path_1
        # Check Gather --> Add --> Unsqueeze --> Concat --> Reshape connection for reshape_v_2_path_2
        # Check Gather --> Unsqueeze --> Concat --> Reshape connection for reshape_v_1_path_1 and reshape_v_1_path_2
        if (
            reshape_v_2_path_1[2].name != gather_1.name
            or reshape_v_2_path_2[2].name != gather_2.name
            or reshape_v_1_path_1[1].name != gather_1.name
            or reshape_v_1_path_2[1].name != gather_2.name
        ):
            return False

        # Check #3: check paths for k nodes
        concat_k_2_path = self.model.match_parent_path(reshape_k_2, ["Concat"], [1])
        if concat_k_2_path is None:
            return False
        concat_k_2 = concat_k_2_path[0]

        reshape_k_2_path_1 = self.model.match_parent_path(
            concat_k_2, ["Unsqueeze", "Mul", "Gather", "Shape"], [0, 0, 0, 0]
        )
        reshape_k_2_path_2 = self.model.match_parent_path(
            concat_k_2, ["Unsqueeze", "Add", "Gather", "Shape"], [2, 0, 0, 0]
        )
        if reshape_k_2_path_1 is None or reshape_k_2_path_2 is None:
            return False

        # Check Gather --> Mul --> Unsqueeze --> Concat --> Reshape connection for reshape_k_2_path_1
        # Check Gather --> Add --> Unsqueeze --> Concat --> Reshape connection for reshape_k_2_path_2
        if reshape_k_2_path_1[2].name != gather_1.name or reshape_k_2_path_2[2].name != gather_2.name:
            return False

        # Check #4: check paths for q nodes
        concat_q_2_path = self.model.match_parent_path(reshape_q_2, ["Concat"], [1])
        if concat_q_2_path is None:
            return False
        concat_q_2 = concat_q_2_path[0]

        reshape_q_2_path_1 = self.model.match_parent_path(
            concat_q_2, ["Unsqueeze", "Mul", "Gather", "Shape"], [0, 0, 0, 0]
        )
        reshape_q_2_path_2 = self.model.match_parent_path(concat_q_2, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        if reshape_q_2_path_1 is None or reshape_q_2_path_2 is None:
            return False

        # Check Gather --> Mul --> Unsqueeze --> Concat --> Reshape connection for reshape_q_2_path_1
        # Check Gather --> Unsqueeze --> Concat --> Reshape connection for reshape_q_2_path_2
        if reshape_q_2_path_1[2].name != gather_1.name or reshape_q_2_path_2[1].name != gather_2.name:
            return False

        # Check #5: check Mul nodes are the same for q, k, v
        mul_q = reshape_q_2_path_1[1]
        mul_k = reshape_k_2_path_1[1]
        mul_v = reshape_v_2_path_1[1]
        gather_1_out = gather_1.output[0]
        if mul_q.input[0] != gather_1_out or mul_k.input[0] != gather_1_out or mul_v.input[0] != gather_1_out:
            return False

        # Check #6: check paths for attention mask nodes
        attn_mask_path_1 = self.model.match_parent_path(add_qk, ["Concat", "Slice", "Slice"], [1, 0, 0])
        attn_mask_path_2 = self.model.match_parent_path(add_qk, ["Cast", "Concat", "Slice", "Slice"], [1, 0, 0, 0])
        concat_qk, slice_qk_2, slice_qk_1 = None, None, None
        if attn_mask_path_1 is not None:
            concat_qk, slice_qk_2, slice_qk_1 = attn_mask_path_1
        elif attn_mask_path_2 is not None:
            _, concat_qk, slice_qk_2, slice_qk_1 = attn_mask_path_2
        else:
            return False
        # Check first input to Slice #1 is 3D attention mask of shape (B,S,T)
        if slice_qk_1.input[0] not in {"attn_mask", "attention_mask"}:
            return False

        slice_qk_2_path = self.model.match_parent_path(
            slice_qk_2, ["Unsqueeze", "Add", "Gather", "Shape"], [2, 0, 1, 0]
        )
        slice_qk_1_path_1 = self.model.match_parent_path(
            slice_qk_1, ["Unsqueeze", "Add", "Gather", "Shape"], [2, 0, 1, 0]
        )
        slice_qk_1_path_2 = self.model.match_parent_path(slice_qk_1, ["Unsqueeze"], [1])
        if slice_qk_2_path is None or slice_qk_1_path_1 is None or slice_qk_1_path_2 is None:
            return False

        # Check Gather --> Add --> Unsqueeze #3 --> Slice #2 connection for slice_qk_2_path
        # Check Gather --> Add --> Unsqueeze #2 --> Slice #1 connection for slice_qk_1_path_1
        if slice_qk_2_path[1].name != slice_qk_1_path_1[1].name or slice_qk_2_path[2].name != slice_qk_1_path_1[2].name:
            return False

        # Check Unsqueeze #1 --> Slice #1 connection for slice_qk_1_path_2
        # Check if first input to Add and Unsqueeze #1 is position ids
        if slice_qk_1_path_1[1].input[0] != slice_qk_1_path_2[0].input[0]:
            return False

        return True

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        if normalize_node.op_type != "SkipSimplifiedLayerNormalization" and normalize_node.op_type != "Add":
            return

        # logger.info(f"Normalize node is {normalize_node.name}")
        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
            [1, 0, 0, 0, 0],
        )
        if qkv_nodes is not None:
            _, reshape_qkv_2, transpose_qkv, reshape_qkv_1, matmul_qkv = qkv_nodes
        else:
            logger.debug("fuse_rotary_attention: failed to match qkv nodes")
            return

        past_v, present_v, past_seq_len = "", "", ""
        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Reshape", "Concat", "Transpose", "Reshape", "MatMul"],
            [1, 0, 1, 0, 0],
        )
        if v_nodes is not None:
            reshape_v_2, concat_v, _, reshape_v_1, matmul_v = v_nodes
            concat_v_path = self.model.match_parent_path(
                concat_v,
                ["Slice", "Unsqueeze"],
                [0, 2],
            )

            if concat_v_path is None:
                # Handle the case where the cache is already unsqueezed
                concat_v_path = self.model.match_parent_path(
                    concat_v,
                    ["Slice"],
                    [0],
                )
                if concat_v_path is None:
                    logger.debug("fuse_rotary_attention: failed to match past/present concat in v path")
                    return

            past_v = concat_v_path[0].input[0]
            past_seq_len = concat_v_path[-1].input[0]
            present_v = concat_v.output[0]
        else:
            logger.debug("fuse_rotary_attention: failed to match v path")
            return

        qk_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Softmax", "Add", "Div", "MatMul"],
            [0, 0, 0, 0],
        )
        add_qk, matmul_qk = None, None
        if qk_nodes is not None:
            _, add_qk, _, matmul_qk = qk_nodes
        else:
            logger.debug("fuse_rotary_attention: failed to match qk nodes")
            return

        attn_mask_nodes_0 = self.model.match_parent_path(
            add_qk,
            ["Slice", "Slice"],
            [1, 0],
        )
        attn_mask_nodes_1 = self.model.match_parent_path(
            add_qk,
            ["Concat", "Slice", "Slice"],
            [1, 0, 0],
        )
        attn_mask_nodes_2 = self.model.match_parent_path(
            add_qk,
            ["Cast", "Concat", "Slice", "Slice"],
            [1, 0, 0, 0],
        )
        attn_mask = ""
        if attn_mask_nodes_0 is not None:
            slice_mask_1, slice_mask_2 = attn_mask_nodes_0
        elif attn_mask_nodes_1 is not None:
            _, slice_mask_1, slice_mask_2 = attn_mask_nodes_1
            attn_mask = slice_mask_1.output[0]
        elif attn_mask_nodes_2 is not None:
            _, _, slice_mask_1, slice_mask_2 = attn_mask_nodes_2
            attn_mask = slice_mask_1.output[0]
        else:
            logger.debug("fuse_rotary_attention: failed to match attention mask nodes")
            return

        past_k, present_k = "", ""
        k_nodes_1 = self.model.match_parent_path(
            matmul_qk,
            ["Reshape", "Transpose", "Concat", "Transpose", "RotaryEmbedding", "MatMul"],
            [1, 0, 0, 1, 0, 0],
        )

        # Try to match the ScatterND path
        k_nodes_2 = self.model.match_parent_path(
            matmul_qk,
            ["Reshape", "Transpose", "Concat", "Transpose", "ScatterND", "ScatterND", "Slice", "Reshape", "MatMul"],
            [1, 0, 0, 1, 0, 0, 0, 0, 0],
        )

        if k_nodes_1 is not None:
            reshape_k_2, _, concat_k, _, rotary_k, matmul_k = k_nodes_1
        elif k_nodes_2 is not None:
            reshape_k_2, _, concat_k, _, rotary_k, _, _, _, matmul_k = k_nodes_2
        else:
            logger.debug("fuse_rotary_attention: failed to match k nodes")
            return

        concat_k_path = self.model.match_parent_path(
            concat_k,
            ["Slice", "Unsqueeze"],
            [0, 2],
        )
        if concat_k_path is None:
            concat_k_path = self.model.match_parent_path(
                concat_k,
                ["Slice"],
                [0],
            )

            if concat_k_path is None:
                logger.debug("fuse_rotary_attention: failed to match past/present concat in k path")
                return

        past_k = concat_k_path[0].input[0]
        shared_past_seq_len = concat_k_path[-1].input[0]
        present_k = concat_k.output[0]

        # assert past_seq_len == shared_past_seq_len

        q_nodes_1 = self.model.match_parent_path(
            matmul_qk,
            ["Reshape", "Transpose", "RotaryEmbedding", "MatMul"],
            [0, 0, 0, 0],
        )

        # Try to match the ScatterND path
        q_nodes_2 = self.model.match_parent_path(
            matmul_qk,
            ["Reshape", "Transpose", "ScatterND", "ScatterND", "Slice", "Reshape", "MatMul"],
            [0, 0, 0, 0, 0, 0, 0],
        )
        if q_nodes_1 is not None:
            reshape_q_2, _, rotary_q, matmul_q = q_nodes_1
        elif q_nodes_2 is not None:
            reshape_q_2, _, rotary_q, _, _, _, matmul_q = q_nodes_2
        else:
            logger.debug("fuse_rotary_attention: failed to match q nodes")
            return

        if matmul_q.input[0] != matmul_k.input[0] and matmul_k.input[0] != matmul_v.input[0]:
            logger.debug("fuse_rotary_attention: failed to find the same root_input for q, k, v paths")
            return

        if q_nodes_1 is not None and not self.check_runtime_shape_paths(
            reshape_qkv_2,
            reshape_qkv_1,
            reshape_q_2,
            reshape_k_2,
            reshape_v_2,
            reshape_v_1,
            add_qk,
            matmul_q.input[0],
        ):
            logger.debug("fuse_rotary_attention: failed to verify runtime shape paths")
            return

        new_node = self.create_mha_node(
            matmul_q.input[0],
            reshape_qkv_2.output[0],
            rotary_q,
            rotary_k,
            matmul_v,
            attn_mask,
            past_k,
            past_v,
            present_k,
            present_v,
        )
        if new_node is None:
            logger.debug("fuse_rotary_attention: failed to create multi-head attention with rotary embeddings")
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend(qkv_nodes[1:])
        self.nodes_to_remove.extend(v_nodes[:-2])
        self.nodes_to_remove.extend(qk_nodes)

        if k_nodes_1 is not None:
            self.nodes_to_remove.extend(k_nodes_1[:-2])
        elif k_nodes_2 is not None:
            self.nodes_to_remove.extend(k_nodes_2[:-5])

        if q_nodes_1 is not None:
            self.nodes_to_remove.extend(q_nodes_1[:-2])
        elif q_nodes_2 is not None:
            self.nodes_to_remove.extend(q_nodes_2[:-5])

        self.prune_graph = True


class FusionRotaryEmbeddings(Fusion):
    def __init__(self, model: OnnxModel):
        self.base_name = "RotaryEmbedding"
        super().__init__(model, self.base_name, [self.base_name, self.base_name + ".1"])

    # The RotaryEmbedding function can have multiple extraneous constant outputs even though the function is supposed to produce only one output.
    # This is a byproduct of a potential CSE bug when using `export_modules_as_functions` in the TorchScript exporter.
    # To work around this issue, we set the extraneous constant values from the RotaryEmbedding function as initializers in the locations where they are actually used.
    def reassign_extra_outputs(self, rot_emb_node: NodeProto, function: FunctionProto):
        # Find extra outputs and Constant nodes attached to those outputs
        extra_constants, extra_outputs = [], []
        for fn_node in function.node:
            if fn_node.op_type == "Constant" and fn_node.input == [] and fn_node.output[0] in function.output:
                extra_constants.append(fn_node)
                output_index = list(function.output).index(fn_node.output[0])
                extra_outputs.append(rot_emb_node.output[output_index])

        # Set extra Constant node outputs as initializers
        extra_initializers = []
        for extra_constant in extra_constants:
            constant_tensorproto = extra_constant.attribute[0].t
            constant_tensorproto.name = self.model.create_node_name("Constant")
            self.model.add_initializer(constant_tensorproto)
            extra_initializers.append(constant_tensorproto.name)

        # Update references of Constant node outputs to initializer references
        for extra_output, extra_initializer in zip(extra_outputs, extra_initializers):
            nodes_to_update = list(filter(lambda entry: extra_output in entry.input, self.model.model.graph.node))
            for node_to_update in nodes_to_update:
                OnnxModel.replace_node_input(node_to_update, extra_output, extra_initializer)

        return extra_outputs

    def create_rotary_embeddings(self, node: NodeProto):
        rotary_emb_node_name = self.model.create_node_name(self.base_name)

        matmul_path = self.model.match_parent_path(
            node,
            ["Reshape", "MatMul"],
            [0, 0],
        )
        if matmul_path is not None:
            reshape_node, matmul_node = matmul_path
        else:
            logger.debug(f"fuse_rotary_embeddings: failed to match MatMul")
            return

        rotary_emb_inputs = [
            matmul_node.output[0],  # x is of shape (B,S,D) instead of (B,S,N,H)
            node.input[1],  # position_ids
            # node.input[2],           # cos_cache
            # node.input[3],           # sin_cache
        ]

        # Convert cos_cache and sin_cache from node attributes to model initializers
        cos_cache_node = list(filter(lambda constant: constant.output[0] == node.input[2], self.model.model.graph.node))
        sin_cache_node = list(filter(lambda constant: constant.output[0] == node.input[3], self.model.model.graph.node))
        cos_cache_name, sin_cache_name = "cos_cache", "sin_cache"

        if (
            len(cos_cache_node) == 1
            and len(sin_cache_node) == 1
            and self.model.get_initializer(cos_cache_name) is None
            and self.model.get_initializer(sin_cache_name) is None
        ):
            cos_cache = numpy_helper.to_array(cos_cache_node[0].attribute[0].t).squeeze()
            sin_cache = numpy_helper.to_array(sin_cache_node[0].attribute[0].t).squeeze()

            cos_cache_tensor = helper.make_tensor(
                name=cos_cache_name,
                data_type=TensorProto.FLOAT,
                dims=list(cos_cache.shape),
                vals=cos_cache.flatten().tolist(),
            )
            self.model.add_initializer(cos_cache_tensor, self.this_graph_name)
            sin_cache_tensor = helper.make_tensor(
                name=sin_cache_name,
                data_type=TensorProto.FLOAT,
                dims=list(sin_cache.shape),
                vals=sin_cache.flatten().tolist(),
            )
            self.model.add_initializer(sin_cache_tensor, self.this_graph_name)

            self.nodes_to_remove.extend([cos_cache_node[0], sin_cache_node[0]])

        rotary_emb_inputs.extend([cos_cache_name, sin_cache_name])

        rotary_emb_outputs = node.output
        if len(rotary_emb_outputs) > 1:
            # Re-assign extraneous constant outputs in RotaryEmbedding functions as initializers
            func = list(filter(lambda fn: fn.name == node.op_type, self.model.model.functions))
            assert len(func) == 1
            extra_outputs = self.reassign_extra_outputs(node, func[0])
            rotary_emb_outputs = list(filter(lambda output_name: output_name not in extra_outputs, rotary_emb_outputs))
            assert len(rotary_emb_outputs) == 1

        rotary_emb_node = helper.make_node(
            self.base_name,
            inputs=rotary_emb_inputs,
            outputs=rotary_emb_outputs,
            name=rotary_emb_node_name,
        )
        rotary_emb_node.domain = "com.microsoft"

        self.nodes_to_remove.append(reshape_node)

        return rotary_emb_node

    # Node is "RotaryEmbedding nn.Module" exported as a function
    # (e.g. export_modules_as_functions={RotaryEmbedding} in torch.onnx.export)
    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if self.base_name not in node.op_type:
            return

        # Verify that function has the correct inputs
        if len(node.input) not in {4, 5} or node.input[1] not in {
            "pos",
            "pos_id",
            "position_id",
            "pos_ids",
            "position_ids",
        }:
            logger.debug("fuse_rotary_embeddings: failed to verify inputs for RotaryEmbedding function")
            return

        rotary_emb_node = self.create_rotary_embeddings(node)
        if rotary_emb_node is None:
            logger.debug("fuse_rotary_embeddings: failed to create RotaryEmbedding node")
            return

        self.increase_counter(self.base_name)

        self.node_name_to_graph_name[rotary_emb_node.name] = self.this_graph_name
        self.nodes_to_add.append(rotary_emb_node)

        # Remove RotaryEmbedding function
        self.nodes_to_remove.append(node)
        self.prune_graph = True

        # Remove RotaryEmbedding function's shape inference stored in value_info
        # The new shape will be calculated during symbolic shape inference
        old_shape_infer = list(
            filter(lambda node: node.name == rotary_emb_node.output[0], self.model.model.graph.value_info)
        )
        assert len(old_shape_infer) == 1
        self.model.model.graph.value_info.remove(old_shape_infer[0])
