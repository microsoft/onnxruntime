# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import List, Optional
from fusion_base import Fusion
from fusion_utils import FusionUtils, NumpyHelper
from onnx import NodeProto, ModelProto, TensorProto, ValueInfoProto, helper, inliner, numpy_helper
from onnx_model import OnnxModel
from fusion_options import FusionOptions
from fusion_skiplayernorm import FusionBiasSkipLayerNormalization, FusionSkipLayerNormalization
import numpy as np
import os

logger = getLogger(__name__)


# TODO: handle the hard-coded values
class ProcessGemmWFunc:
    def __call__(self, x):
        return np.transpose(x, (1, 0))


class ProcessMatMulQFunc:
    def __call__(self, x):
        return np.transpose(np.split(x, 3, 0)[0], (1, 0))


class ProcessMatMulKFunc:
    def __call__(self, x):
        return np.transpose(np.split(x, 3, 0)[1], (1, 0))


class ProcessMatMulVFunc:
    def __call__(self, x):
        return np.transpose(np.split(x, 3, 0)[2], (1, 0))


class ProcessBiasQFunc:
    def __call__(self, x):
        x = np.split(x, 3, -1)[0]
        return x


class ProcessBiasKFunc:
    def __call__(self, x):
        x = np.split(x, 3, -1)[1]
        return x


class ProcessBiasVFunc:
    def __call__(self, x):
        x = np.split(x, 3, -1)[2]
        return x


class ProcessRotCacheFunc:
    def __call__(self, x):
        # half rotary embedding
        assert len(x.shape) == 2
        if x.shape[1] == 32:
            return x[:, 0:16]
        return x


# TODO: move to a seperate file
class Fission(Fusion):
    def __init__(
        self,
        model: OnnxModel,
        nodes_to_find: List[str],
    ):
        super().__init__(model, "DONOTUSE", nodes_to_find)

    def get_uname(self, layer_id, name):
        return name + "_" + str(layer_id)

    def get_io_by_name(self, node, name):
        for input in node.input:
            if input == name or input.endswith(name) or input.startswith(name):
                return input
        for output in node.output:
            if output == name or output.endswith(name) or output.startswith(name):
                return output
        raise Exception(f"input {name} not found in node {node.name}")

    def process_initializer(self, initializer_name, functor, custom_name=None):
        i = self.model.get_initializer(initializer_name)
        i_np_array = NumpyHelper.to_array(i)
        processed_i_np_array = functor(i_np_array)
        new_tensor = helper.make_tensor(
            initializer_name + "_processed" if custom_name is None else custom_name,
            data_type=TensorProto.FLOAT,
            dims=processed_i_np_array.shape,
            vals=processed_i_np_array.flatten().tobytes(),
            raw=True,
        )
        self.model.add_initializer(new_tensor, self.this_graph_name)
        return new_tensor.name

    def add_fp32_value_info(self, name):
        new_value_info = self.model.graph().value_info.add()
        new_value_info.name = name
        new_value_info.type.tensor_type.elem_type = TensorProto.FLOAT

    def add_int64_value_info(self, name):
        new_value_info = self.model.graph().value_info.add()
        new_value_info.name = name
        new_value_info.type.tensor_type.elem_type = TensorProto.INT64

    def replace_fp32_value_info(self, name, shape):
        for value_info in self.model.graph().value_info:
            if value_info.name == name:
                self.model.graph().value_info.remove(value_info)
                break
        new_value_info = helper.make_tensor_value_info(
            name,
            elem_type=TensorProto.FLOAT,
            shape=shape,
        )
        self.model.graph().value_info.extend([new_value_info])

    def set_unique_name_and_add_nodes(
        self, subgraph_nodes: List[NodeProto], layer_id: int, layer_known_edges_names: List[str]
    ):
        for new_node in subgraph_nodes:
            for i, name in enumerate(new_node.input):
                if name == "":
                    continue
                elif name not in layer_known_edges_names:
                    new_node.input[i] = self.get_uname(layer_id, name)
                    self.add_fp32_value_info(new_node.input[i])
            for i, name in enumerate(new_node.output):
                if name == "":
                    continue
                elif name not in layer_known_edges_names:
                    new_node.output[i] = self.get_uname(layer_id, name)
                    self.add_fp32_value_info(new_node.output[i])
            new_node.name = self.get_uname(layer_id, new_node.name)
            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

    def layernorm(self, inputs: List[str], outputs: List[str], prefix: str = ""):
        assert len(inputs) == 3
        assert len(outputs) == 1
        node = helper.make_node(
            "LayerNormalization",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "_LayerNormalization",
            epsilon=9.999999747378752e-06,
        )
        return [node]

    def gemm(self, inputs: List[str], outputs: List[str], prefix: str = ""):
        assert len(inputs) == 3
        assert len(outputs) == 1
        matmul = helper.make_node(
            "MatMul",
            inputs=[inputs[0], inputs[1]],
            outputs=[prefix + "matmul_out"],
            name=prefix + "MatMul",
        )
        add = helper.make_node(
            "Add",
            inputs=[prefix + "matmul_out", inputs[2]],
            outputs=outputs,
            name=prefix + "Bias",
        )
        return [matmul, add]

    def rotary(self, inputs: List[str], outputs: List[str], prefix: str = "", rot_dim=32, num_heads=32):
        assert len(inputs) == 4
        assert len(outputs) == 1
        node = helper.make_node(
            "RotaryEmbedding",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "RotaryEmbedding",
            domain="com.microsoft",
            rotary_embedding_dim=rot_dim,
            num_heads=num_heads,
        )
        return [node]

    def fastgelu(self, inputs: List[str], outputs: List[str], prefix: str = ""):
        assert len(inputs) == 1
        assert len(outputs) == 1
        node = helper.make_node(
            "FastGelu",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "FastGelu",
            domain="com.microsoft",
        )
        return [node]

    def add(self, inputs: List[str], outputs: List[str], prefix: str = ""):
        assert len(inputs) == 2
        assert len(outputs) == 1
        node = helper.make_node(
            "Add",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "Add",
        )
        return [node]

    def mha(self, inputs: List[str], outputs: List[str], prefix: str = "", num_heads=32):
        assert len(inputs) == 8
        assert len(outputs) == 3
        node = helper.make_node(
            "MultiHeadAttention",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "MultiHeadAttention",
            domain="com.microsoft",
            num_heads=num_heads,
            unidirectional=1,
        )
        return [node]

    def gqa(self, inputs: List[str], outputs: List[str], prefix: str = "", num_heads=32):
        assert len(inputs) == 7
        assert len(outputs) == 3
        node = helper.make_node(
            "GroupQueryAttention",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "GroupQueryAttention",
            domain="com.microsoft",
            num_heads=num_heads,
            kv_num_heads=num_heads,
        )
        return [node]

    def attention(self, inputs: List[str], outputs: List[str], prefix: str = "", num_heads=32):
        assert len(inputs) == 5
        assert len(outputs) == 2
        node = helper.make_node(
            "Attention",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "Attention",
            domain="com.microsoft",
            num_heads=num_heads,
            unidirectional=1,
            do_rotary=1,
            rotary_embedding_dim=32,
        )
        return [node]


class FissionTransformerEmbeddingPhi(Fission):
    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, ["torch_nn_modules_sparse_Embedding_model_embed_tokens_1"])

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        logger.info("Optimizing %s...", node.name)

        assert len(node.input) == 2
        assert len(node.output) == 1

        input = node.input[0]
        output = node.output[0]

        embedding = self.get_io_by_name(node, "embed_tokens.weight")

        layer_known_edges_names = [input, output, embedding]

        subgraph_nodes = [
            helper.make_node(
                "Gather",
                inputs=[embedding, input],
                outputs=[output],
                name="Embedding_Gather",
            ),
        ]

        self.set_unique_name_and_add_nodes(subgraph_nodes, 0, layer_known_edges_names)
        self.nodes_to_remove.append(node)
        self.prune_graph = True


class FissionTransformerLayerNormPhi(Fission):
    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, ["torch_nn_modules_normalization_LayerNorm_model_final_layernorm_1"])

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        logger.info("Optimizing %s...", node.name)

        assert len(node.input) == 3
        assert len(node.output) == 1

        input = node.input[0]
        output = node.output[0]

        ln_weight = self.get_io_by_name(node, "final_layernorm.weight")
        ln_bias = self.get_io_by_name(node, "final_layernorm.bias")

        layer_known_edges_names = [input, output, ln_weight, ln_bias]

        subgraph_nodes = []
        subgraph_nodes.extend(self.layernorm([input, ln_weight, ln_bias], [output], "Final"))

        self.set_unique_name_and_add_nodes(subgraph_nodes, 99, layer_known_edges_names)

        self.replace_fp32_value_info(input, ["batch_size", "seq_len", "hidden_size"])
        self.replace_fp32_value_info(output, ["batch_size", "seq_len", "hidden_size"])

        self.nodes_to_remove.append(node)
        self.prune_graph = True


class FissionTransformerCausalLMHeadPhi(Fission):
    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, ["torch_nn_modules_linear_Linear_lm_head_1"])

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        logger.info("Optimizing %s...", node.name)

        assert len(node.input) == 5
        assert len(node.output) == 1

        input = node.input[2]
        output = node.output[0]

        fc_weight = self.process_initializer(self.get_io_by_name(node, "lm_head.weight"), ProcessGemmWFunc())
        fc_bias = self.get_io_by_name(node, "lm_head.bias")

        layer_known_edges_names = [input, output, fc_weight, fc_bias]

        subgraph_nodes = []
        subgraph_nodes.extend(self.gemm([input, fc_weight, fc_bias], [output], "LMHead_"))

        self.set_unique_name_and_add_nodes(subgraph_nodes, 99, layer_known_edges_names)

        self.replace_fp32_value_info(input, ["batch_size", "seq_len", "hidden_size"])
        self.replace_fp32_value_info(output, ["batch_size", "seq_len", 51200])

        self.nodes_to_remove.append(node)
        self.prune_graph = True


class FissionTransformerBlockPhi(Fission):
    def __init__(
        self,
        model: OnnxModel,
        num_heads: int,
    ):
        self.num_heads = num_heads
        max_num_layers = 32
        self.func_to_layer_id = {}
        nodes_to_find = []
        for layer in range(max_num_layers):
            func_name = f"modeling_phi_PhiDecoderLayer_model_layers_{layer}_1"
            nodes_to_find.append(func_name)
            self.func_to_layer_id[func_name] = layer

        super().__init__(model, nodes_to_find)

    def get_layer_id(self, node):
        return self.func_to_layer_id[node.op_type]

    def get_gqa_aux_nodes(self):
        gqa_aux_nodes = [
            helper.make_node(
                "Cast",
                inputs=["attention_mask"],
                outputs=["mask_int64"],
                name="Cast_gqa_aux_0",
                to=TensorProto.INT64,
            ),
            helper.make_node(
                "ReduceSum",
                inputs=["mask_int64", "one"],
                outputs=["mask_row_sums"],
                name="ReduceSum_gqa_aux",
            ),
            helper.make_node(
                "Sub",
                inputs=["mask_row_sums", "one"],
                outputs=["seqlens_k_int64"],
                name="Sub_gqa_aux",
            ),
            helper.make_node(
                "Cast",
                inputs=["seqlens_k_int64"],
                outputs=["seqlens_k"],
                name="Cast_gqa_aux_1",
                to=TensorProto.INT32,
            ),
            helper.make_node("Shape", inputs=["mask_int64"], outputs=["mask_shape"], name="Shape_gqa_aux_0"),
            helper.make_node(
                "Gather",
                inputs=["mask_shape", "one"],
                outputs=["total_seq_len_int64"],
                name="Gather_gqa_aux_0",
                axis=0,
            ),
            helper.make_node(
                "Cast",
                inputs=["total_seq_len_int64"],
                outputs=["total_sequence_length"],
                name="Cast_gqa_aux_2",
                to=TensorProto.INT32,
            ),
        ]
        return gqa_aux_nodes

    def pack_qkv_gemm(self, q_w, k_w, v_w, q_b, k_b, v_b, weight_name, bias_name):
        q_weight = self.model.get_initializer(q_w)
        k_weight = self.model.get_initializer(k_w)
        v_weight = self.model.get_initializer(v_w)
        qw = np.transpose(NumpyHelper.to_array(q_weight), (1, 0))
        kw = np.transpose(NumpyHelper.to_array(k_weight), (1, 0))
        vw = np.transpose(NumpyHelper.to_array(v_weight), (1, 0))
        qkv_weight = np.stack((qw, kw, vw), axis=1)

        q_bias = self.model.get_initializer(q_b)
        k_bias = self.model.get_initializer(k_b)
        v_bias = self.model.get_initializer(v_b)
        qb = NumpyHelper.to_array(q_bias)
        kb = NumpyHelper.to_array(k_bias)
        vb = NumpyHelper.to_array(v_bias)
        qkv_bias = np.stack((qb, kb, vb), axis=0)

        hidden_size = qkv_weight.shape[0]

        # bugbug: shape is wrong
        weight = helper.make_tensor(
            weight_name,
            data_type=TensorProto.FLOAT,
            dims=[hidden_size, hidden_size * 3],
            vals=qkv_weight.flatten().tobytes(),
            raw=True,
        )
        self.model.add_initializer(weight, self.this_graph_name)

        bias = helper.make_tensor(
            bias_name,
            data_type=TensorProto.FLOAT,
            dims=[hidden_size * 3],
            vals=qkv_bias.flatten().tobytes(),
            raw=True,
        )
        self.model.add_initializer(bias, self.this_graph_name)

        self.add_fp32_value_info(weight.name)
        self.add_fp32_value_info(bias.name)

        return weight_name, bias_name

    def fuse(
        self,
        node,
        input_name_to_nodes,
        output_name_to_node,
    ):
        logger.info("Optimizing %s...", node.name)

        attn_type = os.environ.get("AttentionOpType")
        logger.info(f"AttentionOpType: {attn_type}")

        layer_id = self.get_layer_id(node)

        i_hidden_states = node.input[0]
        i_key_cache = self.get_io_by_name(node, "past_key")
        i_value_cache = self.get_io_by_name(node, "past_value")

        o_hidden_states = node.output[3]
        o_key_cache = self.get_io_by_name(node, "present_key")
        o_value_cache = self.get_io_by_name(node, "present_value")

        ln_weight = self.get_io_by_name(node, "input_layernorm.weight")
        ln_bias = self.get_io_by_name(node, "input_layernorm.bias")

        if attn_type != "Attention":
            attn_q_weight = self.process_initializer(
                self.get_io_by_name(node, "self_attn.q_proj.weight"), ProcessGemmWFunc()
            )
            attn_k_weight = self.process_initializer(
                self.get_io_by_name(node, "self_attn.k_proj.weight"), ProcessGemmWFunc()
            )
            attn_v_weight = self.process_initializer(
                self.get_io_by_name(node, "self_attn.v_proj.weight"), ProcessGemmWFunc()
            )
            attn_q_bias = self.get_io_by_name(node, "self_attn.q_proj.bias")
            attn_k_bias = self.get_io_by_name(node, "self_attn.k_proj.bias")
            attn_v_bias = self.get_io_by_name(node, "self_attn.v_proj.bias")

            cos_cache = self.process_initializer(
                self.get_io_by_name(node, "rotary_emb.cos_cached"), ProcessRotCacheFunc()
            )
            sin_cache = self.process_initializer(
                self.get_io_by_name(node, "rotary_emb.sin_cached"), ProcessRotCacheFunc()
            )
        else:
            attn_qkv_weight, attn_qkv_bias = self.pack_qkv_gemm(
                self.get_io_by_name(node, "self_attn.q_proj.weight"),
                self.get_io_by_name(node, "self_attn.k_proj.weight"),
                self.get_io_by_name(node, "self_attn.v_proj.weight"),
                self.get_io_by_name(node, "self_attn.q_proj.bias"),
                self.get_io_by_name(node, "self_attn.k_proj.bias"),
                self.get_io_by_name(node, "self_attn.v_proj.bias"),
                self.get_uname(layer_id, "attn_qkv_weight"),
                self.get_uname(layer_id, "attn_qkv_bias"),
            )

        attn_out_weight = self.process_initializer(
            self.get_io_by_name(node, "self_attn.dense.weight"), ProcessGemmWFunc()
        )
        attn_out_bias = self.get_io_by_name(node, "self_attn.dense.bias")

        mlp_fc1_weight = self.process_initializer(self.get_io_by_name(node, "mlp.fc1.weight"), ProcessGemmWFunc())
        mlp_fc2_weight = self.process_initializer(self.get_io_by_name(node, "mlp.fc2.weight"), ProcessGemmWFunc())
        mlp_fc1_bias = self.get_io_by_name(node, "mlp.fc1.bias")
        mlp_fc2_bias = self.get_io_by_name(node, "mlp.fc2.bias")

        layer_known_edges_names = []
        layer_known_edges_names.extend([i_hidden_states, i_key_cache, i_value_cache])
        layer_known_edges_names.extend([o_hidden_states, o_key_cache, o_value_cache])
        layer_known_edges_names.extend([ln_weight, ln_bias])
        if attn_type != "Attention":
            layer_known_edges_names.extend(
                [
                    attn_q_weight,
                    attn_q_bias,
                    attn_k_weight,
                    attn_k_bias,
                    attn_v_weight,
                    attn_v_bias,
                    cos_cache,
                    sin_cache,
                ]
            )
        else:
            layer_known_edges_names.extend([attn_qkv_weight, attn_qkv_bias])
        layer_known_edges_names.extend(
            [attn_out_weight, attn_out_bias, mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias]
        )
        layer_known_edges_names.extend(["attention_mask", "step", "seqlens_k", "total_sequence_length"])

        subgraph_nodes = []
        subgraph_nodes.extend(self.layernorm([i_hidden_states, ln_weight, ln_bias], ["ln_out"]))
        subgraph_nodes.extend(self.gemm(["attn_out", attn_out_weight, attn_out_bias], ["attn_add_out"], "OutProj_"))
        subgraph_nodes.extend(self.gemm(["ln_out", mlp_fc1_weight, mlp_fc1_bias], ["fc1_out"], "FC1_"))
        subgraph_nodes.extend(self.fastgelu(["fc1_out"], ["gelu_out"]))
        subgraph_nodes.extend(self.gemm(["gelu_out", mlp_fc2_weight, mlp_fc2_bias], ["fc2_out"], "FC2_"))
        subgraph_nodes.extend(self.add(["attn_add_out", "fc2_out"], ["residual_1_out"], "Residual_1"))
        subgraph_nodes.extend(self.add([i_hidden_states, "residual_1_out"], [o_hidden_states], "Residual_2"))
        if attn_type != "Attention":
            subgraph_nodes.extend(self.gemm(["ln_out", attn_q_weight, attn_q_bias], ["query"], "Q_"))
            subgraph_nodes.extend(self.gemm(["ln_out", attn_k_weight, attn_k_bias], ["key"], "K_"))
            subgraph_nodes.extend(self.gemm(["ln_out", attn_v_weight, attn_v_bias], ["value"], "V_"))
            subgraph_nodes.extend(self.rotary(["query", "step", cos_cache, sin_cache], ["query_rot"], "Q_"))
            subgraph_nodes.extend(self.rotary(["key", "step", cos_cache, sin_cache], ["key_rot"], "K_"))
            if attn_type == "MultiHeadAttention":
                subgraph_nodes.extend(
                    self.mha(
                        ["query_rot", "key_rot", "value", "", "attention_mask", "", i_key_cache, i_value_cache],
                        ["attn_out", o_key_cache, o_value_cache],
                    )
                )
            elif attn_type == "GroupQueryAttention":
                subgraph_nodes.extend(
                    self.gqa(
                        [
                            "query_rot",
                            "key_rot",
                            "value",
                            i_key_cache,
                            i_value_cache,
                            "seqlens_k",
                            "total_sequence_length",
                        ],
                        ["attn_out", o_key_cache, o_value_cache],
                    )
                )
                if layer_id == 0:
                    gqa_aux_nodes = self.get_gqa_aux_nodes()
                    for new_node in gqa_aux_nodes:
                        self.nodes_to_add.append(new_node)
                        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
                    self.model.add_initializer(
                        numpy_helper.from_array(np.array([1], dtype="int64"), name="one"), self.this_graph_name
                    )
        else:
            past_name = f"past_{layer_id}"
            present_name = f"present_{layer_id}"
            layer_known_edges_names.extend([past_name, present_name])
            subgraph_nodes.extend(
                self.attention(
                    ["ln_out", attn_qkv_weight, attn_qkv_bias, "attention_mask", past_name], ["attn_out", present_name]
                )
            )

        self.set_unique_name_and_add_nodes(subgraph_nodes, layer_id, layer_known_edges_names)

        self.replace_fp32_value_info(i_hidden_states, ["batch_size", "seq_len", "hidden_size"])
        self.replace_fp32_value_info(o_hidden_states, ["batch_size", "seq_len", "hidden_size"])

        self.nodes_to_remove.append(node)
        self.prune_graph = True


class PhiOnnxModel(OnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, head_size: int = 0):
        super().__init__(model)
        self.fission_transformer_block = FissionTransformerBlockPhi(self, num_heads)
        self.fission_causal_lm_head = FissionTransformerCausalLMHeadPhi(self)
        self.fission_transformer_layernorm = FissionTransformerLayerNormPhi(self)
        self.fission_transformer_embedding = FissionTransformerEmbeddingPhi(self)
        self.fuse_sln = FusionSkipLayerNormalization(self)
        self.fuse_bias_sln = FusionBiasSkipLayerNormalization(self)

    def optimize(self, options: Optional[FusionOptions] = None, add_dynamic_axes: bool = False):
        self.fission_transformer_block.apply()
        self.fission_transformer_layernorm.apply()
        self.fission_causal_lm_head.apply()
        self.fission_transformer_embedding.apply()

        super().prune_graph()

        self.fuse_sln.apply()
        self.fuse_bias_sln.apply()
