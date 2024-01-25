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


def uname(layer_id, name):
    return name + "_" + str(layer_id)


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

    def set_unique_name(self, subgraph_nodes: List[NodeProto], layer_id: int, layer_known_edges_names: List[str]):
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


class FissionTransformerCausalLMHeadPhi(Fission):
    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, ["torch_nn_modules_linear_Linear_lm_head_1"])

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        input = node.input[2]
        output = node.output[0]

        fc_weight = self.process_initializer(self.get_io_by_name(node, "lm_head.weight"), ProcessGemmWFunc())
        fc_bias = self.get_io_by_name(node, "lm_head.bias")

        layer_known_edges_names = [input, output, fc_weight, fc_bias]

        # opt graph construction.
        subgraph_nodes = [
            helper.make_node(
                "MatMul",
                inputs=[input, fc_weight],
                outputs=["matmul_out"],
                name="OutProj_MatMul",
            ),
            helper.make_node(
                "Add",
                inputs=["matmul_out", fc_bias],
                outputs=[output],
                name="OutProj_Add",
            ),
        ]

        self.set_unique_name(subgraph_nodes, 99, layer_known_edges_names)

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

    # def fuse_with_attn(
    #     self,
    #     node,
    #     input_name_to_nodes,
    #     output_name_to_node,
    # ):
    #     layer_id = self.get_layer_id(node)
    #     print(f"fuse layer {layer_id}")

    #     # transformer block input and output
    #     i_hidden_states = node.input[0]
    #     i_attn_mask = node.input[1]
    #     i_kv_cache = node.input[3]
    #     o_hidden_states = node.output[3]
    #     o_kv_cache = node.output[0]

    #     # internal nodes weights
    #     ln_weight = node.input[5]  # float32[2560]
    #     ln_bias = node.input[6]  # float32[2560]
    #     attn_qkv_weight = self.process_initializer(node.input[7], ProcessGemmWFunc())  # float32[7680,2560]
    #     attn_qkv_bias = node.input[8]  # float32[7680]
    #     attn_out_weight = self.process_initializer(node.input[11], ProcessGemmWFunc())  # float32[2560,2560]
    #     attn_out_bias = node.input[12]  # float32[2560]
    #     mlp_fc1_weight = self.process_initializer(node.input[13], ProcessGemmWFunc())  # float32[10240,2560]
    #     mlp_fc1_bias = node.input[14]  # float32[10240]
    #     mlp_fc2_weight = self.process_initializer(node.input[15], ProcessGemmWFunc())  # float32[2560,10240]
    #     mlp_fc2_bias = node.input[16]  # float32[2560]

    #     # opt graph construction.
    #     subgraph_nodes = [
    #         helper.make_node(
    #             "LayerNormalization",
    #             inputs=[i_hidden_states, ln_weight, ln_bias],
    #             outputs=[uname(layer_id, "ln_out")],
    #             name=uname(layer_id, "LayerNormalization"),
    #             epsilon=9.999999747378752e-06,
    #         ),
    #         helper.make_node(
    #             "Attention",
    #             inputs=[
    #                 uname(layer_id, "ln_out"),
    #                 attn_qkv_weight,
    #                 attn_qkv_bias,
    #                 i_attn_mask,
    #                 i_kv_cache,
    #                 # "",
    #                 # "past_sequence_length",
    #             ],
    #             outputs=[uname(layer_id, "attn_out"), o_kv_cache],
    #             name=uname(layer_id, "Attention"),
    #             domain="com.microsoft",
    #             num_heads=32,
    #             unidirectional=1,
    #             do_rotary=1,
    #             rotary_embedding_dim=32,
    #             # past_present_share_buffer=1,
    #         ),
    #         helper.make_node(
    #             "MatMul",
    #             inputs=[uname(layer_id, "attn_out"), attn_out_weight],
    #             outputs=[uname(layer_id, "matmul_out")],
    #             name=uname(layer_id, "OutProj_MatMul"),
    #         ),
    #         helper.make_node(
    #             "Add",
    #             inputs=[uname(layer_id, "matmul_out"), attn_out_bias],
    #             outputs=[uname(layer_id, "add_out")],
    #             name=uname(layer_id, "OutProj_Add"),
    #         ),
    #         helper.make_node(
    #             "MatMul",
    #             inputs=[uname(layer_id, "ln_out"), mlp_fc1_weight],
    #             outputs=[uname(layer_id, "fc1_w_out")],
    #             name=uname(layer_id, "FC1_MatMul"),
    #         ),
    #         helper.make_node(
    #             "Add",
    #             inputs=[uname(layer_id, "fc1_w_out"), mlp_fc1_bias],
    #             outputs=[uname(layer_id, "fc1_b_out")],
    #             name=uname(layer_id, "FC1_Bias"),
    #         ),
    #         helper.make_node(
    #             "FastGelu",
    #             inputs=[uname(layer_id, "fc1_b_out")],
    #             outputs=[uname(layer_id, "new_gelu_out")],
    #             name=uname(layer_id, "FastGelu"),
    #             domain="com.microsoft",
    #         ),
    #         helper.make_node(
    #             "MatMul",
    #             inputs=[uname(layer_id, "new_gelu_out"), mlp_fc2_weight],
    #             outputs=[uname(layer_id, "fc2_w_out")],
    #             name=uname(layer_id, "FC2_MatMul"),
    #         ),
    #         helper.make_node(
    #             "Add",
    #             inputs=[uname(layer_id, "fc2_w_out"), mlp_fc2_bias],
    #             outputs=[uname(layer_id, "fc2_b_out")],
    #             name=uname(layer_id, "FC2_Bias"),
    #         ),
    #         helper.make_node(
    #             "Add",
    #             inputs=[uname(layer_id, "add_out"), uname(layer_id, "fc2_b_out")],
    #             outputs=[uname(layer_id, "residual_1_out")],
    #             name=uname(layer_id, "Residual_Add_1"),
    #         ),
    #         helper.make_node(
    #             "Add",
    #             inputs=[i_hidden_states, uname(layer_id, "residual_1_out")],
    #             outputs=[o_hidden_states],
    #             name=uname(layer_id, "Residual_Add_2"),
    #         ),
    #     ]

    #     for new_node in subgraph_nodes:
    #         self.nodes_to_add.append(new_node)
    #         self.node_name_to_graph_name[new_node.name] = self.this_graph_name

    #     self.add_fp32_value_info(uname(layer_id, "ln_out"))
    #     self.add_fp32_value_info(uname(layer_id, "attn_out"))
    #     self.add_fp32_value_info(uname(layer_id, "matmul_out"))
    #     self.add_fp32_value_info(uname(layer_id, "add_out"))
    #     self.add_fp32_value_info(uname(layer_id, "fc1_w_out"))
    #     self.add_fp32_value_info(uname(layer_id, "fc1_b_out"))
    #     self.add_fp32_value_info(uname(layer_id, "new_gelu_out"))
    #     self.add_fp32_value_info(uname(layer_id, "fc2_w_out"))
    #     self.add_fp32_value_info(uname(layer_id, "fc2_b_out"))
    #     self.add_fp32_value_info(uname(layer_id, "residual_1_out"))

    #     self.replace_fp32_value_info(i_hidden_states, ["batch_size", "seq_len", "hidden_size"])
    #     self.replace_fp32_value_info(o_hidden_states, ["batch_size", "seq_len", "hidden_size"])

    #     self.nodes_to_remove.append(node)
    #     self.prune_graph = True

    def fuse(
        self,
        node,
        input_name_to_nodes,
        output_name_to_node,
    ):
        layer_id = self.get_layer_id(node)
        print(f"fuse layer {layer_id}")

        i_hidden_states = node.input[0]
        i_key_cache = self.get_io_by_name(node, "past_key")
        i_value_cache = self.get_io_by_name(node, "past_value")

        o_hidden_states = node.output[3]
        o_key_cache = self.get_io_by_name(node, "present_key")
        o_value_cache = self.get_io_by_name(node, "present_value")

        ln_weight = self.get_io_by_name(node, "input_layernorm.weight")
        ln_bias = self.get_io_by_name(node, "input_layernorm.bias")
        attn_q_weight = self.process_initializer(
            self.get_io_by_name(node, "self_attn.q_proj.weight"), ProcessGemmWFunc()
        )
        attn_k_weight = self.process_initializer(
            self.get_io_by_name(node, "self_attn.k_proj.weight"), ProcessGemmWFunc()
        )
        attn_v_weight = self.process_initializer(
            self.get_io_by_name(node, "self_attn.v_proj.weight"), ProcessGemmWFunc()
        )
        attn_out_weight = self.process_initializer(
            self.get_io_by_name(node, "self_attn.dense.weight"), ProcessGemmWFunc()
        )

        attn_q_bias = self.get_io_by_name(node, "self_attn.q_proj.bias")
        attn_k_bias = self.get_io_by_name(node, "self_attn.k_proj.bias")
        attn_v_bias = self.get_io_by_name(node, "self_attn.v_proj.bias")
        attn_out_bias = self.get_io_by_name(node, "self_attn.dense.bias")

        cos_cache = self.process_initializer(self.get_io_by_name(node, "rotary_emb.cos_cached"), ProcessRotCacheFunc())
        sin_cache = self.process_initializer(self.get_io_by_name(node, "rotary_emb.sin_cached"), ProcessRotCacheFunc())

        mlp_fc1_weight = self.process_initializer(self.get_io_by_name(node, "mlp.fc1.weight"), ProcessGemmWFunc())
        mlp_fc2_weight = self.process_initializer(self.get_io_by_name(node, "mlp.fc2.weight"), ProcessGemmWFunc())
        mlp_fc1_bias = self.get_io_by_name(node, "mlp.fc1.bias")
        mlp_fc2_bias = self.get_io_by_name(node, "mlp.fc2.bias")

        layer_known_edges_names = []
        layer_known_edges_names.extend([i_hidden_states, i_key_cache, i_value_cache])
        layer_known_edges_names.extend([o_hidden_states, o_key_cache, o_value_cache])
        layer_known_edges_names.extend([ln_weight, ln_bias])
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
                attn_out_weight,
                attn_out_bias,
            ]
        )
        layer_known_edges_names.extend([mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias])
        layer_known_edges_names.extend(["attention_mask", "step", "seqlens_k", "total_sequence_length"])

        # opt graph construction.
        subgraph_nodes = [
            helper.make_node(
                "LayerNormalization",
                inputs=[i_hidden_states, ln_weight, ln_bias],
                outputs=["ln_out"],
                name="LayerNormalization",
                epsilon=9.999999747378752e-06,
            ),
            helper.make_node(
                "MatMul",
                inputs=["ln_out", attn_q_weight],
                outputs=["q_matmul_out"],
                name="Q_MatMul",
            ),
            helper.make_node(
                "Add",
                inputs=["q_matmul_out", attn_q_bias],
                outputs=["query"],
                name="Q_Bias",
            ),
            helper.make_node(
                "MatMul",
                inputs=["ln_out", attn_k_weight],
                outputs=["k_matmul_out"],
                name="K_MatMul",
            ),
            helper.make_node(
                "Add",
                inputs=["k_matmul_out", attn_k_bias],
                outputs=["key"],
                name="K_Bias",
            ),
            helper.make_node(
                "RotaryEmbedding",
                inputs=["query", "step", cos_cache, sin_cache],
                outputs=["query_rot"],
                name="RotaryEmbedding_Q",
                domain="com.microsoft",
                rotary_embedding_dim=32,
                num_heads=self.num_heads,
            ),
            helper.make_node(
                "RotaryEmbedding",
                inputs=["key", "step", cos_cache, sin_cache],
                outputs=["key_rot"],
                name="RotaryEmbedding_K",
                domain="com.microsoft",
                rotary_embedding_dim=32,
                num_heads=self.num_heads,
            ),
            helper.make_node(
                "MatMul",
                inputs=["ln_out", attn_v_weight],
                outputs=["v_matmul_out"],
                name="V_MatMul",
            ),
            helper.make_node(
                "Add",
                inputs=["v_matmul_out", attn_v_bias],
                outputs=["value"],
                name="V_Bias",
            ),
            helper.make_node(
                "MatMul",
                inputs=["attn_out", attn_out_weight],
                outputs=["matmul_out"],
                name="OutProj_MatMul",
            ),
            helper.make_node(
                "Add",
                inputs=["matmul_out", attn_out_bias],
                outputs=["add_out"],
                name="OutProj_Add",
            ),
            helper.make_node(
                "MatMul",
                inputs=["ln_out", mlp_fc1_weight],
                outputs=["fc1_w_out"],
                name="FC1_MatMul",
            ),
            helper.make_node(
                "Add",
                inputs=["fc1_w_out", mlp_fc1_bias],
                outputs=["fc1_b_out"],
                name="FC1_Bias",
            ),
            helper.make_node(
                "FastGelu",
                inputs=["fc1_b_out"],
                outputs=["new_gelu_out"],
                name="FastGelu",
                domain="com.microsoft",
            ),
            helper.make_node(
                "MatMul",
                inputs=["new_gelu_out", mlp_fc2_weight],
                outputs=["fc2_w_out"],
                name="FC2_MatMul",
            ),
            helper.make_node(
                "Add",
                inputs=["fc2_w_out", mlp_fc2_bias],
                outputs=["fc2_b_out"],
                name="FC2_Bias",
            ),
            helper.make_node(
                "Add",
                inputs=["add_out", "fc2_b_out"],
                outputs=["residual_1_out"],
                name="Residual_Add_1",
            ),
            helper.make_node(
                "Add",
                inputs=[i_hidden_states, "residual_1_out"],
                outputs=[o_hidden_states],
                name="Residual_Add_2",
            ),
        ]

        use_mha = False
        if use_mha:
            subgraph_nodes.append(
                helper.make_node(
                    "MultiHeadAttention",
                    inputs=[
                        "query_rot",
                        "key_rot",
                        "value",
                        "",
                        "attention_mask",
                        "",
                        i_key_cache,
                        i_value_cache,
                    ],
                    outputs=["attn_out", o_key_cache, o_value_cache],
                    name="MultiHeadAttention_0",
                    domain="com.microsoft",
                    num_heads=self.num_heads,
                    unidirectional=1,
                )
            )
        else:
            subgraph_nodes.append(
                helper.make_node(
                    "GroupQueryAttention",
                    inputs=[
                        "query_rot",
                        "key_rot",
                        "value",
                        i_key_cache,
                        i_value_cache,
                        "seqlens_k",
                        "total_sequence_length",
                    ],
                    outputs=["attn_out", o_key_cache, o_value_cache],
                    name="GroupQueryAttention_0",
                    domain="com.microsoft",
                    num_heads=self.num_heads,
                    kv_num_heads=self.num_heads,
                ),
            )
            if layer_id == 0:
                gqa_aux_nodes = [
                    helper.make_node(
                        "ReduceSum",
                        inputs=["attention_mask", "one"],
                        outputs=["attention_mask_row_sums"],
                        name="ReduceSum_gqa_aux",
                    ),
                    helper.make_node(
                        "Sub",
                        inputs=["attention_mask_row_sums", "one"],
                        outputs=["seqlens_k_int64"],
                        name="Sub_gqa_aux",
                    ),
                    helper.make_node(
                        "Cast",
                        inputs=["seqlens_k_int64"],
                        outputs=["seqlens_k"],
                        name="Cast_gqa_aux_0",
                        to=TensorProto.INT32,
                    ),
                    helper.make_node(
                        "Shape", inputs=["attention_mask"], outputs=["attention_mask_shape"], name="Shape_gqa_aux_0"
                    ),
                    helper.make_node(
                        "Gather",
                        inputs=["attention_mask_shape", "one"],
                        outputs=["total_seq_len_int64"],
                        name="Gather_gqa_aux_0",
                        axis=0,
                    ),
                    helper.make_node(
                        "Cast",
                        inputs=["total_seq_len_int64"],
                        outputs=["total_sequence_length"],
                        name="Cast_gqa_aux_1",
                        to=TensorProto.INT32,
                    ),
                ]
                for new_node in gqa_aux_nodes:
                    self.nodes_to_add.append(new_node)
                    self.node_name_to_graph_name[new_node.name] = self.this_graph_name
                self.model.add_initializer(
                    numpy_helper.from_array(np.array([1], dtype="int64"), name="one"), self.this_graph_name
                )

        self.set_unique_name(subgraph_nodes, layer_id, layer_known_edges_names)

        self.replace_fp32_value_info(i_hidden_states, ["batch_size", "seq_len", "hidden_size"])
        self.replace_fp32_value_info(o_hidden_states, ["batch_size", "seq_len", "hidden_size"])

        self.nodes_to_remove.append(node)
        self.prune_graph = True


def shape_of(vi):
    return tuple([d.dim_param if (d.dim_param) else d.dim_value for d in vi.type.tensor_type.shape.dim])


def postprocess_io(model: ModelProto):
    graph = model.graph
    new_inputs = []
    for i, vi in enumerate(graph.input):
        if "attention_mask" in vi.name:
            vi = helper.make_tensor_value_info(
                IO_MAPPING[vi.name],
                elem_type=TensorProto.INT32,
                shape=["batch_size", "seq_len"],
            )
            # vi_pid = helper.make_tensor_value_info(
            #     "step",
            #     elem_type=TensorProto.INT64,
            #     shape=[1],
            # )
            new_inputs.extend([vi])
        if "input_ids" in vi.name:
            vi = helper.make_tensor_value_info(
                IO_MAPPING[vi.name],
                elem_type=TensorProto.INT32,
                shape=["batch_size", "seq_len"],
            )
            new_inputs.extend([vi])
        if "kv_cache" in vi.name:
            vi = helper.make_tensor_value_info(
                IO_MAPPING[vi.name],
                elem_type=vi.type.tensor_type.elem_type,
                shape=[2, "batch_size", 32, "past_seq_len", 80],
            )
            new_inputs.extend([vi])
    # add past_sequence_length
    # vi = helper.make_tensor_value_info(
    #     "past_sequence_length",
    #     elem_type=TensorProto.INT32,
    #     shape=[],
    # )
    # new_inputs.extend([vi])

    graph.ClearField("input")
    graph.input.extend(new_inputs)

    new_outputs = []
    for i, vi in enumerate(graph.output):
        if i == 0:
            vi = helper.make_tensor_value_info(
                IO_MAPPING[vi.name], elem_type=vi.type.tensor_type.elem_type, shape=["batch_size", "seq_len", 51200]
            )
        else:
            shape = shape_of(vi)
            vi = helper.make_tensor_value_info(
                IO_MAPPING[vi.name],
                elem_type=vi.type.tensor_type.elem_type,
                shape=[2, "batch_size", 32, "total_seq_len", 80],
            )
        new_outputs.extend([vi])

    graph.ClearField("output")
    graph.output.extend(new_outputs)

    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in IO_MAPPING:
                node.input[i] = IO_MAPPING[name]
        for i, name in enumerate(node.output):
            if name in IO_MAPPING:
                node.output[i] = IO_MAPPING[name]


# def postprocess_value_info(model: ModelProto):
#     for value_info in model.graph.value_info:
#         shape = shape_of(value_info)
#         if len(shape) == 3 and shape[0] == 2:
#             print("value info: ", value_info.name, shape)
#             new_value_info = helper.make_tensor_value_info(
#                 value_info.name,
#                 elem_type=value_info.type.tensor_type.elem_type,
#                 shape=["batch_size", shape[1], shape[2]],
#             )
#             model.graph.value_info.remove(value_info)
#             model.graph.value_info.extend([new_value_info])


class PhiOnnxModel(OnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, head_size: int = 0):
        super().__init__(model)
        self.fission_transformer_block = FissionTransformerBlockPhi(self, num_heads)
        self.fission_causal_lm_head = FissionTransformerCausalLMHeadPhi(self)
        self.fuse_sln = FusionSkipLayerNormalization(self)
        self.fuse_bias_sln = FusionBiasSkipLayerNormalization(self)

    def postprocess(self):
        print("post process")
        # postprocess_io(self.model)
        # postprocess_io_split_kv(self.model, True)

    def optimize(self, options: Optional[FusionOptions] = None, add_dynamic_axes: bool = False):
        self.fission_transformer_block.apply()
        self.fission_causal_lm_head.apply()
        self.fuse_sln.apply()
        self.fuse_bias_sln.apply()
        self.postprocess()

    # def get_fused_operator_statistics(self):
    #     """
    #     Returns node count of fused operators.
    #     """
    #     op_count = {}
    #     return op_count

    # def is_fully_optimized(self, fused_op_count=None):
    #     """
    #     Returns True when the model is fully optimized.
    #     """
    #     return False
