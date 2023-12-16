# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import List, Optional
from fusion_base import Fusion
from fusion_utils import FusionUtils, NumpyHelper
from onnx import GraphProto, ModelProto, TensorProto, ValueInfoProto, helper, inliner
from onnx_model import OnnxModel
from fusion_options import FusionOptions
import numpy as np

logger = getLogger(__name__)

# TODO: handle the hard-coded values
class ProcessGemmWFunc:
    def __call__(self, x):
        return np.transpose(x, (1, 0))

class ProcessAttnWFunc:
    def __call__(self, x):
        x = np.reshape(x, (32, 3, -1))
        x = np.transpose(x, (1, 0, 2))
        x = np.reshape(x, (7680, -1))
        x = np.transpose(x, (1, 0))
        return x

class ProcessAttnBFunc:
    def __call__(self, x):
        x = np.reshape(x, (32, 3, -1))
        x = np.transpose(x, (1, 0, 2))
        x = np.reshape(x, (-1))
        return x

class PostProcessCausalLMHead(Fusion):
    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, "DONOTUSE", ["model_modeling_mixformer_sequential_CausalLMHead_layers__1_1"])

    def fuse(
            self,
            node,
            input_name_to_nodes,
            output_name_to_node,
    ):
        # hack: remove input dulication
        node.input[0] = node.input[1]

class FissionTransformerBlockPhi(Fusion):
    def __init__(
        self,
        model: OnnxModel,
    ):
        self.func_to_layer_id = {}
        nodes_to_find = []
        for layer in range(32):
            func_name = f"model_modeling_mixformer_sequential_ParallelBlock_sub{layer + 1}_1"
            nodes_to_find.append(func_name)
            self.func_to_layer_id[func_name] = layer + 1

        super().__init__(model, "DONOTUSE", nodes_to_find)

    def uname(self, layer_id, name):
        return name + "_" + str(layer_id)

    def get_layer_id(self, node):
        return self.func_to_layer_id[node.op_type]

    def process_initializer(self, initializer_name, functor):
        i = self.model.get_initializer(initializer_name)
        i_np_array = NumpyHelper.to_array(i)
        processed_i_np_array = functor(i_np_array)
        new_tensor = helper.make_tensor(
            initializer_name + "_processed",
            data_type = TensorProto.FLOAT,
            dims = processed_i_np_array.shape,
            vals = processed_i_np_array.flatten().tobytes(),
            raw = True,
        )
        self.model.add_initializer(new_tensor, self.this_graph_name)
        return new_tensor.name

    def fuse_with_attn(
            self,
            node,
            input_name_to_nodes,
            output_name_to_node,
    ):
        layer_id = self.get_layer_id(node)
        print(f"process layer {layer_id}")

        # transformer block input and output
        i_hidden_states = node.input[0] if layer_id == 1 else node.input[2]
        i_attn_mask = node.input[1]
        i_kv_cache = node.input[3]
        o_hidden_states = node.output[1]
        o_kv_cache = node.output[0]

        # internal nodes weights
        ln_weight = node.input[4] #float32[2560]
        ln_bias = node.input[5] #float32[2560]
        attn_qkv_weight = self.process_initializer(node.input[6], ProcessAttnWFunc()) #float32[7680,2560]
        attn_qkv_bias = self.process_initializer(node.input[7], ProcessAttnBFunc()) #float32[7680]
        attn_out_weight = self.process_initializer(node.input[10], ProcessGemmWFunc()) #float32[2560,2560]
        attn_out_bias = node.input[11] #float32[2560]
        mlp_fc1_weight = self.process_initializer(node.input[12], ProcessGemmWFunc()) #float32[10240,2560]
        mlp_fc1_bias = node.input[13] #float32[10240]
        mlp_fc2_weight = self.process_initializer(node.input[14], ProcessGemmWFunc()) #float32[2560,10240]
        mlp_fc2_bias = node.input[15] #float32[2560]

        # opt graph construction.
        subgraph_nodes = [
            helper.make_node(
                'LayerNormalization',
                inputs=[i_hidden_states, ln_weight, ln_bias],
                outputs=[self.uname(layer_id, 'ln_out')],
                name=self.uname(layer_id, 'LayerNormalization'),
                epsilon=9.999999747378752e-06,
            ),
            helper.make_node(
                'Cast',
                inputs=[i_attn_mask],
                outputs=[self.uname(layer_id, 'casted_mask')],
                name=self.uname(layer_id, 'Cast'),
                to=6,
            ),
            helper.make_node(
                'Attention',
                inputs=[self.uname(layer_id, 'ln_out'), attn_qkv_weight, attn_qkv_bias, self.uname(layer_id, 'casted_mask'), i_kv_cache],
                outputs=[self.uname(layer_id, 'attn_out'), o_kv_cache],
                name=self.uname(layer_id, 'Attention'),
                domain='com.microsoft',
                num_heads=32,
                unidirectional=1,
                do_rotary=1,
                rotary_embedding=32,
                #past_present_share_buffers=1,
            ),
            helper.make_node(
                'MatMul',
                inputs=[self.uname(layer_id, 'attn_out'), attn_out_weight],
                outputs=[self.uname(layer_id, 'matmul_out')],
                name=self.uname(layer_id, 'OutProj_MatMul'),
            ),
            helper.make_node(
                'Add',
                inputs=[self.uname(layer_id, 'matmul_out'), attn_out_bias],
                outputs=[self.uname(layer_id, 'add_out')],
                name=self.uname(layer_id, 'OutProj_Add'),
            ),
            helper.make_node(
                'MatMul',
                inputs=[self.uname(layer_id, 'ln_out'), mlp_fc1_weight],
                outputs=[self.uname(layer_id, 'fc1_w_out')],
                name=self.uname(layer_id, 'FC1_MatMul'),
            ),
            helper.make_node(
                'Add',
                inputs=[self.uname(layer_id, 'fc1_w_out'), mlp_fc1_bias],
                outputs=[self.uname(layer_id, 'fc1_b_out')],
                name=self.uname(layer_id, 'FC1_Bias'),
            ),
            helper.make_node(
                'FastGelu',
                inputs=[self.uname(layer_id, 'fc1_b_out')],
                outputs=[self.uname(layer_id, 'new_gelu_out')],
                name=self.uname(layer_id, 'FastGelu'),
                domain='com.microsoft',
            ),
            helper.make_node(
                'MatMul',
                inputs=[self.uname(layer_id, 'new_gelu_out'), mlp_fc2_weight],
                outputs=[self.uname(layer_id, 'fc2_w_out')],
                name=self.uname(layer_id, 'FC2_MatMul'),
            ),
            helper.make_node(
                'Add',
                inputs=[self.uname(layer_id, 'fc2_w_out'), mlp_fc2_bias],
                outputs=[self.uname(layer_id, 'fc2_b_out')],
                name=self.uname(layer_id, 'FC2_Bias'),
            ),
            helper.make_node(
                'Add',
                inputs=[self.uname(layer_id, 'add_out'), self.uname(layer_id, 'fc2_b_out')],
                outputs=[self.uname(layer_id, 'residual_1_out')],
                name=self.uname(layer_id, 'Residual_Add_1'),
            ),
            helper.make_node(
                'Add',
                inputs=[i_hidden_states, self.uname(layer_id, 'residual_1_out')],
                outputs=[o_hidden_states],
                name=self.uname(layer_id, 'Residual_Add_2'),
            ),
        ]

        for new_node in subgraph_nodes:
            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.append(node)
        self.prune_graph = True

    def fuse(
            self,
            node,
            input_name_to_nodes,
            output_name_to_node,
    ):
        self.fuse_with_attn(node, input_name_to_nodes, output_name_to_node)


class PhiOnnxModel(OnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, head_size: int = 0):
        super().__init__(model)
        self.fission_transformer_block = FissionTransformerBlockPhi(self)
        self.postprocess_causal_lm_head = PostProcessCausalLMHead(self)

    def inline_model(self):
        self.model = inliner.inline_local_functions(self.model, False)

    def postprocess(self):
        self.prune_graph()

    def optimize(self, options: Optional[FusionOptions] = None, add_dynamic_axes: bool = False):
        self.fission_transformer_block.apply()
        self.postprocess_causal_lm_head.apply()
        #self.inline_model()

    def get_fused_operator_statistics(self):
        """
        Returns node count of fused operators.
        """
        op_count = {}
        return op_count

    def is_fully_optimized(self, fused_op_count=None):
        """
        Returns True when the model is fully optimized.
        """
        return False
