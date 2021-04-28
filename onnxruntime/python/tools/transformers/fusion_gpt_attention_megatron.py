#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
import numpy as np
from logging import getLogger
from onnx import helper, numpy_helper, TensorProto
from onnx_model import OnnxModel
from fusion_base import Fusion
from fusion_utils import FusionUtils

logger = getLogger(__name__)


class FusionGptAttention(Fusion):
    """
    Fuse GPT-2 Attention with past state subgraph into one Attention node.
    This does not support attention_mask graph input right now.
    """
    def __init__(self, model: OnnxModel, num_heads: int):
        super().__init__(model, "Attention", "LayerNormalization", "with past")
        # TODO: detect num_heads from graph like FusionAttention
        self.num_heads = num_heads
        self.utils = FusionUtils(model)
        self.casted_attention_mask = {}  # map from name of attention mask to the name that casted to int32

    def create_attention_node(self, matmul, add, matmul_qkv, add_qkv, past, present, input, output, mask, is_unidirectional):
        attention_node_name = self.model.create_node_name('GptAttention')
        attention_node = helper.make_node('Attention',
                                          inputs=[input, matmul.input[1], add.input[1], mask, past],
                                          outputs=[attention_node_name + "_output", present],
                                          name=attention_node_name)
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([
            helper.make_attribute("num_heads", self.num_heads),
            helper.make_attribute("unidirectional", 1 if is_unidirectional else 0)
        ])

        matmul_node = helper.make_node('MatMul',
                               inputs=[attention_node_name + "_output", matmul_qkv.input[1]],
                               outputs=[attention_node_name + "_matmul_output"],
                               name=attention_node_name + "_matmul")

        add_node = helper.make_node('Add',
                                    inputs=[attention_node_name + "_matmul_output", add_qkv.input[1]],
                                    outputs=[output],
                                    name=attention_node_name + "_add")
        self.nodes_to_add.extend([attention_node, matmul_node, add_node])

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        past = None
        present = None

        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ['Cast', 'Add', 'Add', 'MatMul', 'Reshape', 'Transpose', 'MatMul'],
            [0,      0,      1,     0,          0,         0,           0],
            output_name_to_node=output_name_to_node,
            ) # yapf: disable
        if qkv_nodes is None:
            return
        (_, add_all_2, add_all, matmul_all, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

        another_input = add_all_2.input[0]

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ['Concat', 'Transpose', 'Reshape', 'Split', 'Add', 'MatMul', 'LayerNormalization'],
            [1,        1,            0,         0,       0,         0,      0]) # yapf: disable
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (concat_v, transpose_v, reshape_v, split_v, add_together, matmul_together, layernorm) = v_nodes

        past_nodes = self.model.match_parent_path(
            concat_v,
            ['Squeeze', 'Split'],
            [0,        0]) # yapf: disable
        (squeeze_past, split_past) = past_nodes
        past = split_past.input[0]
        if not self.model.find_graph_input(past):
            logger.info("expect past to be graph input")
            return
        unsqueeze_present_v = self.model.find_first_child_by_type(concat_v,
                                                                  'Unsqueeze',
                                                                  input_name_to_nodes,
                                                                  recursive=False)
        if not unsqueeze_present_v:
            logger.info("expect unsqueeze for present")
            return
        concat_present = self.model.find_first_child_by_type(unsqueeze_present_v,
                                                             'Concat',
                                                             input_name_to_nodes,
                                                             recursive=False)
        if not concat_present:
            logger.info("expect concat for present")
            return
        #cast_present = self.model.find_first_child_by_type(concat_present,
        #                                                   'Cast',
        #                                                   input_name_to_nodes,
        #                                                   recursive=False)
        #if not cast_present:
        #    logger.info("expect concat for present")
        #    return
        present = concat_present.output[0]
        #if not self.model.find_graph_output(present):
        #    logger.info("expect present to be graph input")
        #    return

        layernorm_before_attention = layernorm
        if layernorm_before_attention is None or layernorm_before_attention.op_type != 'LayerNormalization':
            logger.debug(f"failed to get layernorm before gemm. Got {layernorm_before_attention.op_type}")
            return

        is_unidirectional = True
        slice_mask = None
        input_mask_nodes = None
        qk_nodes = self.model.match_parent_path(matmul_qkv, ['Cast', 'Softmax', 'Sub', 'Mul', 'Cast', 'MatMul'], [0, 0, 0, 0, 0, 0])
        if qk_nodes is not None:
            (cast_qk, softmax_qk, sub_qk, mul_qk, cast_qk2, matmul_qk) = qk_nodes
            mask_nodes = self.model.match_parent_path(
                sub_qk,
                ['Mul', 'Sub', 'Cast', 'Unsqueeze', 'Slice', 'Gather'],
                [1,      0,     1,       0,       0,           0])  # yapf: disable
            if mask_nodes is None:
                logger.debug("fuse_attention: failed to match unidirectional mask path")
                return

        gather_mask = mask_nodes[-1]

        self.model.add_node(
            helper.make_node(
                "Cast",
                ["attention_mask"],
                ['added_cast_output' + gather_mask.name],
                'added_cast' + gather_mask.name,
                to=6
            )
        )
        init_axes = helper.make_tensor('added_squeeze_axes', TensorProto.INT64, [1], [1])
        self.model.add_initializer(init_axes)
        self.model.add_node(
            helper.make_node(
                "Squeeze",
                #inputs = ['added_cast_output' + gather_mask.name, 'added_squeeze_axes'],
                inputs = ['added_cast_output' + gather_mask.name],
                outputs = ['added_squeeze_output' + gather_mask.name],
                axes=[1],
                name = 'added_squeeze' + gather_mask.name)
        )
        gather_mask.input[0] = 'added_squeeze_output' + gather_mask.name

        attention_mask = gather_mask.input[0]

        q_nodes = self.model.match_parent_path(matmul_qk, ['Div', 'Transpose', 'Reshape', 'Split'], [0, 0, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        (_, transpose_q, reshape_q, split_q) = q_nodes
        if split_v != split_q:
            logger.debug("fuse_attention: skip since split_v != split_q")
            return

        k_nodes = self.model.match_parent_path(matmul_qk, ['Div', 'Transpose', 'Concat', 'Transpose', 'Reshape', 'Split'], [1, 0, 0, 1, 0, 0])
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return
        (_, _, concat_k, transpose_k, reshape_k, split_k) = k_nodes
        if split_v != split_k:
            logger.debug("fuse_attention: skip since split_v != split_k")
            return


        #     concat_k <-- Transpose (perm=0,1,3,2) <-- Gather(axes=0, indices=0) <-- past
        #        |
        #     Transpose (perm=0,1,3,2)
        #        |
        #      unsqueeze
        #        |
        #    concat  -->  present
        past_k_nodes = self.model.match_parent_path(concat_k, ['Squeeze', 'Split'], [0, 0])
        if past_k_nodes is None:
            logger.debug("fuse_attention: failed to match past_k_nodes path")
            return

        split_past_k = past_k_nodes[-1]

        attention_mask_input_name = attention_mask

        self.create_attention_node(matmul_together, add_together, matmul_all, add_all,  past, present, layernorm_before_attention.output[0],
                                   reshape_qkv.output[0], attention_mask_input_name, is_unidirectional)

        # we rely on prune_graph() to clean old subgraph nodes:
        # qk_nodes + q_nodes + k_nodes + v_nodes + mask_nodes + [reshape_qkv, transpose_qkv, matmul_qkv]
        self.prune_graph = True
