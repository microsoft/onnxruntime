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

    def create_attention_node(self, gemm, gemm_qkv, past, present, input, output, mask, is_unidirectional):
        attention_node_name = self.model.create_node_name('GptAttention')
        attention_node = helper.make_node('Attention',
                                          inputs=[input, gemm.input[1], gemm.input[2], mask, past],
                                          outputs=[attention_node_name + "_output", present],
                                          name=attention_node_name)
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([
            helper.make_attribute("num_heads", self.num_heads),
            helper.make_attribute("unidirectional", 1 if is_unidirectional else 0)
        ])

        matmul_node = helper.make_node('MatMul',
                                       inputs=[attention_node_name + "_output", gemm_qkv.input[1]],
                                       outputs=[attention_node_name + "_matmul_output"],
                                       name=attention_node_name + "_matmul")

        add_node = helper.make_node('Add',
                                    inputs=[attention_node_name + "_matmul_output", gemm_qkv.input[2]],
                                    outputs=[output],
                                    name=attention_node_name + "_add")
        self.nodes_to_add.extend([attention_node, matmul_node, add_node])

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        past = None
        present = None
        return_indice = []
        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ['Add', 'Reshape', 'Gemm', 'Reshape', 'Reshape', 'Transpose', 'MatMul'],
            [0,      None,      0,     0,          0,         0,           0],
            output_name_to_node=output_name_to_node,
            return_indice=return_indice
            ) # yapf: disable
        if qkv_nodes is None:
            return
        (add_qkv, reshape_qkv, gemm_qkv, reshape_1, reshape_2, transpose_qkv, matmul_qkv) = qkv_nodes

        another_input = add_qkv.input[1 - return_indice[0]]

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ['Concat', 'Transpose', 'Reshape', 'Split', 'Reshape', 'Gemm', 'Reshape'],
            [1,        1,            0,         0,       0,         0,      0]) # yapf: disable
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (concat_v, transpose_v, reshape_v, split_v, reshape_after_gemm, gemm, reshape_before_gemm) = v_nodes

        #      concat <-- Gather(indices=1) <-- past
        #        |
        #      unsqueeze
        #        |
        #    concat  -->  present
        gather_v = self.model.get_parent(concat_v, 0, output_name_to_node)
        if gather_v.op_type != 'Gather':
            logger.info("expect Gather for past")
            return
        if not self.model.find_constant_input(gather_v, 1) == 1:
            logger.info("expect indices=1 for Gather of past")
            return
        past = gather_v.input[0]
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
        present = concat_present.output[0]
        if not self.model.find_graph_output(present):
            logger.info("expect present to be graph input")
            return

        layernorm_before_attention = self.model.get_parent(reshape_before_gemm, 0, output_name_to_node)
        if layernorm_before_attention is None or layernorm_before_attention.op_type != 'LayerNormalization':
            logger.debug(f"failed to get layernorm before gemm. Got {layernorm_before_attention.op_type}")
            return

        if not another_input in layernorm_before_attention.input:
            logger.debug("Add and LayerNormalization shall have one same input")
            return

        is_unidirectional = True
        slice_mask = None
        input_mask_nodes = None
        qk_nodes = self.model.match_parent_path(matmul_qkv, ['Softmax', 'Sub', 'Mul', 'Div', 'MatMul'], [0, 0, 0, 0, 0])
        if qk_nodes is not None:
            (softmax_qk, sub_qk, mul_qk, div_qk, matmul_qk) = qk_nodes
            mask_nodes = self.model.match_parent_path(
                sub_qk,
                ['Mul', 'Sub', 'Slice', 'Slice', 'Unsqueeze', 'Sub', 'Squeeze', 'Slice', 'Shape', 'Div'],
                [1,      0,     1,       0,       1,           0,     0,         0,       0,       0])  # yapf: disable
            if mask_nodes is None:
                logger.debug("fuse_attention: failed to match unidirectional mask path")
                return
            div_mask = mask_nodes[-1]
            slice_mask = mask_nodes[3]

            if div_qk != div_mask:
                logger.debug("fuse_attention: skip since div_qk != div_mask")
                return
        else:
            # New pattern for gpt2 from PyTorch 1.5.0 and Transformers 2.9.0.
            i, qk_nodes, _ = self.model.match_parent_paths(
                matmul_qkv, [(['Softmax', 'Where', 'Div', 'MatMul'], [0, 0, 1, 0]),
                             (['Softmax', 'Add', 'Where', 'Div', 'MatMul'], [0, 0, 0, 1, 0])], output_name_to_node)
            if qk_nodes is None:
                logger.debug("fuse_attention: failed to match qk nodes")
                return

            where_qk = qk_nodes[-3]
            div_qk = qk_nodes[-2]
            matmul_qk = qk_nodes[-1]

            if i == 1:
                add_qk = qk_nodes[1]
                _, input_mask_nodes, _ = self.model.match_parent_paths(
                    add_qk, [(['Mul', 'Sub', 'Cast', 'Unsqueeze', 'Unsqueeze', 'Reshape'], [1, 0, 1, 0, 0, 0]),
                             (['Mul', 'Sub', 'Unsqueeze', 'Unsqueeze', 'Reshape'], [1, 0, 1, 0, 0])],
                    output_name_to_node)
                if input_mask_nodes is None:
                    logger.debug("fuse_attention: failed to match input attention mask path")
                    return

            mask_nodes = self.model.match_parent_path(
                where_qk,
                ['Cast', 'Slice', 'Slice', 'Unsqueeze', 'Sub', 'Squeeze', 'Slice', 'Shape', 'Div'],
                [ 0,     0,       0,       1,           0,     0,         0,       0,       0])  # yapf: disable
            if mask_nodes is None:
                logger.debug("fuse_attention: failed to match mask path")
                return
            div_mask = mask_nodes[-1]
            slice_mask = mask_nodes[2]

            if div_qk != div_mask:
                logger.debug("fuse_attention: skip since div_qk != div_mask")
                return

        # Validate that the mask data is either lower triangular (unidirectional) or all ones
        mask_data = numpy_helper.to_array(self.model.get_initializer(slice_mask.input[0]))
        if not (len(mask_data.shape) == 4 and mask_data.shape[:2] == (1, 1)
                and mask_data.shape[2] == mask_data.shape[3]):
            logger.debug("fuse_attention: skip since mask shape is not 1x1xWxW")
            return
        if np.allclose(mask_data, np.ones_like(mask_data)):
            is_unidirectional = False
        elif not np.allclose(mask_data, np.tril(np.ones_like(mask_data))):
            logger.debug("fuse_attention: skip since mask is neither lower triangular nor ones")
            return

        q_nodes = self.model.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Split'], [0, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        (transpose_q, reshape_q, split_q) = q_nodes
        if split_v != split_q:
            logger.debug("fuse_attention: skip since split_v != split_q")
            return

        k_nodes = self.model.match_parent_path(matmul_qk, ['Concat', 'Transpose', 'Reshape', 'Split'], [1, 1, 0, 0])
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return
        (concat_k, transpose_k, reshape_k, split_k) = k_nodes
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
        past_k_nodes = self.model.match_parent_path(concat_k, ['Transpose', 'Gather'], [0, 0])
        if past_k_nodes is None:
            logger.debug("fuse_attention: failed to match past_k_nodes path")
            return

        gather_past_k = past_k_nodes[-1]
        if not self.model.find_constant_input(gather_past_k, 0) == 1:
            logger.info("expect indices=0 for Gather k of past")
            return
        past_k = gather_past_k.input[0]
        if past != past_k:
            logger.info("expect past to be same")
            return

        attention_mask_input_name = ''
        if input_mask_nodes is not None:
            input_name = input_mask_nodes[-1].input[0]
            if input_name in self.casted_attention_mask:
                attention_mask_input_name = self.casted_attention_mask[input_name]
            elif self.model.find_graph_input(input_name):
                casted, attention_mask_input_name = self.utils.cast_graph_input_to_int32(input_name)
                self.casted_attention_mask[input_name] = attention_mask_input_name
            else:
                attention_mask_input_name, cast_node = self.utils.cast_input_to_int32(input_name)
                self.casted_attention_mask[input_name] = attention_mask_input_name

        self.create_attention_node(gemm, gemm_qkv, past, present, layernorm_before_attention.output[0],
                                   reshape_qkv.output[0], attention_mask_input_name, is_unidirectional)

        # we rely on prune_graph() to clean old subgraph nodes:
        # qk_nodes + q_nodes + k_nodes + v_nodes + mask_nodes + [reshape_qkv, transpose_qkv, matmul_qkv]
        self.prune_graph = True
