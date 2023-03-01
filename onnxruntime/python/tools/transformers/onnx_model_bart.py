# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from fusion_attention import AttentionMask, FusionAttention
from fusion_options import FusionOptions
from fusion_reshape import FusionReshape
from onnx import numpy_helper, TensorProto, helper
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel
from typing import Optional

logger = logging.getLogger(__name__)


class FusionBartAttention(FusionAttention):
    """
    Fuse Bart Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
        fuse_mha_bias: bool = False,
    ):
        super().__init__(model, hidden_size, num_heads, attention_mask)
        self.fuse_mha_bias = fuse_mha_bias

    def check_runtime_shape_path(
        self,
        reshape_qkv_2,
        reshape_qkv_1,
        reshape_q_2,
        reshape_k_2,
        reshape_v_2,
        root_input,
    ):
        concat_qkv_2_path = self.model.match_parent_path(reshape_qkv_2, ["Concat"], [1])
        if concat_qkv_2_path is None:
            print("fail 1")
            return False
        concat_qkv_2 = concat_qkv_2_path[0]

        reshape_qkv_2_path_1 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_qkv_2_path_2 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        #reshape_qkv_2_path_3 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [2, 0, 0])
        if reshape_qkv_2_path_1 is None or reshape_qkv_2_path_2 is None: #or reshape_qkv_2_path_3 is None:
            print("fail 2")
            return False

        _, gather_1, shape_1 = reshape_qkv_2_path_1
        _, gather_2, shape_2 = reshape_qkv_2_path_2
        #_, _, shape_3 = reshape_qkv_2_path_3

        if shape_1.input[0] != root_input or shape_2.input[0] != root_input: # or shape_3.input[0] != root_input:
            print(shape_1.name, shape_2.name, root_input)
            print("fail 3")
            return False

        reshape_qkv_1_path_1 = self.model.match_parent_path(reshape_qkv_1, ["Concat", "Unsqueeze", "Gather"], [1, 0, 0])
        reshape_qkv_1_path_2 = self.model.match_parent_path(reshape_qkv_1, ["Concat", "Unsqueeze", "Gather"], [1, 2, 0])
        if reshape_qkv_1_path_1 is None or reshape_qkv_1_path_2 is None:
            print("fail 4")
            return False
        if reshape_qkv_1_path_1[-1].name != gather_1.name or reshape_qkv_1_path_2[-1].name != gather_2.name:
            print("fail 5")
            return False

        reshape_q_2_path = self.model.match_parent_path(reshape_q_2, ["Concat", "Unsqueeze", "Mul"], [1, 0, 0])
        reshape_k_2_path = self.model.match_parent_path(reshape_k_2, ["Concat", "Unsqueeze", "Mul"], [1, 0, 0])
        reshape_v_2_path = self.model.match_parent_path(reshape_v_2, ["Concat", "Unsqueeze", "Mul"], [1, 0, 0])
        if reshape_q_2_path is None or reshape_k_2_path is None or reshape_v_2_path is None:
            print("fail 6")
            return False

        mul_q = reshape_q_2_path[-1]
        mul_k = reshape_k_2_path[-1]
        mul_v = reshape_v_2_path[-1]

        gather_1_out = gather_1.output[0]
        if mul_q.input[0] != gather_1_out or mul_k.input[0] != gather_1_out or mul_v.input[0] != gather_1_out:
            print("fail 7")
            return False

        return True

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["Add", "MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
            [1, 1, 0, 0, 0, 0],
        )
        if qkv_nodes is not None:
            (
                add_out,
                matmul_out,
                reshape_qkv_2,
                transpose_qkv,
                reshape_qkv_1,
                matmul_qkv,
            ) = qkv_nodes
        else:
            return

        #import os
        #if not os.path.exists('temp.onnx'):
        #    self.model.save_model_to_file('temp.onnx', use_external_data_format=True)
        
        other_inputs = []
        for i, input in enumerate(normalize_node.input):
            if input not in output_name_to_node:
                continue
            if input == qkv_nodes[0].output[0]:
                continue
            other_inputs.append(input)
        if len(other_inputs) != 1:
            return
        root_input = other_inputs[0]
        #print("Init root input:", root_input)

        # Sometimes the input name to the attention MatMul nodes does not match the input name to the end 
        # SkipLayerNormalization node (name saved in root_input). We find the true input name to the MatMul
        # nodes by getting the initial SkipLayerNormalization node and checking how many MatMul nodes are
        # children nodes for each of its output names.
        """
                                        root_input
                    +---------------------------------------------------+
                    |                                                   |
                    |                                                   |
        SkipLayerNormalization --> Attention --> MatMul --> SkipLayerNormalization
        """
        skip_layernorm = output_name_to_node[root_input]
        # For some attention blocks, the end SkipLayerNormalization node may point to an Add node whose
        # child is the SkipLayerNormalization node.
        if skip_layernorm.op_type == "Add":
            skip_layernorm = self.model.get_children(skip_layernorm)[0]
        for i, output in enumerate(skip_layernorm.output):
            if output == "":
                continue
            children = input_name_to_nodes[output]
            children_types = [child.op_type for child in children]
            if children_types.count("MatMul") >= 1:
                root_input = output
                break

        #print("Final root input:", root_input)
        graph_input_names = [node.name for node in self.model.graph().input]
        graph_output_names = [node.name for node in self.model.graph().output]

        #print("checkpoint 2")
        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Reshape", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 0, 0, None],
        )
        v_nodes_with_past_decoder = self.model.match_parent_path(
            matmul_qkv,
            ["Reshape", "Concat", "Transpose", "Reshape", "Add", "MatMul"], # Decoder attention with past decoder value
            [1, 0, 1, 0, 0, None],
        )
        v_nodes_with_past_encoder = self.model.match_parent_path(
            matmul_qkv,
            ["Reshape"], # Decoder attention with past encoder value
            [1],
        )
        past_v, present_v = "", ""
        if v_nodes is not None:
            (reshape_v_2, transpose_v, reshape_v_1, add_v, matmul_v) = v_nodes
            present_v = transpose_v.output[0] # For initial pass through encoder-decoder_with_past to get starting past values (beam search)
        elif v_nodes_with_past_decoder is not None:
            (reshape_v_2, concat_v, transpose_v, reshape_v_1, add_v, matmul_v) = v_nodes_with_past_decoder
            v_nodes = v_nodes_with_past_decoder
            past_v = concat_v.input[0]
            present_v = concat_v.output[0]
        elif v_nodes_with_past_encoder is not None and v_nodes_with_past_encoder[-1].input[0] in graph_input_names:
            v_nodes = v_nodes_with_past_encoder
            past_v = v_nodes[-1].input[0]
            identity_node_v = list(filter(lambda node: node.op_type == "Identity", self.model.input_name_to_nodes()[past_v]))
            present_v = identity_node_v[0].output[0]
        else:
            logger.debug("fuse_attention: failed to match v path")
            return
        past_v = past_v if past_v in graph_input_names else ""
        present_v = present_v if present_v in graph_output_names else ""

        #print("checkpoint 3")
        qk_nodes_1 = self.model.match_parent_path(matmul_qkv, ["Softmax", "MatMul"], [0, 0])
        qk_nodes_2 = self.model.match_parent_path(matmul_qkv, ["Softmax", "Reshape", "Add", "Reshape", "MatMul"], [0, 0, 0, 0, 0])
        if qk_nodes_1 is not None:
            _, matmul_qk = qk_nodes_1
            qk_nodes = qk_nodes_1
        elif qk_nodes_2 is not None:
            _, _, add_qk, _, matmul_qk = qk_nodes_2
            qk_nodes = qk_nodes_2
        else:
            return

        #print("checkpoint 4")
        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Reshape", "Transpose", "Reshape", "Mul", "Add", "MatMul"],
            [0, 0, 0, 0, 0, 1],
        )
        if q_nodes is not None:
            reshape_q_2, transpose_q, reshape_q_1, mul_q, add_q, matmul_q = q_nodes
        else:
            return

        #print("checkpoint 5")
        k_nodes_with_bias = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "Transpose", "Reshape", "Add", "MatMul"], # K path with bias=True
            [1, 0, 0, 0, 0, 1],
        )
        k_nodes_no_bias = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "Transpose", "Reshape", "MatMul"], # K path with bias=False
            [1, 0, 0, 0, 0],
        )
        k_nodes_no_bias_with_past_decoder = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "Concat", "Transpose", "Reshape", "MatMul"], # K path with bias=False for decoder with past decoder value
            [1, 0, 0, 1, 0, 0],
        )
        k_nodes_no_bias_with_past_encoder = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape"], # K path with bias=False for decoder with past encoder value
            [1, 0],
        )
        past_k, present_k = "", ""
        if k_nodes_with_bias is not None:
            _, reshape_k_2, transpose_k_1, reshape_k_1, add_k, matmul_k = k_nodes_with_bias
            k_nodes = k_nodes_with_bias
        elif k_nodes_no_bias is not None:
            _, reshape_k_2, transpose_k_1, reshape_k_1, matmul_k = k_nodes_no_bias
            k_nodes = k_nodes_no_bias
            present_k = transpose_k_1.output[0] # For initial pass through encoder-decoder_with_past to get starting past values (beam search) 
        elif k_nodes_no_bias_with_past_decoder is not None:
            _, reshape_k_2, concat_k, _, reshape_k_1, matmul_k = k_nodes_no_bias_with_past_decoder
            k_nodes = k_nodes_no_bias_with_past_decoder
            past_k = concat_k.input[0]
            present_k = concat_k.output[0]
        elif k_nodes_no_bias_with_past_encoder is not None and k_nodes_no_bias_with_past_encoder[-1].input[0] in graph_input_names:
            k_nodes = k_nodes_no_bias_with_past_encoder 
            past_k = k_nodes[-1].input[0]
            identity_node_k = list(filter(lambda node: node.op_type == "Identity", self.model.input_name_to_nodes()[past_k]))
            present_k = identity_node_k[0].output[0]
        else:
            return
        past_k = past_k if past_k in graph_input_names else ""
        present_k = present_k if present_k in graph_output_names else ""

        if k_nodes == k_nodes_no_bias or k_nodes == k_nodes_no_bias_with_past_decoder:
            # Create empty Add node for attention graph
            bias_dim = self.model.get_initializer(add_v.input[0]).dims[0]
            empty_bias_name = "empty_bias"
            empty_tensor = self.model.get_initializer(empty_bias_name)
            if empty_tensor == None:
                empty_tensor = helper.make_tensor(empty_bias_name, TensorProto.FLOAT, [bias_dim], [0.0] * bias_dim)
                self.model.add_initializer(empty_tensor, self.this_graph_name)

            add_name = self.model.create_node_name("Add")
            add_k = helper.make_node("Add", [empty_bias_name, matmul_k.output[0]], [reshape_k_1.name], add_name)
            # matmul_k.output[0] = add_name
            # reshape_k_1.input[0] = add_name
            # self.nodes_to_add.append(add_k)
            # self.node_name_to_graph_name[add_name] = self.this_graph_name

        # if k_nodes == k_nodes_no_bias_with_past_encoder and v_nodes == v_nodes_with_past_encoder:
        #     # Create empty Add node for attention graph
        #     bias_dim = self.model.get_initializer(add_q.input[0]).dims[0]
        #     empty_bias_name = "empty_bias"
        #     empty_tensor = self.model.get_initializer(empty_bias_name)
        #     if empty_tensor == None:
        #         empty_tensor = helper.make_tensor(empty_bias_name, TensorProto.FLOAT, [bias_dim], [0.0] * bias_dim)
        #         self.model.add_initializer(empty_tensor, self.this_graph_name)

        #     add_name = self.model.create_node_name("Add")
        #     add_k = helper.make_node("Add", [empty_bias_name, empty_bias_name], ["add_k_dummy_output"], add_name)
        #     add_v = add_k

        #print("checkpoint 6")
        if past_k == "" and not self.check_runtime_shape_path(
            reshape_qkv_2,
            reshape_qkv_1,
            reshape_q_2,
            reshape_k_2,
            reshape_v_2,
            root_input,
        ):
            return

        #print("checkpoint 7")
        three_root_inputs = past_k != "" and past_v != "" and "matmul_k" not in locals() and "matmul_v" not in locals()
        one_root_input = not three_root_inputs and matmul_k.input[0] == root_input and matmul_q.input[0] == root_input and matmul_v.input[0] == root_input 
        two_root_inputs = not three_root_inputs and matmul_q.input[0] == root_input and matmul_k.input[0] == matmul_v.input[0] and matmul_k.input[0] != matmul_q.input[0]
        
        # There are 4 types of attention:
        # 1) Encoder attention with one_root_input=True and qk_nodes=qk_nodes_1
        # 2) Decoder attention with one_root_input=True and qk_nodes=qk_nodes_2
        # 3) Decoder cross attention with two_root_inputs=True and qk_nodes=qk_nodes_1
        # 4) Decoder cross attention with past encoder with three_root_inputs=True and qk_nodes=qk_nodes_1
        encoder_attention = one_root_input and qk_nodes == qk_nodes_1
        decoder_attention = one_root_input and qk_nodes == qk_nodes_2
        decoder_cross_attention = two_root_inputs and qk_nodes == qk_nodes_1
        decoder_cross_attention_with_past = three_root_inputs and qk_nodes == qk_nodes_1

        # For decoder_attention, the attention mask needs to be included in the attention node
        mask_index = None
        if decoder_attention:
            mask_nodes = self.model.match_parent_path(
                add_qk,
                ["Expand", "Unsqueeze", "Unsqueeze", "Where"],
                [1, 0, 0, 0],
            )
            mask_index = mask_nodes[0].output[-1]
            # print("Add QK:", add_qk)
            # print("Attn mask:", mask_nodes[0])
            # mask_index = self.attention_mask.process_mask(mask_nodes[0].output[-1])
            # self.attention_mask_index = mask_index
            # print(mask_index)

        # mha_inputs_no_bias_fused = []
        # if decoder_cross_attention:
        #     # These inputs for multihead attention are for when the bias is not fused
        #     mha_inputs_no_bias_fused = [
        #         transpose_q.output[0],
        #         transpose_k_1.output[0],
        #         transpose_v.output[0],
        #     ]

        if decoder_cross_attention or decoder_cross_attention_with_past:
            # Skip the 1/sqrt(H) Mul in the Q path
            reshape_q_1.input[0] = add_q.output[0]

        if encoder_attention or decoder_attention or decoder_cross_attention or decoder_cross_attention_with_past: 
            attention_last_node = reshape_qkv_2
            num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_q_1)

            if num_heads <= 0 or hidden_size <= 0 or (hidden_size % num_heads) != 0:
                logger.debug("fuse_attention: failed to detect num_heads or hidden_size")
                return
            
            new_node = None
            print("Past/present:", past_k, past_v, present_k, present_v)
            print("Attention type:", encoder_attention, decoder_attention, decoder_cross_attention, decoder_cross_attention_with_past)

            if decoder_cross_attention and not self.fuse_mha_bias:
                # Check if present_k and present_v are calculated before or after MHA
                present_k = "" if present_k == transpose_k_1.output[0] else present_k
                present_v = "" if present_v == transpose_v.output[0] else present_v
                new_node = self.create_multihead_attention_node(
                    transpose_q.output[0],
                    transpose_k_1.output[0],
                    transpose_v.output[0],
                    num_heads,
                    hidden_size,
                    attention_last_node.output[0],
                    present_kv=present_k.replace(".key", "").replace("_key", "").replace(".", "_"),
                    present_k=present_k,
                    present_v=present_v,
                )
            elif decoder_cross_attention_with_past:
                present_k = "" if present_k == identity_node_k[0].output[0] else present_k
                present_v = "" if present_v == identity_node_v[0].output[0] else present_v
                # print("Past/present redone:", past_k, past_v, present_k, present_v)
                new_node = self.create_multihead_attention_node(
                    matmul_q.output[0] if self.fuse_mha_bias else transpose_q.output[0],
                    past_k,
                    past_v,
                    num_heads,
                    hidden_size,
                    attention_last_node.output[0],
                    present_kv=present_k.replace(".key", "").replace("_key", "").replace(".", "_"), #past_k.replace("past_key_values", "present").replace(".key", "").replace(".", "_"),
                    present_k=present_k,
                    present_v=present_v,
                )
            #print(past_k, past_v, present_k, present_v)
            # if decoder_cross_attention_with_past:
            #     q_input = matmul_q.output[0] if self.fuse_mha_bias else add_q.output[0]
            #     present_kv_name = past_k.replace("past_key_values", "present").replace(".", "_")[:-4]
            #     k_input, v_input = self.reshape_kv([past_k, past_v])
            #     self.split_kv([present_k, present_v], present_kv_name)    
            #     # Create MHA node
            #     new_node = helper.make_node(
            #         "MultiHeadAttention",
            #         inputs=[q_input, k_input, v_input],
            #         outputs=[attention_last_node.output[0], present_kv_name],
            #         name=self.model.create_node_name("Attention"),
            #     )
            #     new_node.domain = "com.microsoft"
            #     new_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
            else:
                self.use_multi_head_attention = decoder_cross_attention #or decoder_cross_attention_with_past
                new_node = self.create_attention_node(
                    None, # mask_index,
                    matmul_q,
                    matmul_k,
                    matmul_v,
                    add_q,
                    add_k,
                    add_v,
                    num_heads,
                    hidden_size,
                    root_input,
                    attention_last_node.output[0],
                    add_qk_str=mask_index if decoder_attention else None,
                    past_k=past_k,
                    past_v=past_v,
                    present_k=present_k,
                    present_v=present_v,
                )
            #print("checkpoint 8")
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            self.nodes_to_remove.extend([attention_last_node, transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)
            
            # When using cross attention, keep most nodes in original graph
            if decoder_cross_attention:
                q_nodes = [q_nodes[0], q_nodes[3]]
                k_nodes = k_nodes[:2]
                v_nodes = v_nodes[:1]
            elif decoder_cross_attention_with_past:
                q_nodes = [q_nodes[0], q_nodes[3]]
            else:
                pass

            self.nodes_to_remove.extend(q_nodes)
            self.nodes_to_remove.extend(k_nodes)
            self.nodes_to_remove.extend(v_nodes)

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            self.prune_graph = True

    # def apply(self):
    #     FusionAttention.apply(self)

    #     # Remove Cast to int32 applied by attention fusion for 4D attention mask
    #     input_name_to_nodes = self.model.input_name_to_nodes()
    #     output_name_to_node = self.model.output_name_to_node()

    #     # Get nodes in Expand --> Cast --> Concat path
    #     expand_node = output_name_to_node[self.attention_mask_index]
    #     print(expand_node)
    #     cast_int32_node = input_name_to_nodes[self.attention_mask_index][0]
    #     print(cast_int32_node)
    #     concat_node = input_name_to_nodes[cast_int32_node.output[0]][0]
    #     print(concat_node)

    #     # Set each input of Concat to output of Expand
    #     for i in range(len(concat_node.input)):
    #         concat_node.input[i] = expand_node.output[0]

    #     # Rename Concat output to replace 'int32' label with 'float32' label
    #     label_int32 = concat_node.output[0]
    #     label_float32 = label_int32.replace("int", "float")
    #     concat_node.output[0] = label_float32

    #     # Update attention fusion inputs to take new mask name
    #     attention_nodes = input_name_to_nodes[label_int32]
    #     for attn_node in attention_nodes:
    #         print(attn_node.input)
    #         attn_node.input[5] = label_float32

    #     # Remove Cast node from graph
    #     self.model.remove_node(cast_int32_node)
    #     self.model.update_graph()


class FusionBartReshape(FusionReshape):
    def __init__(self, model: OnnxModel):
        super().__init__(model)

    def fuse(self, reshape_node, input_name_to_nodes, output_name_to_node):
        if reshape_node.input[1] not in output_name_to_node:
            return

        concat_node = output_name_to_node[reshape_node.input[1]]
        if concat_node.op_type != "Concat" or len(concat_node.input) != 4:
            return

        path0 = self.model.match_parent_path(
            concat_node,
            ["Unsqueeze", "Gather", "Shape"],
            [0, 0, 0],
            output_name_to_node,
        )
        if path0 is None:
            return

        (_, gather_0, shape_0) = path0

        shape = []
        gather_value = self.model.get_constant_value(gather_0.input[1])
        if gather_value == 0:
            shape.append(0)

        path1 = self.model.match_parent_path(
            concat_node,
            ["Unsqueeze", "Gather", "Shape"],
            [1, 0, 0],
            output_name_to_node,
        )
        if path1 is None:
            input_1_proto = self.model.get_initializer(concat_node.input[1])
            input_2_proto = self.model.get_initializer(concat_node.input[2])
            input_3_proto = self.model.get_initializer(concat_node.input[3])
            if input_1_proto is None or input_2_proto is None or input_3_proto is None:
                return

            input_1 = numpy_helper.to_array(input_1_proto)
            input_2 = numpy_helper.to_array(input_2_proto)
            input_3 = numpy_helper.to_array(input_3_proto)
            if len(input_1) != 1 or len(input_2) != 1 or len(input_3) != 1:
                return

            if not (input_1[0] == -1 and input_2[0] > 0 and input_3[0] > 0):
                return

            shape.extend(input_1)
            shape.extend(input_2)
            shape.extend(input_3)
            gemm_path_with_bias = self.model.match_parent_path(reshape_node, ["Add", "MatMul"], [0, 1], output_name_to_node)
            gemm_path_no_bias = self.model.match_parent_path(reshape_node, ["MatMul"], [0], output_name_to_node) # K path with bias=False in PyTorch before exporting to ONNX
            if gemm_path_with_bias is not None:
                gemm_path = gemm_path_with_bias
            elif gemm_path_no_bias is not None:
                gemm_path = gemm_path_no_bias
            else:
                return

            top_matmul = gemm_path[-1]
            root_input = top_matmul.input[0]
            # First invariant: output of SkipLayerNorm = first input of shape_0 (if attention has one root input)
            # Second invariant: first input of matmul_k = first input of matmul_v (if attention has two root inputs)
            # if shape_0.input[0] != root_input:
            #     return

            #print("If:", reshape_node.name)
            self.replace_reshape_node(shape, reshape_node, concat_node)
        else:
            (_, gather_1, shape_1) = path1

            gather_value = self.model.get_constant_value(gather_1.input[1])
            if gather_value == 1:
                shape.append(0)

            input_2_proto = self.model.get_initializer(concat_node.input[2])
            input_3_proto = self.model.get_initializer(concat_node.input[3])
            if input_2_proto is None or input_3_proto is None:
                return

            input_2 = numpy_helper.to_array(input_2_proto)
            input_3 = numpy_helper.to_array(input_3_proto)
            if len(input_2) != 1 or len(input_3) != 1:
                return

            if not (input_2[0] > 0 and input_3[0] > 0):
                return

            shape.extend(input_2)
            shape.extend(input_3)
            gemm_path = self.model.match_parent_path(
                reshape_node, ["Mul", "Add", "MatMul"], [0, 0, 1], output_name_to_node
            )
            if gemm_path is None:
                return

            top_matmul = gemm_path[-1]
            root_input = top_matmul.input[0]
            if shape_0.input[0] != root_input or shape_1.input[0] != root_input:
                return

            #print("Else:", reshape_node.name)
            self.replace_reshape_node(shape, reshape_node, concat_node)


class BartOnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionBartAttention(self, self.hidden_size, self.num_heads, self.attention_mask)
        self.bart_reshape_fusion_preprocess = FusionBartReshape(self)

    def optimize(self, options: Optional[FusionOptions] = None, add_dynamic_axes: bool = False):
        self.attention_fusion.fuse_mha_bias = options.fuse_mha_bias
        super().optimize(options, add_dynamic_axes)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def preprocess(self):
        self.adjust_reshape_and_expand()
        self.bart_reshape_fusion_preprocess.apply()
