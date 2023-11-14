# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import sys
import unittest
from typing import List

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper
from parity_utilities import find_transformers_source

if find_transformers_source():
    from fusion_options import FusionOptions
    from onnx_model import OnnxModel
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.optimizer import optimize_model


def float_tensor(name: str, shape: List[int], random=False):
    low = 0.0
    high = 1.0
    total_elements = 1
    for x in shape:
        total_elements *= x
    weights = [np.random.uniform(low, high) for _ in range(total_elements)] if random else [1.0] * total_elements
    return helper.make_tensor(name, TensorProto.FLOAT, shape, weights)


class TestRotaryAttentionFusion(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.sequence_length = 8
        self.num_heads = 4
        self.head_size = 6
        self.hidden_size = self.num_heads * self.head_size

        self.past_sequence_length = 2
        self.max_sequence_length = 12

    def verify_fusion(self, expected_model_path, original_model_path):
        expected_model = OnnxModel(onnx.load(expected_model_path))
        expected_model.topological_sort(is_deterministic=True)

        model_type = "gpt2"
        options = FusionOptions(model_type)
        optimized_model = optimize_model(
            original_model_path,
            model_type,
            self.num_heads,
            self.hidden_size,
            optimization_options=options,
            opt_level=0,
        )
        optimized_model.topological_sort(is_deterministic=True)

        self.assertTrue(str(expected_model.model.graph), str(optimized_model.model.graph))

    def create_initializers(self, fused_model: bool = False):
        initializers = [
            float_tensor("cos_cache", [self.max_sequence_length, self.head_size // 2]),
            float_tensor("sin_cache", [self.max_sequence_length, self.head_size // 2]),
            float_tensor("q_weight", [self.hidden_size, self.hidden_size]),
            float_tensor("k_weight", [self.hidden_size, self.hidden_size]),
            float_tensor("v_weight", [self.hidden_size, self.hidden_size]),
            float_tensor("o_weight", [self.hidden_size, self.hidden_size]),
            helper.make_tensor(
                "sqrt_head_size", TensorProto.FLOAT, [1], np.array([np.sqrt(self.head_size)], dtype=np.float32)
            ),
            helper.make_tensor("neg_int_max", TensorProto.FLOAT, [1], np.array([-sys.maxsize - 1], dtype=np.int64)),
            helper.make_tensor("num_heads", TensorProto.FLOAT, [1], np.array([self.num_heads], dtype=np.float32)),
            helper.make_tensor("head_size", TensorProto.FLOAT, [1], np.array([self.head_size], dtype=np.float32)),
            helper.make_tensor("hidden_size", TensorProto.FLOAT, [1], np.array([self.hidden_size], dtype=np.float32)),
            helper.make_tensor("zero", TensorProto.FLOAT, [1], np.array([0], dtype=np.int64)),
            helper.make_tensor("one", TensorProto.FLOAT, [1], np.array([1], dtype=np.int64)),
            helper.make_tensor("two", TensorProto.FLOAT, [1], np.array([2], dtype=np.int64)),
            helper.make_tensor("three", TensorProto.FLOAT, [1], np.array([3], dtype=np.int64)),
        ]
        return initializers

    def create_inputs_and_outputs(self, model_type: str):
        attn_mask_size = [self.batch_size, self.sequence_length]
        if model_type == "llama2_msft":
            attn_mask_size.append(self.sequence_length)

        inputs = [
            helper.make_tensor_value_info(
                "input_0", TensorProto.FLOAT, [self.batch_size, self.sequence_length, self.hidden_size]
            ),
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, [self.batch_size, self.sequence_length]),
            helper.make_tensor_value_info("attn_mask", TensorProto.INT64, attn_mask_size),
        ]
        if model_type in {"past", "merged", "llama2_msft", "70b_distributed_merged"}:
            inputs.extend(
                [
                    helper.make_tensor_value_info(
                        "past_key",
                        TensorProto.FLOAT,
                        [self.batch_size, self.num_heads, self.past_sequence_length, self.head_size],
                    ),
                    helper.make_tensor_value_info(
                        "past_value",
                        TensorProto.FLOAT,
                        [self.batch_size, self.num_heads, self.past_sequence_length, self.head_size],
                    ),
                ]
            )
        outputs = [
            helper.make_tensor_value_info(
                "output_0", TensorProto.FLOAT, [self.batch_size, self.sequence_length, self.hidden_size]
            ),
            helper.make_tensor_value_info(
                "present_key",
                TensorProto.FLOAT,
                [self.batch_size, self.num_heads, self.past_sequence_length + 1, self.head_size],
            ),
            helper.make_tensor_value_info(
                "present_value",
                TensorProto.FLOAT,
                [self.batch_size, self.num_heads, self.past_sequence_length + 1, self.head_size],
            ),
        ]
        return inputs, outputs

    def create_matmul_nodes(self, is_fused: bool, model_type: str):
        q_matmul_node = helper.make_node(
            "MatMul",
            inputs=["input_0", "q_weight"],
            outputs=["q_out" if is_fused or model_type == "llama2_msft" else "q_matmul_out"],
            name="Q_MatMul",
        )

        k_matmul_node = helper.make_node(
            "MatMul",
            inputs=["input_0", "k_weight"],
            outputs=["k_out" if is_fused or model_type == "llama2_msft" else "k_matmul_out"],
            name="K_MatMul",
        )

        v_matmul_node = helper.make_node(
            "MatMul",
            inputs=["input_0", "v_weight"],
            outputs=["v_out"],
            name="V_MatMul",
        )

        return [q_matmul_node, k_matmul_node, v_matmul_node]

    def create_rotary_embeddings(
        self,
        is_fused: bool,
        model_type: str,
        interleaved: bool,
        inputs: List[TensorProto],
        initializers: List[TensorProto],
    ):
        def get_first_rope_input(node_type: str):
            if is_fused or model_type == "llama2_msft":
                # q_out/k_out
                return f"{node_type}_out"
            if model_type in {"no_past", "past", "merged", "70b_distributed_merged"}:
                if node_type == "k":
                    return "k_before_rope"
                return "q_before_rope"
            return ""

        def get_first_rope_output(node_type: str):
            if is_fused or model_type in {"llama2_msft", "past", "merged", "70b_distributed_merged"}:
                if node_type == "q":
                    return "q_rope"
                return "k_rope"
            if model_type in {"no_past"}:
                if node_type == "k":
                    return "present_key"
                return "q_rope"
            return ""

        q_rope_node = helper.make_node(
            "RotaryEmbedding",
            inputs=[get_first_rope_input("q"), inputs[1].name, initializers[0].name, initializers[1].name],
            outputs=[get_first_rope_output("q")],
            name="Q_RotaryEmbedding",
            interleaved=int(interleaved),
        )

        k_rope_node = helper.make_node(
            "RotaryEmbedding",
            inputs=[get_first_rope_input("k"), inputs[1].name, initializers[0].name, initializers[1].name],
            outputs=[get_first_rope_output("k")],
            name="K_RotaryEmbedding",
            interleaved=int(interleaved),
        )

        return [q_rope_node, k_rope_node]

    def create_q_path(self, model_type: str):
        if model_type == "llama2_msft":
            transpose_q_node = helper.make_node(
                "Transpose",
                inputs=["q_rope"],
                outputs=["q_transposed"],
                name="Transpose_q",
                perm=[0, 2, 1, 3],
            )
            reshape_q_node = helper.make_node(
                "Reshape",
                inputs=["q_transposed", "concat_q_extra_out"],
                outputs=["q"],
                name="Reshape_q",
            )
            return [transpose_q_node, reshape_q_node]

        reshape_q_node = helper.make_node(
            "Reshape",
            inputs=["q_matmul_out", "concat_q_extra_out"],
            outputs=["q_reshaped"],
            name="Reshape_q",
        )
        transpose_q_node = helper.make_node(
            "Transpose",
            inputs=["q_reshaped"],
            outputs=["q_before_rope"],
            name="Transpose_q",
        )
        return [reshape_q_node, transpose_q_node]

    def create_k_path_llama2_msft(self):
        # Create k cache slicing path
        k_cache_unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs=["position_ids", "zero"],
            outputs=["k_pos_id"],
        )
        k_cache_slice_node = helper.make_node(
            "Slice",
            inputs=["past_key", "zero", "k_pos_id", "two", "one"],
            outputs=["k_cache_sliced"],
        )
        # Create k path
        transpose_k_1_node = helper.make_node(
            "Transpose",
            inputs=["k_rope"],
            outputs=["k_rope_transposed"],
            name="Transpose_k_1",
            perm=[0, 2, 1, 3],
        )
        concat_k_node = helper.make_node(
            "Concat",
            inputs=["k_cache_sliced", "k_rope_transposed"],
            outputs=["present_key"],
            name="Concat_k",
            axis=2,
        )
        transpose_k_2_node = helper.make_node(
            "Transpose",
            inputs=["present_key"],
            outputs=["present_key_transposed"],
            name="Transpose_k_2",
            perm=[0, 2, 3, 1],
        )
        reshape_k_node = helper.make_node(
            "Reshape",
            inputs=["present_key_transposed", "concat_k_extra_out"],
            outputs=["k"],
            name="Reshape_k",
        )
        return [
            k_cache_unsqueeze_node,
            k_cache_slice_node,
            transpose_k_1_node,
            concat_k_node,
            transpose_k_2_node,
            reshape_k_node,
        ]

    def create_k_path_hf(self, model_type: str):
        reshape_k_node = helper.make_node(
            "Reshape",
            inputs=["k_matmul_out", "concat_k_extra_out"],
            outputs=["k_reshaped"],
            name="Reshape_k",
        )
        transpose_k_1_node = helper.make_node(
            "Transpose",
            inputs=["k_reshaped"],
            outputs=["k_before_rope"],
            name="Transpose_k_1",
            perm=[0, 2, 1, 3],
        )
        k_nodes = [reshape_k_node, transpose_k_1_node]

        if model_type == "70b_distributed_merged":
            concat_k_node = helper.make_node(
                "Concat",
                inputs=["past_key", "k_rope"],
                outputs=["present_key"],
                axis=2,
            )
            shape_k1 = helper.make_node("Shape", inputs=["present_value"], outputs=["shape_k1_out"], name="Shape_k1")
            shape_k2 = helper.make_node("Shape", inputs=["present_value"], outputs=["shape_k2_out"], name="Shape_k2")
            shape_k3 = helper.make_node("Shape", inputs=["present_value"], outputs=["shape_k3_out"], name="Shape_k3")
            shape_k4 = helper.make_node("Shape", inputs=["present_value"], outputs=["shape_k4_out"], name="Shape_k4")

            gather_k_1 = helper.make_node(
                "Gather",
                inputs=["shape_k1_out", "one"],
                outputs=["gather_k1_out"],
                name="Gather_k_1",
                axis=0,
            )
            gather_k_2 = helper.make_node(
                "Gather",
                inputs=["shape_k2_out", "one"],
                outputs=["gather_k2_out"],
                name="Gather_k_2",
                axis=0,
            )
            gather_k_3 = helper.make_node(
                "Gather",
                inputs=["shape_k3_out", "one"],
                outputs=["gather_k3_out"],
                name="Gather_k_3",
                axis=0,
            )
            gather_k_4 = helper.make_node(
                "Gather",
                inputs=["shape_k4_out", "one"],
                outputs=["gather_k4_out"],
                name="Gather_k_4",
                axis=0,
            )

            unsqueeze_k_1 = helper.make_node(
                "Unsqueeze",
                inputs=["present_value", "zero"],
                outputs=["unsqueeze_k1_out"],
                name="Unsqueeze_k1",
            )
            unsqueeze_k_2 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_k1_out", "zero"],
                outputs=["unsqueeze_k2_out"],
                name="Unsqueeze_k2",
            )
            unsqueeze_k_3 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_k2_out", "zero"],
                outputs=["unsqueeze_k3_out"],
                name="Unsqueeze_k3",
            )
            unsqueeze_k_4 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_k3_out", "zero"],
                outputs=["unsqueeze_k4_out"],
                name="Unsqueeze_k4",
            )
            unsqueeze_k_5 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_k4_out", "zero"],
                outputs=["unsqueeze_k5_out"],
                name="Unsqueeze_k5",
            )

            concat_k_2 = helper.make_node(
                "Concat",
                inputs=["unsqueeze_k2_out", "unsqueeze_k3_out", "One", "unsqueeze_k4_out", "unsqueeze_k5_out"],
                outputs=["concat_k2_ouot"],
                name="Concat_k2",
                axis=0,
            )
            reshape_k_2 = helper.make_node(
                "Reshape",
                inputs=["concat_k2_ouot", "One"],
                outputs=["reshape_k2_out"],
                name="Reshape_k_2",
            )
            shape_k5 = helper.make_node("Shape", inputs=["reshape_k2_out"], outputs=["shape_k5_out"], name="Shape_k5")
            constant_of_shape_k_1 = helper.make_node(
                "ConstantOfShape",
                inputs=["shape_k5_out"],
                outputs=["constant_of_shape_k1_out"],
                name="ConstantOfShape_k1",
            )
            mul_k_1 = helper.make_node(
                "Mul",
                inputs=["constant_of_shape_k1_out", "One"],
                outputs=["mul_k1_out"],
                name="mul_k1",
            )
            equal_k_1 = helper.make_node(
                "Equal",
                inputs=["reshape_k2_out", "mul_k1_out"],
                outputs=["equal_k_1_out"],
                name="equal_k1",
            )
            where_k_1 = helper.make_node(
                "Where",
                inputs=["equal_k_1_out", "constant_of_shape_k1_out", "reshape_k2_out"],
                outputs=["where_k_1_out"],
                name="where_k1",
            )
            unsqueeze_k_6 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_k1_out", "zero"],
                outputs=["unsqueeze_k6_out"],
                name="Unsqueeze_k6",
            )
            mul_k_2 = helper.make_node(
                "Mul",
                inputs=["gather_k2_out", "One"],
                outputs=["mul_k2_out"],
                name="mul_k2",
            )
            unsqueeze_k_7 = helper.make_node(
                "Unsqueeze",
                inputs=["mul_k2_out", "zero"],
                outputs=["unsqueeze_k7_out"],
                name="Unsqueeze_k7",
            )
            unsqueeze_k_8 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_k3_out", "zero"],
                outputs=["unsqueeze_k8_out"],
                name="Unsqueeze_k8",
            )
            unsqueeze_k_9 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_k4_out", "zero"],
                outputs=["unsqueeze_k9_out"],
                name="Unsqueeze_k9",
            )
            concat_k_3 = helper.make_node(
                "Concat",
                inputs=["unsqueeze_k6_out", "unsqueeze_k7_out", "unsqueeze_k8_out", "unsqueeze_k9_out"],
                outputs=["concat_k3_out"],
                name="Concat_k3",
                axis=0,
            )
            expand_k_1 = helper.make_node(
                "Expand",
                inputs=["unsqueeze_k1_out", "where_k_1_out"],
                outputs=["expand_k1_out"],
                name="expand_k1",
            )
            reshape_k_3 = helper.make_node(
                "Reshape",
                inputs=["expand_k1_out", "concat_k3_out"],
                outputs=["reshape_k3_out"],
                name="Reshape_k_3",
            )
            transpose_k_2_node = helper.make_node(
                "Transpose",
                inputs=["reshape_k3_out"],
                outputs=["k"],
                name="Transpose_k_2",
                perm=[0, 1, 3, 2],
            )

            k_nodes_for_70b_model = [
                concat_k_node,
                shape_k1,
                shape_k2,
                shape_k3,
                shape_k4,
                gather_k_1,
                gather_k_2,
                gather_k_3,
                gather_k_4,
                unsqueeze_k_1,
                unsqueeze_k_2,
                unsqueeze_k_3,
                unsqueeze_k_4,
                unsqueeze_k_5,
                concat_k_2,
                reshape_k_2,
                shape_k5,
                constant_of_shape_k_1,
                mul_k_1,
                equal_k_1,
                where_k_1,
                unsqueeze_k_6,
                mul_k_2,
                unsqueeze_k_7,
                unsqueeze_k_8,
                unsqueeze_k_9,
                concat_k_3,
                expand_k_1,
                reshape_k_3,
                transpose_k_2_node,
            ]
            k_nodes.extend(k_nodes_for_70b_model)
            return k_nodes
        else:
            if model_type in {"past", "merged"}:
                concat_k_node = helper.make_node(
                    "Concat",
                    inputs=["past_key", "k_rope"],
                    outputs=["present_key"],
                    axis=2,
                )
                k_nodes.append(concat_k_node)

            transpose_k_2_node = helper.make_node(
                "Transpose",
                inputs=["present_key"],
                outputs=["k"],
                name="Transpose_k_2",
                perm=[0, 1, 3, 2],
            )
            return k_nodes + [transpose_k_2_node]  # noqa: RUF005

    def create_k_path(self, model_type: str):
        if model_type == "llama2_msft":
            return self.create_k_path_llama2_msft()
        return self.create_k_path_hf(model_type)

    def create_attn_mask_path_llama2_msft(self):
        x_shape_node = helper.make_node(
            "Shape",
            inputs=["input_0"],
            outputs=["input_0_shape"],
            name="Shape_input",
        )
        x_get_seq_len_node = helper.make_node(
            "Gather",
            inputs=["input_0_shape", "one"],
            outputs=["input_0_seq_len"],
            name="Gather_input",
            axis=0,
        )
        x_new_seq_len_node = helper.make_node(
            "Add",
            inputs=["position_ids", "input_0_seq_len"],
            outputs=["new_seq_len"],
            name="Add_mask",
        )
        unsqueeze_0_node = helper.make_node(
            "Unsqueeze",
            inputs=["position_ids", "zero"],
            outputs=["unsqueeze_mask_0_out"],
            name="Unsqueeze_mask_0",
        )
        unsqueeze_1_node = helper.make_node(
            "Unsqueeze",
            inputs=["new_seq_len", "zero"],
            outputs=["unsqueeze_mask_1_out"],
            name="Unsqueeze_mask_1",
        )
        unsqueeze_2_node = helper.make_node(
            "Unsqueeze",
            inputs=["new_seq_len", "zero"],
            outputs=["unsqueeze_mask_2_out"],
            name="Unsqueeze_mask_2",
        )
        slice_mask_1_node = helper.make_node(
            "Slice",
            inputs=["attn_mask", "unsqueeze_mask_0_out", "unsqueeze_mask_1_out", "one", "one"],
            outputs=["slice_mask_1_out"],
            name="Slice_mask_1",
        )
        slice_mask_2_node = helper.make_node(
            "Slice",
            inputs=["slice_mask_1_out", "zero", "unsqueeze_mask_2_out", "two", "one"],
            outputs=["slice_mask_2_out"],
            name="Slice_mask_2",
        )
        concat_mask_node = helper.make_node(
            "Concat",
            inputs=["slice_mask_2_out" for _ in range(self.num_heads)],
            outputs=["attn_mask_out"],
            name="Concat_mask",
            axis=0,
        )
        return [
            x_shape_node,
            x_get_seq_len_node,
            x_new_seq_len_node,
            unsqueeze_0_node,
            unsqueeze_1_node,
            unsqueeze_2_node,
            slice_mask_1_node,
            slice_mask_2_node,
            concat_mask_node,
        ]

    def create_attn_mask_path_hf(self, model_type: str):
        unsqueeze_1_node = helper.make_node(
            "Unsqueeze",
            inputs=["attn_mask", "one"],
            outputs=["unsqueeze_1_mask_out"],
            name="Unsqueeze_1_mask",
        )
        unsqueeze_2_node = helper.make_node(
            "Unsqueeze",
            inputs=["unsqueeze_1_mask_out", "two"],
            outputs=["unsqueeze_2_mask_out"],
            name="Unsqueeze_2_mask",
        )
        expand_node = helper.make_node(
            "Expand",
            inputs=["unsqueeze_2_mask_out", "zero"],
            outputs=["expand_out"],
            name="Expand_mask",
        )
        cast_node = helper.make_node(
            "Cast",
            inputs=["expand_out"],
            outputs=["cast_out"],
            name="Cast_mask",
            to=TensorProto.FLOAT,
        )
        sub_node = helper.make_node(
            "Sub",
            inputs=["one", "cast_out"],
            outputs=["sub_out"],
            name="Sub_mask",
        )
        where_node = helper.make_node(
            "Where",
            inputs=["zero", "neg_int_max", "sub_out"],
            outputs=["where_out" if model_type != "past" else "attn_mask_out"],
            name="Where_mask",
        )
        attn_mask_nodes = [unsqueeze_1_node, unsqueeze_2_node, expand_node, cast_node, sub_node, where_node]

        if model_type == "past":
            return attn_mask_nodes

        add_node = helper.make_node(
            "Add",
            inputs=["where_out", "zero"],
            outputs=["attn_mask_out"],
            name="Add_mask",
        )
        return attn_mask_nodes + [add_node]  # noqa: RUF005

    def create_attn_mask_path(self, is_fused: bool, model_type: str):
        if model_type == "llama2_msft":
            attn_mask_nodes = self.create_attn_mask_path_llama2_msft()
            if is_fused:
                attn_mask_nodes.pop()
                attn_mask_nodes[-1].output[0] = "attn_mask_out"
            return attn_mask_nodes

        attn_mask_nodes = self.create_attn_mask_path_hf(model_type)
        if is_fused:
            new_output_name = "attn_mask_out_mask"
            attn_mask_nodes[-1].output[0] = new_output_name
            concat_mask_node = helper.make_node(
                "Concat",
                inputs=[new_output_name for _ in range(self.num_heads)],
                outputs=["attn_mask_out"],
                name="Concat_mask",
                axis=0,
            )
            attn_mask_nodes.append(concat_mask_node)
        return attn_mask_nodes

    def create_qk_path(self, model_type: str):
        matmul_qk_node = helper.make_node(
            "MatMul",
            inputs=["q" if model_type == "llama2_msft" else "q_rope", "k"],
            outputs=["qk"],
            name="MatMul_q_k",
        )
        div_node = helper.make_node(
            "Div",
            inputs=["qk", "sqrt_head_size"],
            outputs=["qk_div"],
            name="Div_0",
        )
        add_node = helper.make_node(
            "Add",
            inputs=["qk_div", "attn_mask_out"],
            outputs=["qk_plus_mask"],
            name="Add_0",
        )
        softmax_node = helper.make_node(
            "Softmax",
            inputs=["qk_plus_mask"],
            outputs=["softmax_out"],
            name="Softmax_0",
        )
        return [matmul_qk_node, div_node, add_node, softmax_node]

    def create_v_path(self, model_type: str):
        reshape_v_1_node = helper.make_node(
            "Reshape",
            inputs=["v_out", "concat_v_1_extra_out"],
            outputs=["reshape_v_1_out"],
            name="Reshape_v_1",
        )
        transpose_v_1_node = helper.make_node(
            "Transpose",
            inputs=["reshape_v_1_out"],
            outputs=["transpose_v_1_out" if model_type != "no_past" else "present_value"],
            name="Transpose_v_1",
        )
        v_nodes = [reshape_v_1_node, transpose_v_1_node]

        if model_type == "no_past":
            return v_nodes

        if model_type in {"past", "merged", "70b_distributed_merged"}:
            concat_v_node = helper.make_node(
                "Concat",
                inputs=["past_value", "transpose_v_1_out"],
                outputs=["present_value"],
                name="Concat_v",
                axis=2,
            )

            if model_type != "70b_distributed_merged":
                return v_nodes + [concat_v_node]  # noqa: RUF005

            shape_v1 = helper.make_node("Shape", inputs=["present_value"], outputs=["shape_1_out"], name="Shape_v1")
            shape_v2 = helper.make_node("Shape", inputs=["present_value"], outputs=["shape_2_out"], name="Shape_v2")
            shape_v3 = helper.make_node("Shape", inputs=["present_value"], outputs=["shape_3_out"], name="Shape_v3")
            shape_v4 = helper.make_node("Shape", inputs=["present_value"], outputs=["shape_4_out"], name="Shape_v4")
            gather_v_1 = helper.make_node(
                "Gather",
                inputs=["shape_1_out", "one"],
                outputs=["gather_1_out"],
                name="Gather_v1",
                axis=0,
            )
            gather_v_2 = helper.make_node(
                "Gather",
                inputs=["shape_2_out", "one"],
                outputs=["gather_2_out"],
                name="Gather_v2",
                axis=0,
            )
            gather_v_3 = helper.make_node(
                "Gather",
                inputs=["shape_3_out", "one"],
                outputs=["gather_3_out"],
                name="Gather_v3",
                axis=0,
            )
            gather_v_4 = helper.make_node(
                "Gather",
                inputs=["shape_4_out", "one"],
                outputs=["gather_4_out"],
                name="Gather_v4",
                axis=0,
            )
            unsqueeze_v_1 = helper.make_node(
                "Unsqueeze",
                inputs=["present_value", "zero"],
                outputs=["unsqueeze_v1_out"],
                name="Unsqueeze_v1",
            )
            unsqueeze_v_2 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_1_out", "zero"],
                outputs=["unsqueeze_v2_out"],
                name="Unsqueeze_v2",
            )
            unsqueeze_v_3 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_2_out", "zero"],
                outputs=["unsqueeze_v3_out"],
                name="Unsqueeze_v3",
            )
            unsqueeze_v_4 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_3_out", "zero"],
                outputs=["unsqueeze_v4_out"],
                name="Unsqueeze_v4",
            )
            unsqueeze_v_5 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_4_out", "zero"],
                outputs=["unsqueeze_v5_out"],
                name="Unsqueeze_v5",
            )
            concat_v_2 = helper.make_node(
                "Concat",
                inputs=["unsqueeze_v2_out", "unsqueeze_v3_out", "One", "unsqueeze_v4_out", "unsqueeze_v5_out"],
                outputs=["concat_v2_ouot"],
                name="Concat_v2",
                axis=0,
            )
            reshape_v_2 = helper.make_node(
                "Reshape",
                inputs=["concat_v2_ouot", "One"],
                outputs=["reshape_v2_out"],
                name="Reshape_v2",
            )
            shape_v5 = helper.make_node("Shape", inputs=["reshape_v2_out"], outputs=["shape_5_out"], name="Shape_v5")
            constant_of_shape_v_1 = helper.make_node(
                "ConstantOfShape",
                inputs=["shape_5_out"],
                outputs=["constant_of_shape_v1_out"],
                name="ConstantOfShape_v1",
            )
            mul_v_1 = helper.make_node(
                "Mul",
                inputs=["constant_of_shape_v1_out", "One"],
                outputs=["mul_v1_out"],
                name="mul_v1",
            )
            equal_v_1 = helper.make_node(
                "Equal",
                inputs=["reshape_v2_out", "mul_v1_out"],
                outputs=["equal_v_1_out"],
                name="equal_v1",
            )
            where_v_1 = helper.make_node(
                "Where",
                inputs=["equal_v_1_out", "constant_of_shape_v1_out", "reshape_v2_out"],
                outputs=["where_v_1_out"],
                name="where_v1",
            )
            unsqueeze_v_6 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_1_out", "zero"],
                outputs=["unsqueeze_v6_out"],
                name="Unsqueeze_v6",
            )
            mul_v_2 = helper.make_node(
                "Mul",
                inputs=["gather_2_out", "One"],
                outputs=["mul_v2_out"],
                name="mul_v2",
            )
            unsqueeze_v_7 = helper.make_node(
                "Unsqueeze",
                inputs=["mul_v2_out", "zero"],
                outputs=["unsqueeze_v7_out"],
                name="Unsqueeze_v7",
            )
            unsqueeze_v_8 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_3_out", "zero"],
                outputs=["unsqueeze_v8_out"],
                name="Unsqueeze_v8",
            )
            unsqueeze_v_9 = helper.make_node(
                "Unsqueeze",
                inputs=["gather_4_out", "zero"],
                outputs=["unsqueeze_v9_out"],
                name="Unsqueeze_v9",
            )
            concat_v_3 = helper.make_node(
                "Concat",
                inputs=["unsqueeze_v6_out", "unsqueeze_v7_out", "unsqueeze_v8_out", "unsqueeze_v9_out"],
                outputs=["concat_v3_out"],
                name="Concat_v3",
                axis=0,
            )
            expand_v_1 = helper.make_node(
                "Expand",
                inputs=["unsqueeze_v1_out", "where_v_1_out"],
                outputs=["expand_v1_out"],
                name="expand_v1",
            )
            reshape_v_3 = helper.make_node(
                "Reshape",
                inputs=["expand_v1_out", "concat_v3_out"],
                outputs=["reshape_v3_out"],
                name="Reshape_v3",
            )

            v_nodes_for_70b_model = [
                concat_v_node,
                shape_v1,
                shape_v2,
                shape_v3,
                shape_v4,
                gather_v_1,
                gather_v_2,
                gather_v_3,
                gather_v_4,
                unsqueeze_v_1,
                unsqueeze_v_2,
                unsqueeze_v_3,
                unsqueeze_v_4,
                unsqueeze_v_5,
                concat_v_2,
                reshape_v_2,
                shape_v5,
                constant_of_shape_v_1,
                mul_v_1,
                equal_v_1,
                where_v_1,
                unsqueeze_v_6,
                mul_v_2,
                unsqueeze_v_7,
                unsqueeze_v_8,
                unsqueeze_v_9,
                concat_v_3,
                expand_v_1,
                reshape_v_3,
            ]
            v_nodes.extend(v_nodes_for_70b_model)

            return v_nodes

        # Create extra nodes for `position_ids`
        unsqueeze_v_node = helper.make_node(
            "Unsqueeze",
            inputs=["position_ids", "zero"],
            outputs=["unsqueeze_v_out"],
            name="Unsqueeze_v",
        )
        slice_v_node = helper.make_node(
            "Slice",
            inputs=["past_value", "zero", "unsqueeze_v_out", "two", "one"],
            outputs=["v_cache_sliced_out"],
            name="Slice_v",
        )
        concat_v_node = helper.make_node(
            "Concat",
            inputs=["v_cache_sliced_out", "transpose_v_1_out"],
            outputs=["present_value"],
            name="Concat_v",
            axis=2,
        )
        v_nodes.extend([unsqueeze_v_node, slice_v_node, concat_v_node])

        # Create remaining nodes for v path
        transpose_v_2_node = helper.make_node(
            "Transpose",
            inputs=["present_value"],
            outputs=["transpose_v_2_out"],
            name="Transpose_v_2",
        )
        reshape_v_2_node = helper.make_node(
            "Reshape",
            inputs=["transpose_v_2_out", "concat_v_2_extra_out"],
            outputs=["v"],
            name="Reshape_v_2",
        )
        return v_nodes + [transpose_v_2_node, reshape_v_2_node]  # noqa: RUF005

    def create_qkv_path(self, model_type: str):
        matmul_qkv_node = helper.make_node(
            "MatMul",
            inputs=["softmax_out", "v" if model_type == "llama2_msft" else "present_value"],
            outputs=["softmax_v_out"],
            name="MatMul_softmax_v",
        )
        qkv_nodes = [matmul_qkv_node]

        if model_type == "llama2_msft":
            reshape_qkv_1_node = helper.make_node(
                "Reshape",
                inputs=["softmax_v_out", "concat_qkv_1_extra_out"],
                outputs=["reshape_qkv_1_out"],
                name="Reshape_qkv_1",
            )
            qkv_nodes.append(reshape_qkv_1_node)

        transpose_qkv_node = helper.make_node(
            "Transpose",
            inputs=["reshape_qkv_1_out" if model_type == "llama2_msft" else "softmax_v_out"],
            outputs=["transpose_qkv_out"],
            name="Transpose_qkv",
        )
        reshape_qkv_2_node = helper.make_node(
            "Reshape",
            inputs=["transpose_qkv_out", "concat_qkv_2_extra_out"],
            outputs=["attn_output"],
            name="Reshape_qkv_2",
        )

        return qkv_nodes + [transpose_qkv_node, reshape_qkv_2_node]  # noqa: RUF005

    def create_concat_unsqueeze_paths(self, model_type: str, reshape_nodes: List[NodeProto]):
        # Create initial shape paths
        shape_0_node = helper.make_node(
            "Shape",
            inputs=["input_0"],
            outputs=["input_0_shape_0"],
            name="Shape_0",
        )
        gather_0_node = helper.make_node(
            "Gather",
            inputs=["input_0_shape_0", "zero"],
            outputs=["input_0_shape_0_indexed"],
            name="Gather_0",
            axis=0,
        )
        shape_1_node = helper.make_node(
            "Shape",
            inputs=["input_0"],
            outputs=["input_0_shape_1"],
            name="Shape_1",
        )
        gather_1_node = helper.make_node(
            "Gather",
            inputs=["input_0_shape_1", "one"],
            outputs=["input_0_shape_1_indexed"],
            name="Gather_1",
            axis=0,
        )
        extra_nodes = [shape_0_node, gather_0_node, shape_1_node, gather_1_node]

        if model_type == "llama2_msft":
            mul_node = helper.make_node(
                "Mul",
                inputs=[gather_0_node.output[0], "num_heads"],
                outputs=["mul_extra_out"],
                name="Mul_extra_0",
            )
            add_node = helper.make_node(
                "Add",
                inputs=[gather_1_node.output[0], "position_ids"],
                outputs=["add_extra_out"],
                name="Add_extra_0",
            )
            extra_nodes.extend([mul_node, add_node])

        for i, reshape_node in enumerate(reshape_nodes):
            use_mul_and_add_nodes_0 = model_type == "llama2_msft" and reshape_node.output[0] in {"q", "k", "v"}
            use_mul_and_add_nodes_1 = model_type == "llama2_msft" and reshape_node.output[0] in {"k", "v"}

            unsqueeze_0_node = helper.make_node(
                "Unsqueeze",
                inputs=[gather_0_node.output[0] if not use_mul_and_add_nodes_0 else "mul_extra_out", "zero"],
                outputs=[f"unsqueeze_extra_{2*i}"],
                name=f"Unsqueeze_extra_{2*i}",
            )
            unsqueeze_1_node = helper.make_node(
                "Unsqueeze",
                inputs=[gather_1_node.output[0] if not use_mul_and_add_nodes_1 else "add_extra_out", "zero"],
                outputs=[f"unsqueeze_extra_{2*i + 1}"],
                name=f"Unsqueeze_extra_{2*i + 1}",
            )

            reshape_name = reshape_node.name
            if reshape_name == "Reshape_qkv_2":
                concat_node_inputs = [unsqueeze_0_node.output[0], unsqueeze_1_node.output[0], "hidden_size"]
            elif reshape_name == "Reshape_qkv_1":
                concat_node_inputs = [unsqueeze_0_node.output[0], "num_heads", unsqueeze_1_node.output[0], "head_size"]
            elif reshape_name == "Reshape_v_2":
                concat_node_inputs = [unsqueeze_0_node.output[0], unsqueeze_1_node.output[0], "head_size"]
            elif reshape_name == "Reshape_v_1":
                concat_node_inputs = [unsqueeze_0_node.output[0], unsqueeze_1_node.output[0], "num_heads", "head_size"]
            elif reshape_name == "Reshape_k":
                concat_node_inputs = [unsqueeze_0_node.output[0], "head_size", unsqueeze_1_node.output[0]]
            elif reshape_name == "Reshape_q":
                concat_node_inputs = [unsqueeze_0_node.output[0], unsqueeze_1_node.output[0], "head_size"]

            concat_node = helper.make_node(
                "Concat",
                inputs=concat_node_inputs,
                outputs=[reshape_nodes[i].input[1]],
                name=f"Concat_extra_{i}",
                axis=0,
            )
            extra_nodes.extend([unsqueeze_0_node, unsqueeze_1_node, concat_node])

        return extra_nodes

    def create_end_nodes(self, model_type):
        if model_type == "70b_distributed_merged":
            matmul_o_node = helper.make_node(
                "MatMul",
                inputs=["attn_output", "o_weight"],
                outputs=["output_proj"],
                name="MatMul_o_proj",
            )
            all_reduce = helper.make_node(
                "AllReduce",
                inputs=["output_proj"],
                outputs=["allreduce_proj"],
                name="allreduce_proj",
            )
            end_node = helper.make_node(
                "Add",
                inputs=["zero", "allreduce_proj"],
                outputs=["output_0"],
                name="Add_normalize_node",
            )
            return [matmul_o_node, all_reduce, end_node]

        matmul_o_node = helper.make_node(
            "MatMul",
            inputs=["attn_output", "o_weight"],
            outputs=["output_proj"],
            name="MatMul_o_proj",
        )
        end_node = helper.make_node(
            "Add",
            inputs=["zero", "output_proj"],
            outputs=["output_0"],
            name="Add_normalize_node",
        )
        return [matmul_o_node, end_node]

    def create_fused_model(self, model_type: str, interleaved: bool, initializers: List[TensorProto]):
        inputs, outputs = self.create_inputs_and_outputs(model_type)
        matmul_nodes = self.create_matmul_nodes(True, model_type=model_type)
        rope_nodes = self.create_rotary_embeddings(True, model_type, interleaved, inputs, initializers)
        attn_mask_nodes = self.create_attn_mask_path(True, model_type)

        mha_inputs = [
            rope_nodes[0].output[0],  # q
            rope_nodes[1].output[0],  # k
            matmul_nodes[-1].output[0],  # v
            "",  # bias
            "attn_mask_out" if model_type == "llama2_msft" else "",  # attn_mask
            "attn_mask_out" if model_type != "llama2_msft" else "",  # add_qk
            "past_key" if model_type != "no_past" else "",  # past_key
            "past_value" if model_type != "no_past" else "",  # past_value
        ]
        mha_node = helper.make_node(
            "MultiHeadAttention",
            inputs=mha_inputs,
            outputs=["attn_output", "present_key", "present_value"],
            name="MultiHeadAttention_0",
            num_heads=self.num_heads,
        )

        end_nodes = self.create_end_nodes(model_type)

        graph = helper.make_graph(
            nodes=matmul_nodes + rope_nodes + attn_mask_nodes + [mha_node] + end_nodes,
            name="RotaryAttention_Graph",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )
        opset_import = helper.make_opsetid(domain="com.microsoft", version=1)
        model = helper.make_model(graph, opset_imports=[opset_import])
        return model

    def create_test_model(self, model_type: str, interleaved: bool, initializers: List[TensorProto]):
        inputs, outputs = self.create_inputs_and_outputs(model_type)
        matmul_nodes = self.create_matmul_nodes(False, model_type)
        rope_nodes = self.create_rotary_embeddings(False, model_type, interleaved, inputs, initializers)

        # Create main paths
        q_nodes = self.create_q_path(model_type)
        k_nodes = self.create_k_path(model_type)
        attn_mask_nodes = self.create_attn_mask_path(False, model_type)
        qk_nodes = self.create_qk_path(model_type)
        v_nodes = self.create_v_path(model_type)
        qkv_nodes = self.create_qkv_path(model_type)

        reshape_nodes = list(filter(lambda node: node.op_type == "Reshape", q_nodes + k_nodes + v_nodes + qkv_nodes))
        extra_nodes = self.create_concat_unsqueeze_paths(model_type, reshape_nodes)

        end_nodes = self.create_end_nodes(model_type)

        first_set_of_nodes = matmul_nodes + rope_nodes + q_nodes + k_nodes + attn_mask_nodes
        second_set_of_nodes = qk_nodes + v_nodes + qkv_nodes + extra_nodes + end_nodes
        graph = helper.make_graph(
            nodes=first_set_of_nodes + second_set_of_nodes,
            name="RotaryAttention_Graph",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )
        opset_import = helper.make_opsetid(domain="ai.onnx", version=17)
        model = helper.make_model(graph, opset_imports=[opset_import])
        return model

    def check_models(self, model_type: str, interleaved: bool):
        initializers = self.create_initializers()

        expected_model_filename = "expected_model.onnx"
        expected_model = self.create_fused_model(model_type, interleaved, initializers)
        onnx.save(expected_model, expected_model_filename)

        original_model_filename = "original_model.onnx"
        original_model = self.create_test_model(model_type, interleaved, initializers)
        onnx.save(original_model, original_model_filename)

        self.verify_fusion(expected_model_filename, original_model_filename)
        os.remove(expected_model_filename)
        os.remove(original_model_filename)

    def test_llama2_msft_model(self):
        model_type = "llama2_msft"
        interleaved = True
        self.check_models(model_type, interleaved)

    def test_hf_decoder_model(self):
        model_type = "no_past"
        interleaved = False
        self.check_models(model_type, interleaved)

    def test_hf_decoder_with_past_model(self):
        model_type = "past"
        interleaved = False
        self.check_models(model_type, interleaved)

    def test_hf_decoder_merged_model(self):
        model_type = "merged"
        interleaved = False
        self.check_models(model_type, interleaved)

    def test_hf_70b_distributed_decoder_merged_model(self):
        model_type = "70b_distributed_merged"
        interleaved = False
        self.check_models(model_type, interleaved)


if __name__ == "__main__":
    unittest.main()
