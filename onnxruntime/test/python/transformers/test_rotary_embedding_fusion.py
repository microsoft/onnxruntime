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
from onnx import TensorProto, helper
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


class TestRotaryEmbeddingFusion(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.sequence_length = 8
        self.num_heads = 4
        self.head_size = 6
        self.hidden_size = num_heads * head_size

        self.past_sequence_length = 2
        self.max_sequence_length = 12

    def verify_fusion(self, expected_model_path, original_model_path):
        expected_model = OnnxModel(onnx.load(expected_model_path))
        expected_model.topological_sort(is_deterministic=True)

        options = FusionOptions("gpt2")
        optimized_model = optimize_model(original_model_path, optimization_options=options, opt_level=0)
        optimized_model.topological_sort(is_deterministic=True)

        self.assertTrue(str(expected_model.model.graph), str(optimized_model.model.graph))

    def create_initializers(self):
        initializers = [
            float_tensor("cos_cache", [self.max_sequence_length, self.head_size]),
            float_tensor("sin_cache", [self.max_sequence_length, self.head_size]),
            helper.make_tensor(
                "pos_ids_new_shape",
                TensorProto.FLOAT,
                [2],
                np.array([self.batch_size, self.sequence_length], dtype=np.int64),
            ),
            helper.make_tensor("zero", TensorProto.FLOAT, [1], np.array([0], dtype=np.int64)),
            helper.make_tensor("one", TensorProto.FLOAT, [1], np.array([1], dtype=np.int64)),
            helper.make_tensor("two", TensorProto.FLOAT, [1], np.array([2], dtype=np.int64)),
            helper.make_tensor("three", TensorProto.FLOAT, [1], np.array([3], dtype=np.int64)),
            helper.make_tensor("int_max", TensorProto.FLOAT, [1], np.array([sys.maxsize], dtype=np.int64)),
        ]
        return initializers

    def create_inputs_and_outputs(self, model_type: str = ""):
        inputs = [
            helper.make_tensor_value_info(
                "input_0",
                TensorProto.FLOAT,
                [self.batch_size, self.sequence_length, self.num_heads, self.head_size],
            ),
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, [self.batch_size, self.sequence_length]),
        ]
        if model_type in {"past", "merged"}:
            # Input will be removed in fused model since it's not used in RotaryEmbedding.
            # We create this input so that we can check the `past_seq_len` path during
            # RotaryEmbedding fusion.
            inputs.append(
                helper.make_tensor_value_info(
                    "past_key",
                    TensorProto.FLOAT,
                    [self.batch_size, self.num_heads, self.past_sequence_length, self.head_size],
                )
            )
        # Dummy input to test nodes for `curr_seq_len` path
        if model_type != "":
            inputs.append(
                helper.make_tensor_value_info(
                    "curr_key",
                    TensorProto.FLOAT,
                    [self.batch_size, self.sequence_length, self.num_heads, self.head_size],
                )
            )
        outputs = [
            helper.make_tensor_value_info(
                "output_0",
                TensorProto.FLOAT,
                [self.batch_size, self.num_heads, self.sequence_length, self.head_size],
            )
        ]
        if model_type in {"merged"}:
            # Dummy output to test that nodes for `past_seq_len` path are not removed for merged model
            outputs.append(helper.make_tensor_value_info("past_seq_len_plus_zero", TensorProto.FLOAT, [1]))
        return inputs, outputs

    def create_fused_model(self, interleaved: bool, initializers: List[TensorProto]):
        inputs, outputs = self.create_inputs_and_outputs()

        rope_node = helper.make_node(
            "RotaryEmbedding",
            inputs=[inputs[0].name, inputs[1].name, initializers[0].name, initializers[1].name],
            outputs=[outputs[0].name],
            name="RotaryEmbedding_0",
            interleaved=int(interleaved),
        )

        graph = helper.make_graph(
            nodes=[rope_node],
            name="RotaryEmbedding_Graph",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )
        opset_import = helper.make_opsetid(domain="com.microsoft", version=1)
        model = helper.make_model(graph, opset_imports=[opset_import])
        return model

    def create_cache_path(self, model_type: str, use_redundant_squeeze_ops: bool):
        # Create position ids path
        reshape_node = helper.make_node(
            "Reshape",
            inputs=["position_ids", "pos_ids_new_shape"],
            outputs=["pos_ids_reshaped"],
            name="Reshape_0",
        )
        pos_ids_nodes = [reshape_node]

        # Create cos path
        cos_init_unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs=["new_seq_len", "zero"],
            outputs=["cos_unsqueeze"],
            name="Unsqueeze_2",
        )
        cos_slice_node = helper.make_node(
            "Slice",
            inputs=["cos_cache", "zero", "cos_unsqueeze", "two", "one"],
            outputs=["cos_sliced"],
            name="Slice_2",
        )
        cos_nodes = [cos_init_unsqueeze_node, cos_slice_node]

        if use_redundant_squeeze_ops:
            # These two nodes are eliminated by this transformers PR: https://github.com/huggingface/transformers/pull/26162
            cos_squeeze_1_node = helper.make_node(
                "Squeeze",
                inputs=["cos_sliced", "zero"],
                outputs=["cos_squeeze_1"],
                name="Squeeze_0",
            )
            cos_squeeze_2_node = helper.make_node(
                "Squeeze",
                inputs=["cos_squeeze_1", "zero"],
                outputs=["cos_squeeze_2"],
                name="Squeeze_1",
            )
            cos_nodes.extend([cos_squeeze_1_node, cos_squeeze_2_node])

        cos_gather_node = helper.make_node(
            "Gather",
            inputs=["cos_squeeze_2" if use_redundant_squeeze_ops else "cos_sliced", "pos_ids_reshaped"],
            outputs=["cos_indexed"],
            name="Gather_1",
        )
        cos_end_unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs=["cos_indexed", "one"],
            outputs=["cos"],
            name="Unsqueeze_3",
        )
        cos_nodes.extend([cos_gather_node, cos_end_unsqueeze_node])

        # Create sin path
        sin_init_unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs=["new_seq_len", "zero"],
            outputs=["sin_unsqueeze"],
            name="Unsqueeze_4",
        )
        sin_slice_node = helper.make_node(
            "Slice",
            inputs=["sin_cache", "zero", "sin_unsqueeze", "two", "one"],
            outputs=["sin_sliced"],
            name="Slice_3",
        )
        sin_nodes = [sin_init_unsqueeze_node, sin_slice_node]

        if use_redundant_squeeze_ops:
            sin_squeeze_1_node = helper.make_node(
                "Squeeze",
                inputs=["sin_sliced", "zero"],
                outputs=["sin_squeeze_1"],
                name="Squeeze_2",
            )
            sin_squeeze_2_node = helper.make_node(
                "Squeeze",
                inputs=["sin_squeeze_1", "zero"],
                outputs=["sin_squeeze_2"],
                name="Squeeze_3",
            )
            sin_nodes.extend([sin_squeeze_1_node, sin_squeeze_2_node])

        sin_gather_node = helper.make_node(
            "Gather",
            inputs=["sin_squeeze_2" if use_redundant_squeeze_ops else "sin_sliced", "pos_ids_reshaped"],
            outputs=["sin_indexed"],
            name="Gather_2",
        )
        sin_end_unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs=["sin_indexed", "one"],
            outputs=["sin"],
            name="Unsqueeze_5",
        )
        sin_nodes.extend([sin_gather_node, sin_end_unsqueeze_node])

        # Create beginning nodes before cos and sin paths

        # Create curr seq len path
        curr_transpose_node = helper.make_node(
            "Transpose",
            inputs=["curr_key"],
            outputs=["curr_key_transposed"],
            name="Transpose_curr",
            perm=[0, 2, 1, 3],
        )
        curr_shape_node = helper.make_node(
            "Shape",
            inputs=["curr_key_transposed"],
            outputs=["curr_shape"],
            name="Shape_curr",
        )
        curr_gather_node = helper.make_node(
            "Gather",
            inputs=["curr_shape", "two"],
            outputs=["curr_seq_len" if model_type in {"past", "merged"} else "new_seq_len"],
            name="Gather_curr",
        )
        beginning_nodes = [curr_transpose_node, curr_shape_node, curr_gather_node]

        if model_type in {"past", "merged"}:
            # Create past seq len path
            past_shape_node = helper.make_node(
                "Shape",
                inputs=["past_key"],
                outputs=["past_shape"],
                name="Shape_past",
            )
            past_gather_node = helper.make_node(
                "Gather",
                inputs=["past_shape", "two"],
                outputs=["past_seq_len"],
                name="Gather_past",
            )
            add_node = helper.make_node(
                "Add",
                inputs=["curr_seq_len", "past_seq_len"],
                outputs=["new_seq_len"],
                name="Add_1",
            )
            beginning_nodes.extend([past_shape_node, past_gather_node, add_node])

        if model_type == "merged":
            dummy_node = helper.make_node(
                "Add",
                inputs=["past_seq_len", "zero"],
                outputs=["past_seq_len_plus_zero"],
                name="Add_dummy_node",
            )
            beginning_nodes.append(dummy_node)

        return pos_ids_nodes + cos_nodes + sin_nodes + beginning_nodes

    def create_apply_rope_path(self):
        start_node = helper.make_node(
            "Transpose",
            inputs=["input_0"],
            outputs=["x"],
            name="Transpose_0",
            perm=[0, 2, 1, 3],
        )

        # Calculate x_half_shape
        shape_node = helper.make_node(
            "Shape",
            inputs=["x"],
            outputs=["x_shape"],
            name="Shape_0",
        )
        gather_node = helper.make_node(
            "Gather",
            inputs=["x_shape", "three"],
            outputs=["x_last_idx_shape"],
            name="Gather_0",
            axis=0,
        )
        div_node = helper.make_node(
            "Div",
            inputs=["x_last_idx_shape", "two"],
            outputs=["x_half_shape"],
            name="Div_0",
        )
        unsqueeze_0_node = helper.make_node(
            "Unsqueeze",
            inputs=["x_half_shape", "zero"],
            outputs=["x_half_shape_0"],
            name="Unsqueeze_0",
        )
        unsqueeze_1_node = helper.make_node(
            "Unsqueeze",
            inputs=["x_half_shape", "zero"],
            outputs=["x_half_shape_1"],
            name="Unsqueeze_1",
        )
        x_half_shape_nodes = [shape_node, gather_node, div_node, unsqueeze_0_node, unsqueeze_1_node]

        # Calculate rotate_half
        x1_node = helper.make_node(
            "Slice",
            inputs=["x", "zero", "x_half_shape_0", "three", "one"],
            outputs=["x1"],
            name="Slice_0",
        )
        x2_node = helper.make_node(
            "Slice",
            inputs=["x", "x_half_shape_1", "int_max", "three", "one"],
            outputs=["x2"],
            name="Slice_1",
        )
        neg_node = helper.make_node(
            "Neg",
            inputs=["x2"],
            outputs=["x2_neg"],
            name="Neg_0",
        )
        x_rotate_half_node = helper.make_node(
            "Concat",
            inputs=["x2_neg", "x1"],
            outputs=["x_rotate_half"],
            name="Concat_0",
            axis=-1,
        )
        rotate_half_nodes = [x1_node, x2_node, neg_node, x_rotate_half_node]

        # Calculate x_embed
        x_cos_node = helper.make_node(
            "Mul",
            inputs=["x", "cos"],
            outputs=["x_cos"],
            name="Mul_0",
        )
        x_sin_node = helper.make_node(
            "Mul",
            inputs=["x_rotate_half", "sin"],
            outputs=["x_rotate_half_sin"],
            name="Mul_1",
        )
        end_node = helper.make_node(
            "Add",
            inputs=["x_cos", "x_rotate_half_sin"],
            outputs=["output_0"],
            name="Add_0",
        )
        x_embed_nodes = [start_node, x_cos_node, x_sin_node, end_node]

        return x_half_shape_nodes + rotate_half_nodes + x_embed_nodes

    def create_test_model(self, model_type: str, use_redundant_squeeze_ops: bool, initializers: List[TensorProto]):
        apply_rope_nodes = self.create_apply_rope_path()
        cache_nodes = self.create_cache_path(model_type, use_redundant_squeeze_ops)
        inputs, outputs = self.create_inputs_and_outputs(model_type)

        graph = helper.make_graph(
            nodes=apply_rope_nodes + cache_nodes,
            name="RotaryEmbedding_Graph",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )
        opset_import = helper.make_opsetid(domain="ai.onnx", version=13)
        model = helper.make_model(graph, opset_imports=[opset_import])
        return model

    def check_models(self, interleaved: bool, model_type: str):
        initializers = self.create_initializers()

        expected_model_filename = "expected_model.onnx"
        expected_model = self.create_fused_model(interleaved, initializers)
        onnx.save(expected_model, expected_model_filename)

        original_model_filename = "original_model.onnx"
        use_redundant_squeeze_ops = True
        original_model = self.create_test_model(model_type, use_redundant_squeeze_ops, initializers)
        onnx.save(original_model, original_model_filename)

        self.verify_fusion(expected_model_filename, original_model_filename)
        os.remove(original_model_filename)

        use_redundant_squeeze_ops = False
        original_model = self.create_test_model(model_type, use_redundant_squeeze_ops, initializers)
        onnx.save(original_model, original_model_filename)

        self.verify_fusion(expected_model_filename, original_model_filename)
        os.remove(expected_model_filename)
        os.remove(original_model_filename)

    # Hugging Face's `decoder_model.onnx`
    def test_hf_decoder_model(self):
        interleaved = False  # HF model does not use interleaving
        model_type = "no_past"
        self.check_models(interleaved, model_type)

    # Hugging Face's `decoder_with_past_model.onnx`
    def test_hf_decoder_with_past_model(self):
        interleaved = False  # HF model does not use interleaving
        model_type = "past"
        self.check_models(interleaved, model_type)

    # Hugging Face's `decoder_merged.onnx`
    def test_hf_decoder_merged_model(self):
        interleaved = False  # HF model does not use interleaving
        model_type = "merged"
        self.check_models(interleaved, model_type)


if __name__ == "__main__":
    unittest.main()
