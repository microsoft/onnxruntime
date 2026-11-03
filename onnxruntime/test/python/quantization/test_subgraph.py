# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import tempfile
import unittest

import numpy as np
import onnx

from onnxruntime.quantization import quantize_dynamic


def _make_merged_decoder_like_model():
    """Builds a minimal model shaped like a merged (with-past/without-past) decoder.

    A single float initializer ("shared.weight_merged_0") lives on the top-level graph and is
    consumed by a Gather node inside each branch of an If node, mirroring the structure of the
    merged T5 decoder model that this test used to download from Hugging Face. This keeps the
    test deterministic and avoids any network dependency, while still exercising the same
    subgraph-quantization behavior: hoisting the quantized initializer (and its scale) for an
    outer-scope weight back up to the top-level graph.
    """
    shared_weight_name = "shared.weight_merged_0"
    shared_weight = onnx.numpy_helper.from_array(
        (np.arange(32, dtype=np.float32) / 10.0).reshape(8, 4), shared_weight_name
    )

    input_ids = onnx.helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, [3])
    use_cache_branch = onnx.helper.make_tensor_value_info("use_cache_branch", onnx.TensorProto.BOOL, [])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [3, 4])

    def make_branch(name):
        gather_output = f"{name}_gather_out"
        # Gather references outer-scope tensors ("shared.weight_merged_0" and "input_ids")
        # directly, as merged decoder models with a with-past/without-past If node do.
        gather_node = onnx.helper.make_node("Gather", [shared_weight_name, "input_ids"], [gather_output], axis=0)
        branch_output = onnx.helper.make_tensor_value_info(gather_output, onnx.TensorProto.FLOAT, [3, 4])
        return onnx.helper.make_graph([gather_node], name, [], [branch_output])

    if_node = onnx.helper.make_node(
        "If",
        ["use_cache_branch"],
        ["output"],
        then_branch=make_branch("then_branch"),
        else_branch=make_branch("else_branch"),
    )

    graph = onnx.helper.make_graph(
        [if_node],
        "decoder_model_merged_like",
        [input_ids, use_cache_branch],
        [output],
        initializer=[shared_weight],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return model


class TestDynamicQuantizationSubgraph(unittest.TestCase):
    def test_dynamic_quantization_subgraph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "decoder_model_merged.onnx")
            quantized_onnx_path = os.path.join(tmpdir, "decoder_model_merged_quantized.onnx")
            onnx.save(_make_merged_decoder_like_model(), onnx_path)

            quantize_dynamic(
                model_input=onnx_path,
                model_output=quantized_onnx_path,
                per_channel=True,
                op_types_to_quantize=[
                    "Conv",
                    "MatMul",
                    "Attention",
                    "LSTM",
                    "Gather",
                    "Transpose",
                    "EmbedLayerNormalization",
                ],
                extra_options={"EnableSubgraph": True},
            )
            model = onnx.load(quantized_onnx_path)

            # The initializer `shared.weight_merged_0` is attached to the top-level graph, and used in a Gather node in each subgraphs.
            # We expect the quantized Gather (after which a DequantizeLinear is attached) initializer to also be attached to the top-level graph.
            found_gather_quantized = False
            for initializer in model.graph.initializer:
                if initializer.name == "shared.weight_merged_0_quantized":
                    found_gather_quantized = True
                    break
            self.assertTrue(found_gather_quantized)

            found_gather_scale = False
            for initializer in model.graph.initializer:
                if initializer.name == "shared.weight_merged_0_scale":
                    found_gather_scale = True
                    break
            self.assertTrue(found_gather_scale)

            # No initializers related to the Gather node should be attached to the subgraphs.
            for node in model.graph.node:
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        for initializer in attr.g.initializer:
                            self.assertTrue("shared.weight" not in initializer.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
