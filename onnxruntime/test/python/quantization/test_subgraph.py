# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import tempfile
import unittest
import urllib.request

import onnx

from onnxruntime.quantization import quantize_dynamic


class TestDynamicQuantizationSubgraph(unittest.TestCase):
    def test_dynamic_quantization_subgraph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "decoder_model_merged.onnx")
            quantized_onnx_path = os.path.join(tmpdir, "decoder_model_merged_quantized.onnx")
            url = "https://huggingface.co/fxmarty/t5-tiny-onnx-testing/resolve/main/decoder_model_merged.onnx"
            try:
                urllib.request.urlretrieve(url, onnx_path)
            except urllib.request.HTTPError as e:
                # The unit test should not fail for this kind of issue.
                # TODO: use another way to retrieve the model.
                raise unittest.SkipTest(f"Unable to fetch {url!r} due to {e}")  # noqa: B904

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
