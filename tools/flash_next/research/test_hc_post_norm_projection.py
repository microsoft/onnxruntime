import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fuse_hc_post_norm_projection import DOMAIN, OPS, fuse
from build_hc_post_norm_four_columns import comparator_text, transform


def fixture(shared=False):
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape) for name, shape in
              (("streams", [1, 1, 10240]), ("branch", [1, 1, 2560]), ("gate", [1, 1, 4]))]
    norm_inputs = ["streams", "branch", "gate", "norm_scale"]
    if shared:
        inputs += [helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape) for name, shape in
                   (("shared", [1, 1, 2560]), ("shared_gate", [1, 1, 1]))]
        norm_inputs += ["shared", "shared_gate"]
    op = "HCSharedPostNormF16" if shared else "HCPostNormF16"
    nodes = [
        helper.make_node(op, norm_inputs, ["y", "normalized"], domain=DOMAIN, epsilon=1e-6),
        helper.make_node("Identity", ["normalized"], ["saved_normalized"]),
        helper.make_node("HCProjectionSiLUF16", ["normalized", "sizes", "weights", "scales"],
            ["activated", "tail"], domain=DOMAIN, K=10240, N=324, bits=4, block_size=32,
            accuracy_level=4, alpha=1.0)]
    initializers = [
        numpy_helper.from_array(np.ones(2560, dtype=np.float16), "norm_scale"),
        numpy_helper.from_array(np.array([320, 4], dtype=np.int64), "sizes"),
        numpy_helper.from_array(np.zeros((324, 320, 16), dtype=np.uint8), "weights"),
        numpy_helper.from_array(np.ones((324, 320), dtype=np.float16), "scales")]
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape) for name, shape in
               (("y", [1, 1, 10240]), ("saved_normalized", [1, 1, 10240]),
                ("activated", [1, 1, 320]), ("tail", [1, 1, 4]))]
    graph = helper.make_graph(nodes, "post_norm_projection", inputs, outputs, initializers,
        value_info=[helper.make_tensor_value_info("normalized", TensorProto.FLOAT16, [1, 1, 10240])])
    return helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 21), helper.make_opsetid(DOMAIN, 1)], ir_version=10)


class PostNormProjectionTests(unittest.TestCase):
    def test_preserves_four_outputs_other_consumers_and_initializers(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                original = fixture(shared)
                candidate, count = fuse(original)
                self.assertEqual(count, 1)
                self.assertEqual(len(candidate.graph.node), 2)
                self.assertEqual(list(candidate.graph.node[0].output),
                                 ["y", "normalized", "activated", "tail"])
                self.assertEqual(candidate.graph.node[1].input[0], "normalized")
                self.assertEqual([t.SerializeToString() for t in original.graph.initializer],
                                 [t.SerializeToString() for t in candidate.graph.initializer])
                self.assertEqual(len(original.graph.node), 3)
                onnx.checker.check_model(candidate)

    def test_rejects_late_projection_dependencies(self):
        model = fixture()
        model.graph.node.insert(1, helper.make_node("Identity", ["weights"], ["late_weights"]))
        model.graph.value_info.append(helper.make_tensor_value_info(
            "late_weights", TensorProto.UINT8, [324, 320, 16]))
        model.graph.node[-1].input[2] = "late_weights"
        _, count = fuse(model)
        self.assertEqual(count, 0)

    def test_rejects_repeated_fusion_and_noncanonical_attributes(self):
        candidate, _ = fuse(fixture())
        with self.assertRaisesRegex(ValueError, "Already fused"):
            fuse(candidate)
        model = fixture()
        model.graph.node[-1].attribute.append(helper.make_attribute("unsupported", 1))
        _, count = fuse(model)
        self.assertEqual(count, 0)
        self.assertEqual({f.name for f in candidate.functions}, {OPS["HCPostNormF16"]})

    def test_rejects_non_fp16_and_unknown_normalization_attributes(self):
        for invalid_type in (TensorProto.FLOAT, TensorProto.BFLOAT16):
            with self.subTest(invalid_type=invalid_type):
                model = fixture()
                model.graph.input[0].type.tensor_type.elem_type = invalid_type
                self.assertEqual(fuse(model)[1], 0)
        model = fixture()
        model.graph.node[0].attribute.append(helper.make_attribute("unsupported", 1))
        self.assertEqual(fuse(model)[1], 0)

    def test_padded_normalization_storage_matches_every_projection_read(self):
        writes = {(logical // 32) * 36 + logical % 32 for logical in range(10240)}
        reads = {block * 36 + i * 8 + half * 4 + component
                 for block in range(320) for i in range(4)
                 for half in range(2) for component in range(4)}
        self.assertEqual(writes, reads)
        self.assertEqual(len(writes), 10240)
        self.assertLess(max(writes), 11520)
        columns = [group * 8 + local // 32 for group in range(41)
                   for local in range(0, 256, 32)]
        self.assertEqual([col for col in columns if col < 324], list(range(324)))
        self.assertEqual(11520 * 2 + 1024 * 4 + 256 * 2, 27648)

    def test_four_column_variant_has_complete_unique_outputs(self):
        columns = [group * 4 + local // 32 for group in range(81)
                   for local in range(0, 256, 32) if local < 128]
        self.assertEqual(columns, list(range(324)))
        self.assertEqual(81 * 4, 324)

    def test_four_column_variant_preserves_baselines_and_separates_replay(self):
        parent = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2")
        source = (parent / "hc_post_norm_projection.cc").read_text()
        candidate = transform(source)
        stop = source.index("class HCPostNormProjectionProgram final")
        self.assertEqual(source[:stop], candidate[:stop])
        self.assertIn("four_columns ? replay_four_ : replay_", candidate)
        self.assertIn("four_columns ? 81 : 41", candidate)
        self.assertIn("local_idx < 128u && col < 324u", candidate)
        self.assertIn("local_idx < 128u && lane == 0u && col < 324u", candidate)
        self.assertIn("program.CacheHint(shared_, shared_scale, four_columns)", candidate)
        comparator = comparator_text((parent / "native_compare.cpp").read_text())
        self.assertIn('setenv("ORT_WEBGPU_HC_POST_NORM_FOUR_COLUMNS", "2"', comparator)


if __name__ == "__main__":
    unittest.main()
