from pathlib import Path
import unittest

from build_qmoe_prefill2048 import FALLBACK_ROWS, FLAG, ROWS, transform, transform_compare
from build_prefill2048_dsa import ANCHOR, EXTENSION, transform as transform_dsa
from run_prefill256_model import verify_prefill_flags


BASE = Path("/home/sayanshaw/repos")


class Prefill2048Tests(unittest.TestCase):
    def test_exact_gate_dispatch_coverage_within_each_device_dimension(self):
        source = (BASE / "qwen38-int4-qmoe-prefill1024/build-v1/qmoe.cc").read_text()
        actual = transform(source)
        self.assertIn("moe_params.num_rows == 2048", actual)
        self.assertIn("use_prefill2048 ? 20480 : use_prefill1024 ? 10240 :", actual)
        self.assertIn("key.push_back(prefill2048);", actual)
        self.assertIn("use_prefill2048 ? 2u : 1u", actual)
        self.assertEqual(2048 * 10 * 640 // 128, 51200 * 2)
        self.assertLessEqual(51200, 65535)
        self.assertLessEqual(2048 * 10, 65535)
        indices = [y * 51200 + x for y in range(2) for x in range(51200)]
        self.assertEqual(indices, list(range(102400)))
        helper = (BASE / "onnxruntime-webgpu-h200-perf/onnxruntime/core/providers/webgpu/shader_helper.cc").read_text()
        self.assertIn("workgroup_id.y * num_workgroups[0] + workgroup_id.x", helper)

    def test_attention_arithmetic_and_old_bounds_are_unchanged(self):
        source = (BASE / "qwen38-int4-dsa-query-cache/build-v2/dynamic_sparse_attention.cc").read_text()
        actual = transform_dsa(source)
        self.assertEqual(actual.replace(EXTENSION, ANCHOR), source)
        self.assertLessEqual(2048 * 24, 65535)
        self.assertIn("parameters.batch_size == 1 && parameters.sequence_length == 2048", actual)

    def test_native_boundaries_and_exact_comparison_budget(self):
        source = (BASE / "qwen38-int4-qmoe-prefill1024/build-v1/native_compare.cpp").read_text()
        actual = transform_compare(source)
        self.assertIn("{16, 128, 256, 512, 1024, 2047, 2048, 2049}", actual)
        self.assertEqual(actual.count("rows == 2047 || rows == 2049"), 2)
        self.assertEqual(6 * 2560 * sum(r * (2 if r in FALLBACK_ROWS else 8) for r in ROWS), 615383040)

    def test_all_parent_and_candidate_flags_are_required(self):
        names = ("ORT_WEBGPU_DENSE_TILE4", "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_QSA_SCORE16",
                 "ORT_WEBGPU_QMOE_PREFILL512", "ORT_WEBGPU_QMOE_PREFILL1024", FLAG,
                 "ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DENSE_BLOCK32")
        flags = {name: "1" for name in names}
        verify_prefill_flags({"load_experiment_flags": flags}, 2048)
        for name in names:
            with self.assertRaises(AssertionError):
                verify_prefill_flags({"load_experiment_flags": dict(flags, **{name: "0"})}, 2048)


if __name__ == "__main__":
    unittest.main()
