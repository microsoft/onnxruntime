import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from build_prefill1280_dsa import ANCHOR, EXTENSION, transform as transform_dsa
from build_qmoe_prefill1280 import FALLBACK_ROWS, ROWS, transform, transform_compare
from qualify_prefill256 import main as qualify
from qualify_quickgelu_mul_long_context import ROOT
from run_prefill256_model import verify_prefill_flags


BASE = Path("/home/sayanshaw/repos")


class Prefill1280Tests(unittest.TestCase):
    def test_existing_chunk1024_standard_qualification_reproduces_without_writes(self):
        output = ROOT / "prefill1024-v2-contexts-20261030.qualification.json"
        expected = json.loads(output.read_text())
        exists = Path.exists
        captured = []

        def capture(path, text):
            self.assertEqual(path, output)
            captured.append(json.loads(text))
            return len(text)

        with patch.object(Path, "exists", lambda path: False if path == output else exists(path)), \
                patch.object(Path, "write_text", capture), contextlib.redirect_stdout(io.StringIO()):
            qualify(variant="prefill1024-v2", stamp="20261030", parent_variant="prefill512-v3",
                    parent_stamp="20261030", previous_chunk=512, chunk=1024)
        self.assertEqual(captured, [expected])

    def test_required_metadata_cannot_silently_drop_parent_or_candidate_flags(self):
        flags = {flag: "1" for flag in (
            "ORT_WEBGPU_DENSE_TILE4", "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_QSA_SCORE16",
            "ORT_WEBGPU_QMOE_PREFILL512", "ORT_WEBGPU_QMOE_PREFILL1024",
            "ORT_WEBGPU_QMOE_PREFILL1280", "ORT_WEBGPU_DSA_QUERY_CACHE")}
        verify_prefill_flags({"load_experiment_flags": flags}, 1280)
        for flag in flags:
            with self.assertRaises(AssertionError):
                verify_prefill_flags({"load_experiment_flags": dict(flags, **{flag: "0"})}, 1280)

    def test_only_exact_rows_fit_dispatch_budget(self):
        source = (BASE / "qwen38-int4-qmoe-prefill1024/build-v1/qmoe.cc").read_text()
        actual = transform(source)
        self.assertIn("(moe_params.num_rows == 768 || moe_params.num_rows == 1280)", actual)
        self.assertIn("moe_params.num_rows == 1280 ? 64000u : 38400u", actual)
        self.assertIn("use_prefill1280 ? 12800 : use_prefill1024 ? 10240 :", actual)
        self.assertIn("key.push_back(prefill1280);", actual)
        self.assertEqual((1280 * 10 * 640 + 127) // 128, 64000)
        self.assertLessEqual(64000, 65535)
        self.assertGreater((1536 * 10 * 640 + 127) // 128, 65535)
        self.assertEqual(32768 % 1280, 768)
        for rows in (768, 1280):
            self.assertLessEqual(rows * 24, 65535)

    def test_original_shaders_and_admission_are_preserved(self):
        source = (BASE / "qwen38-int4-dsa-query-cache/build-v2/dynamic_sparse_attention.cc").read_text()
        actual = transform_dsa(source)
        self.assertEqual(actual.replace(EXTENSION, ANCHOR), source)
        for condition in ("parameters.sequence_length == 768 || parameters.sequence_length == 1280",
                          "parameters.batch_size == 1", "maxComputeWorkgroupsPerDimension",
                          "extended_sequence1024 ? 1024u", "extended_sequence ? 512u"):
            self.assertIn(condition, actual)

    def test_native_boundaries_and_exact_comparison_budget(self):
        source = (BASE / "qwen38-int4-qmoe-prefill1024/build-v1/native_compare.cpp").read_text()
        actual = transform_compare(source)
        self.assertIn("{16, 128, 256, 512, 768, 1024, 1279, 1280, 1281}", actual)
        self.assertEqual(actual.count("rows == 1279 || rows == 1281"), 2)
        comparisons = 6 * 2560 * sum(rows * (2 if rows in FALLBACK_ROWS else 8) for rows in ROWS)
        self.assertEqual(comparisons, 568197120)


if __name__ == "__main__":
    unittest.main()
