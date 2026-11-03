import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from build_prefill1024_dsa import ANCHOR, EXTENSION, transform as transform_dsa
from build_qmoe_prefill1024 import transform, transform_compare
from qualify_prefill256 import main as qualify
from qualify_quickgelu_mul_long_context import ROOT


class Prefill1024Tests(unittest.TestCase):
    def test_existing_chunk512_standard_qualification_reproduces_without_writes(self):
        output = ROOT / "prefill512-v3-contexts-20261030.qualification.json"
        expected = json.loads(output.read_text())
        exists = Path.exists
        captured = []

        def capture(path, text):
            self.assertEqual(path, output)
            captured.append(json.loads(text))
            return len(text)

        with patch.object(Path, "exists", lambda path: False if path == output else exists(path)), \
                patch.object(Path, "write_text", capture), contextlib.redirect_stdout(io.StringIO()):
            qualify(variant="prefill512-v3", stamp="20261030", parent_variant="dense-tile4-v1",
                    parent_stamp="20261030", previous_chunk=256, chunk=512)
        self.assertEqual(captured, [expected])

    def test_exact_prefill_admission_dispatch_budget_and_existing_paths(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill512/build-v2/qmoe.cc").read_text()
        actual = transform(source)
        self.assertIn("device_limits.maxComputeWorkgroupsPerDimension >= 51200", actual)
        self.assertIn("moe_params.num_rows == 1024", actual)
        self.assertIn("use_prefill1024 ? 10240 : use_prefill512 ? 5120 :", actual)
        self.assertIn("key.push_back(prefill1024);", actual)
        comparator = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill512/build-v2/native_compare.cpp").read_text()
        comparator = transform_compare(comparator)
        self.assertIn("{16, 128, 256, 512, 1023, 1024, 1025}", comparator)
        self.assertIn("rows > 512 && rows != 1024", comparator)
        self.assertEqual((10240 * 640 + 127) // 128, 51200)

    def test_attention_only_admission_changes_with_legacy_bound_preserved(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill512/build-v3/dynamic_sparse_attention.cc").read_text()
        actual = transform_dsa(source)
        self.assertEqual(actual.replace(EXTENSION, ANCHOR), source)
        for condition in ("prefill1024 == 0 || prefill1024 == 1",
                          "parameters.sequence_length == 1024", "parameters.batch_size == 1",
                          'vendor == std::string_view{"nvidia"}', "maxComputeWorkgroupsPerDimension",
                          "extended_sequence1024 ? 1024u : extended_sequence ? 512u : kMaxSequenceLength"):
            self.assertIn(condition, actual)


if __name__ == "__main__":
    unittest.main()
