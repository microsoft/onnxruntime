from pathlib import Path
import unittest

from build_prefill512_dsa import EXTENDED_BOUND, ORIGINAL_BOUND, transform


SOURCE = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/dynamic_sparse_attention.cc")


class Prefill512AttentionTests(unittest.TestCase):
    def test_only_host_admission_changes(self):
        source = SOURCE.read_text()
        actual = transform(source)
        self.assertEqual(actual.replace(EXTENDED_BOUND, ORIGINAL_BOUND), source)
        self.assertEqual(actual[:actual.index(EXTENDED_BOUND)], source[:source.index(ORIGINAL_BOUND)])

    def test_extension_is_exact_flagged_bounded_single_batch_nvidia(self):
        for condition in ("prefill512 == 0 || prefill512 == 1", "prefill512 == 1",
                          'vendor == std::string_view{"nvidia"}', "parameters.batch_size == 1",
                          "parameters.sequence_length == 512", "maxComputeWorkgroupsPerDimension",
                          "extended_sequence ? 512u : kMaxSequenceLength"):
            self.assertIn(condition, EXTENDED_BOUND)
        self.assertIn("static_cast<uint64_t>(parameters.sequence_length) * parameters.num_heads", EXTENDED_BOUND)
        self.assertIn('NotImplementedBound("sequence_length"', EXTENDED_BOUND)


if __name__ == "__main__":
    unittest.main()
