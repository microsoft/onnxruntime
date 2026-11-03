from pathlib import Path
import unittest

import numpy as np

from build_gdn_decode_view import transform


class DecodeViewTests(unittest.TestCase):
    def test_single_token_layout_is_exactly_identical(self):
        for query_size, value_size in ((7, 5), (66, 68), (2048, 6144)):
            packed = np.arange(2 * query_size + value_size, dtype=np.int32).reshape(1, -1)
            unpacked = np.concatenate((packed[:, :query_size].reshape(-1),
                                       packed[:, query_size:2 * query_size].reshape(-1),
                                       packed[:, 2 * query_size:].reshape(-1)))
            np.testing.assert_array_equal(packed.reshape(-1), unpacked)

    def test_two_total_tokens_are_not_an_identity_even_for_two_single_token_batches(self):
        packed = np.arange(2 * (2 * 7 + 5), dtype=np.int32).reshape(2, -1)
        unpacked = np.concatenate((packed[:, :7].reshape(-1), packed[:, 7:14].reshape(-1),
                                   packed[:, 14:].reshape(-1)))
        self.assertFalse(np.array_equal(packed.reshape(-1), unpacked))

    def test_transform_is_guarded_and_keeps_allocating_fallback(self):
        path = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/gated_delta_net.cc")
        text = transform(path.read_text())
        self.assertIn("input_is_packed_qkv && total_tokens == 1 && decode_view_flag == 1", text)
        self.assertIn('("ORT_WEBGPU_GDN_PACKED_DECODE_VIEW", 0)', text)
        self.assertIn("decode_view_flag == 0 || decode_view_flag == 1", text)
        self.assertEqual(text.count("&*packed_qkv"), 3)
        self.assertNotIn(".AddOutput({packed_qkv_input", text)
        self.assertNotIn("ProgramOutput::BufferView(packed_qkv_input", text)
        self.assertIn("packed_qkv_input = query;", text)
        self.assertIn("else if (use_packed_qkv)", text)
        self.assertIn("GatedDeltaNetUnpackQkvProgram unpack_program;", text)


if __name__ == "__main__":
    unittest.main()
