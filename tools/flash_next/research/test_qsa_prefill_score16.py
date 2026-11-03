from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

from build_qsa_prefill_score16 import ANCHOR, FLAG, STOP, transform
from prepare_qsa_prefill_score16_fixtures import ROOT, reference_indices
from run_qsa_indexer_stack_native import programs
from run_qsa_prefill_score16_native import PROGRAM, verify_programs
from run_qsa_prefill_score16_model import model_command, prefill_program_count


class QsaPrefillScore16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qsa-score16/build-v5/"
                          "sparse_attention_indexer.cc").read_text()
        cls.result = transform(cls.source)

    def test_original_shader_and_decode_dispatch_unchanged(self):
        start, stop = self.source.index(ANCHOR), self.source.index(STOP)
        self.assertIn(self.source[start:stop], self.result)
        functions_end = self.source.index("Status SparseAttentionIndexer::ComputeQsa")
        self.assertEqual(self.source[:functions_end], self.result[:functions_end])

    def test_selector_is_default_off_and_rejects_invalid_values(self):
        self.assertIn(f'ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0)', self.result)
        self.assertIn("prefill_score16 == 0 || prefill_score16 == 1", self.result)
        self.assertIn(f"{FLAG} must be 0 or 1", self.result)

    def test_canonical_bounds_and_separate_cache_hint(self):
        for guard in ("batch_size == 1", "(rows == 512 || rows == 1024)", "rows == sequence_length",
                      "query->IsDataType<MLFloat16>()", "present->IsDataType<MLFloat16>()",
                      "num_heads == 4 && head_size == 128 && query_shape[2] == 640",
                      "compress_ratio_ == 4 && token_budget_ == 2048",
                      "score_stride >= 128 && score_stride <= 16384"):
            self.assertIn(guard, self.result)
        self.assertEqual(self.result.count('"prefill-score16-2d"'), 1)
        self.assertIn(".SetDispatchGroupSize(DivideRoundUp(ToUint32(score_stride), 16u), ToUint32(rows))",
                      self.result)

    def test_two_dimensional_flattening_and_candidate_tails(self):
        for rows in (512, 1024):
            for stride in (128, 129, 520, 8200, 16384):
                packed = (stride + 15) // 16
                self.assertLessEqual(max(rows, packed), 65535)
                for row in (0, rows - 1):
                    for group in (0, packed - 1):
                        flat = row * packed + group
                        self.assertEqual(divmod(flat, packed), (row, group))
                self.assertLess((packed - 1) * 16, stride)
                self.assertGreaterEqual(packed * 16, stride)
        self.assertGreater(1024 * ((8200 + 15) // 16), 65535)

    def test_reference_matches_existing_independent_frozen_vectors(self):
        for past in (2048, 32768):
            path = ROOT / "qsa-parallel-fixtures" / (
                f"qsa_past{past}_rows16_ratio4_half1_packed1_h4_d128_budget2048_model_rotary.onnx")
            values = {t.name: numpy_helper.to_array(t) for t in onnx.load(path).graph.initializer}
            keys = np.concatenate((values["past_key"][0], values["query"][0, :, -128:]))
            actual = reference_indices(values["query"][0, 0, :512].reshape(4, 128), keys,
                values["mask"][0], values["query_norm"], values["key_norm"], values["cos"], values["sin"],
                past, 16)
            reference = onnx.load(path.with_name(path.stem + "_reference.onnx"))
            expected = next(numpy_helper.to_array(t) for t in reference.graph.initializer
                            if t.name == "expected_indices")
            np.testing.assert_array_equal(actual, expected)

    def test_reference_empty_masks_zero_ties_and_uncompressed_tails(self):
        query, keys = np.ones((4, 128)), np.zeros((8, 128))
        cosine, sine = np.ones((8, 32)), np.zeros((8, 32))
        for visible in (False, True):
            actual = reference_indices(query, keys, np.full(8, visible), np.ones(128), np.ones(128),
                                       cosine, sine, 0, 8)
            for row in range(8):
                count = row + 1 if visible else 0
                np.testing.assert_array_equal(actual[0, row, :count], np.arange(count))
                self.assertTrue(np.all(actual[0, row, count:] == -1))

    def test_exact_program_coverage_and_unchanged_fallback_gates(self):
        before = programs([dict(cat="Kernel", name="node&SparseAttentionIndexerQsaScore")] * 18 +
                          [dict(cat="Kernel", name="node&QsaConcatVector4")] * 18)
        after = before.copy()
        after["SparseAttentionIndexerQsaScore"] = 0
        after[PROGRAM] = 18
        verify_programs(before, after, True)
        verify_programs(before, before, False)
        for changed in ("QsaConcatVector4", PROGRAM):
            bad = after.copy()
            bad[changed] -= 1
            with self.assertRaises(AssertionError):
                verify_programs(before, bad, True)
        with self.assertRaises(AssertionError):
            verify_programs(before, after, False)

    def test_model_command_preserves_protocol_and_fresh_evidence(self):
        provider = Path("/home/sayanshaw/repos/qwen38-int4-qsa-prefill-score16/build-v1/"
                        "libonnxruntime_providers_webgpu.so")
        old = ["bash", "-lc", "runner --tokens 256 --repetitions 3 --chunk 1024 "
               "--webgpu-validation-mode full --plugin /workspace/qwen38-int4-dense-block32/build-v2/"
               "libonnxruntime_providers_webgpu.so --output baseline.json STATS=baseline-stats STATIC=baseline-static"]
        for mode in (0, 1):
            for profile in (False, True):
                actual = model_command(old, Path("/evidence/new"), "baseline", provider, mode, profile)
                self.assertIn(f"export {FLAG}={mode}", actual[-1])
                self.assertIn("--tokens " + ("2" if profile else "36"), actual[-1])
                self.assertIn("--repetitions 1 --chunk 1024", actual[-1])
                self.assertIn("--webgpu-validation-mode full", actual[-1])
                self.assertEqual("--profile-prefix" in actual[-1], profile)
                self.assertNotIn("baseline", actual[-1])
                self.assertIn("baseline", old[-1])

    def test_prefill_coverage_does_not_count_decode_or_other_kernels(self):
        events = [
            dict(cat="Kernel", name="node&" + PROGRAM, args=dict(shapes="inputs[0] = {1,1024,640} ")),
            dict(cat="Kernel", name="node&" + PROGRAM, args=dict(shapes="inputs[0] = {1,1,640} ")),
            dict(cat="Kernel", name="node&Other", args=dict(shapes="inputs[0] = {1,1024,640} ")),
            dict(cat="Node", name="node&" + PROGRAM, args=dict(shapes="inputs[0] = {1,1024,640} ")),
        ]
        self.assertEqual(prefill_program_count(events), 1)


if __name__ == "__main__":
    unittest.main()
