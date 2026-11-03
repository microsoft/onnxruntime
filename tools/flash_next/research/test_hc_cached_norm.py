from pathlib import Path
import unittest

from build_hc_cached_norm import FLAG, PROGRAM, STORAGE_BYTES, projection_program, transform


class CachedNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/"
                          "build-v2/hc_post_norm_projection.cc").read_text()

    def test_projection_math_and_sigmoid_are_byte_identical(self):
        start = self.source.index('         << "let lane = local_idx % 32u;')
        stop = self.source.index("    return Status::OK();", start)
        self.assertIn(self.source[start:stop], projection_program(self.source))
        self.assertIn(f'Program{{"{PROGRAM}"}}', projection_program(self.source))
        self.assertEqual(STORAGE_BYTES, 23552)

    def test_original_norm_and_fused_and_fallback_programs_preserved(self):
        result = transform(self.source)
        start = self.source.index("class HCPostNormProjectionProgram final")
        stop = self.source.index("class HCPostNormProjectionF16", start)
        self.assertIn(self.source[start:stop], result)
        start = self.source.index("    HCPostNormProjectionBaselineNormProgram norm")
        stop = self.source.index("    if (projection_fused)", start)
        self.assertIn(self.source[start:stop], result)
        self.assertIn("if (fused && cached_norm == 1)", result)
        start = result.index("    if (fused && cached_norm == 1)")
        stop = result.index("    if (fused) {", start)
        self.assertNotIn("replay_.Run", result[start:stop])
        self.assertIn("ORT_RETURN_IF_ERROR(context.RunProgram(norm))", result[start:stop])
        self.assertIn("return context.RunProgram(program)", result[start:stop])
        self.assertIn("return replay_.Run(context, {y, normalized, activated, tail}", result[stop:])
        self.assertIn(f"{FLAG} must be 0 or 1", result)

    def test_normalized_buffer_load_ownership_and_padding(self):
        indices = [i for thread in range(256) for i in range(thread, 10240, 256)]
        addresses = [(i // 32) * 36 + i % 32 for i in indices]
        self.assertEqual(sorted(indices), list(range(10240)))
        self.assertEqual(len(set(addresses)), 10240)
        self.assertLess(max(addresses), 11520)
        self.assertNotIn("var<workgroup> sums", projection_program(self.source))
        self.assertNotIn("inverseSqrt", projection_program(self.source))

    def test_window_cli_metadata_and_compound_dispatch_gate(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-hc-cached-norm-windows", benchmark)
        self.assertIn("args.compare_hc_cached_norm_windows or", benchmark)
        self.assertIn('"hc_cached_norm_decode_windows" if args.compare_hc_cached_norm_windows', benchmark)
        self.assertIn("HC_CACHED_NORM_FLAG if args.compare_hc_cached_norm_windows", benchmark)
        self.assertIn(f'"{FLAG}",', benchmark)
        self.assertIn('HC_CACHED_NORM_FLAG: "hc_cached_norm"',
                      (root / "qsa_rank_decode_windows.py").read_text())
        runner = (root / "run_hc_up_parallel2_model.py").read_text()
        self.assertIn('variant == "hc-cached-norm-v2"', runner)
        self.assertIn('native["compound_candidate_not_replay_cached"]', runner)


if __name__ == "__main__":
    unittest.main()
