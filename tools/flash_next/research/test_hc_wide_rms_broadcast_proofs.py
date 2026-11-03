import ast
from collections import Counter
import unittest

from prepare_hc_wide_rms_broadcast_proofs import (
    ROOT, invalid_source, model_source, native_source, screen_source, standard_source,
)


class WideRmsProofTests(unittest.TestCase):
    def test_reused_comparator_and_fresh_original_outputs(self):
        source = native_source()
        ast.parse(source)
        self.assertIn('qwen38-int4-hc-narrow-rms-broadcast/native-proof-v1', source)
        self.assertIn('DATA = BUILD.parent / "original-references-v1"', source)
        self.assertIn('qwen38-int4-hc-wide-rms-broadcast/build-v1', source)
        self.assertIn('("capture", ORIGINAL, "capture", 0)', source)
        self.assertIn('("default", provider, "compare", 0)', source)
        self.assertIn('("candidate", provider, "compare", 1)', source)
        self.assertIn("active_candidate_kernel_coverage_verified=False", source)

    def test_actual_model_gate_and_correct_selector_family(self):
        source = model_source()
        ast.parse(source)
        self.assertIn('native_proof["comparisons_per_complete_phase"] == 607143936', source)
        namespace = {"__name__": "model-test", "__file__": str(ROOT / "run_hc_wide_rms_broadcast_model.py")}
        exec(compile(source, "wide-model", "exec"), namespace)
        counts = Counter(HCPostNormProjectionWideRmsBroadcastF16=94, MatMulNBitsBlock32F16=159,
                         DynamicSparseAttentionQueryCache=12)
        namespace["verify_profile"](counts, "candidate")
        counts["HCPostNormProjectionF16"] = 1
        with self.assertRaises(AssertionError):
            namespace["verify_profile"](counts, "candidate")
        for build_source in (standard_source(), screen_source(), invalid_source()):
            ast.parse(build_source)
            self.assertNotIn("hc-narrow-rms-broadcast", build_source)
            self.assertNotIn("hc_narrow_rms_broadcast", build_source)
        self.assertIn('if len(fixtures) != 36:', invalid_source())
        self.assertIn('previous = 2048 if args.context == 8192 else 8192', standard_source())
        self.assertIn("repeat-screen", standard_source())


if __name__ == "__main__":
    unittest.main()
