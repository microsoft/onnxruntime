import ast
from pathlib import Path
import unittest

from prepare_hc_narrow_rms_broadcast_native import comparator_source


class IndependentHCReferenceTests(unittest.TestCase):
    def test_phase_isolation_replay_and_coverage_gates(self):
        source = Path(__file__).with_name("run_hc_narrow_rms_broadcast_native.py").read_text()
        ast.parse(source)
        self.assertIn('("capture", ORIGINAL, "capture", 0)', source)
        self.assertIn('("default", provider, "compare", 0)', source)
        self.assertIn('("candidate", provider, "compare", 1)', source)
        self.assertIn('row["requests"] != 96', source)
        self.assertIn('row["binding_hits"] <= 0', source)
        self.assertIn('row["evictions"] <= 0', source)
        self.assertIn("active_candidate_kernel_coverage_verified=False", source)

    def test_original_input_schedule_and_independent_output_transfer(self):
        source = comparator_source()
        self.assertIn("for (int file = 4; file < argc; ++file)", source)
        self.assertIn("round < 128", source)
        self.assertIn('flag("0");', source)
        self.assertIn('flag("1");', source)
        self.assertIn('{"maxNumPendingDispatches", file % 2 ? "1" : "64"}', source)
        self.assertEqual(source.count("TransferOriginalOutput("), 1)
        self.assertIn("actual[output]", source)
        self.assertIn("Existing original reference", source)
        self.assertIn("Trailing original reference bytes", source)
        self.assertIn("catch (const NumericMismatch& error)", source)
        self.assertIn("return 2;", source)
        self.assertIn('PASS total_exact_elements=', source)


if __name__ == "__main__":
    unittest.main()
