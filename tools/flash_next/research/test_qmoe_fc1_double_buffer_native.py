import ast
from pathlib import Path
import unittest

from prepare_qmoe_fc1_double_buffer_native import comparator_source


class FC1IndependentReferenceTests(unittest.TestCase):
    def test_original_fixture_and_changing_route_schedule_preserved(self):
        source = comparator_source()
        self.assertIn("for (int file = 4; file < argc; ++file)", source)
        self.assertIn('{"maxNumPendingDispatches", file % 2 ? "1" : "64"}', source)
        self.assertIn("round < 32", source)
        self.assertIn("const int pattern = round % 6;", source)
        self.assertIn('set_flag("ORT_WEBGPU_QMOE_FC1_ACTIVATION", "0");', source)
        self.assertIn('set_flag("ORT_WEBGPU_QMOE_FC1_ACTIVATION", "1");', source)
        self.assertIn('set_flag("ORT_WEBGPU_QMOE_FC1_ACTIVATION", "2");', source)
        self.assertNotIn("ORT_WEBGPU_QMOE_FC2_COMPACT20", source)
        self.assertNotIn("set_flag(\"ORT_WEBGPU_QMOE_FC1_DOUBLE_BUFFER\"", source)

    def test_separate_outputs_and_error_classification(self):
        source = comparator_source()
        self.assertEqual(source.count("TransferOriginalOutput("), 1)
        self.assertIn("original, capture, actual[0]", source)
        self.assertIn("Existing original reference", source)
        self.assertIn("Trailing original reference bytes", source)
        self.assertIn("Cannot close original reference", source)
        self.assertIn("throw NumericMismatch", source)
        self.assertIn("catch (const NumericMismatch& error)", source)
        self.assertIn("return 2;", source)
        self.assertIn("return 1;", source)

    def test_native_phase_replay_and_coverage_gates(self):
        source = Path(__file__).with_name("run_qmoe_fc1_double_buffer_native.py").read_text()
        ast.parse(source)
        self.assertIn('("capture", ORIGINAL, "capture", 0)', source)
        self.assertIn('("default", provider, "compare", 0)', source)
        self.assertIn('("candidate", provider, "compare", 1)', source)
        self.assertIn('row["requests"] != 64', source)
        self.assertIn('row["captures"] + row["hits"] != row["requests"]', source)
        self.assertIn("active_candidate_kernel_coverage_verified=False", source)
        self.assertIn("model_eligible=False", source)
        self.assertIn('raise RuntimeError("Changed sealed original output")', source)


if __name__ == "__main__":
    unittest.main()
