import unittest

from run_early_scratch_operator_v2 import FIXTURE, ROOT, command_for


class FixtureLaunchTests(unittest.TestCase):
    def test_corrected_fixture_is_inside_original_external_data_directory(self):
        self.assertEqual(FIXTURE.parent.name, "fixtures")
        self.assertTrue((FIXTURE.parent / "model.onnx.data").is_file())
        prefix = ROOT / "early-scratch-v2-test"
        for candidate in (False, True):
            command = command_for(prefix, int(candidate), candidate, True)
            self.assertIn(FIXTURE.name, command[-1])
            self.assertNotIn("/native-v1/qmoe_batch32.onnx", command[-1])
            self.assertIn("ORT_WEBGPU_QMOE_SUBGROUP_GATE=1", command[-1])
            self.assertIn("ORT_WEBGPU_QMOE_COMMAND_REPLAY=0", command[-1])
            self.assertTrue(command[-1].endswith("early-scratch-v2-test-ort-profile"))


if __name__ == "__main__":
    unittest.main()
