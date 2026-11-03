import unittest

from prepare_unused_scratch_native import SOURCE, transform
from run_unused_scratch_native_v2 import FLAG, ROOT, command_for


class NativeScratchTests(unittest.TestCase):
    def test_selected_launch_changes_only_current_provider_and_owned_outputs(self):
        prefix = ROOT / "scratch-selected-test-gpu0"
        command = command_for(prefix, 1, 0, True, operator=True)
        self.assertEqual(command[:5], ["taskset", "-c", "0-7", "bash", "-lc"])
        self.assertIn(f"{FLAG}=1", command[-1])
        self.assertIn("ORT_WEBGPU_QMOE_COMMAND_REPLAY=0", command[-1])
        self.assertIn("ORT_WEBGPU_QMOE_FC2_PRIVATE_A=0", command[-1])
        self.assertIn("scratch-selected-test-gpu0.json", command[-1])
        self.assertNotIn("fc2-private-a-v1-before-gpu0.json", command[-1])
        self.assertNotIn("native-flag0-gpu0-shaders.txt", command[-1])

    def test_invalid_selector_and_replay_are_explicit(self):
        prefix = ROOT / "scratch-invalid-test-gpu0"
        command = command_for(prefix, 2, 0, True)
        self.assertIn(f"{FLAG}=2", command[-1])
        replay = command_for(prefix, 1, 1, True)
        self.assertIn("ORT_WEBGPU_QMOE_COMMAND_REPLAY=1", replay[-1])
        with self.assertRaises(ValueError):
            command_for(prefix, 3, 0, True)

    def test_all_native_output_bytes_enter_fingerprint_without_removing_comparison(self):
        changed = transform(SOURCE.read_text())
        self.assertIn("std::memcmp(&expected, &value, sizeof(value))", changed)
        self.assertIn("byte < sizeof(value)", changed)
        self.assertIn("fingerprint ^= bytes[byte]", changed)
        self.assertIn("round < 32", changed)
        self.assertIn('set_flag("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", "0")', changed)
        self.assertIn('set_flag("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", "1")', changed)


if __name__ == "__main__":
    unittest.main()
