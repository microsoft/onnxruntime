import unittest

from run_qmoe_invocation_flags_v1 import BUILD, FLAG, ROOT, command_for


class InvocationLaunchTests(unittest.TestCase):
    def test_native_and_batch_use_new_provider_flag_and_existing_safe_assets(self):
        for candidate in (False, True):
            for operator in (False, True):
                command = command_for(
                    ROOT / "unused-invocation-flags-launch-test", 1, 0,
                    candidate, operator, operator)
                self.assertEqual(command[:5], ["taskset", "-c", "0-7", "bash", "-lc"])
                shell = command[-1]
                self.assertIn(f"export {FLAG}=1;", shell)
                self.assertNotIn("ORT_WEBGPU_QMOE_EARLY_SCRATCH", shell)
                self.assertNotIn("/qwen38-int4-qmoe-early-scratch/build-v1/", shell)
                self.assertIn("ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1", shell)
                self.assertIn("ORT_WEBGPU_QMOE_SUBGROUP_GATE=1", shell)
                if candidate:
                    self.assertIn(BUILD.parent.name + "/build-v1/libonnxruntime", shell)
                if operator:
                    self.assertIn("/fixtures/early_scratch_batch32_v2.onnx", shell)
                    self.assertIn("unused-invocation-flags-launch-test-ort-profile", shell)
                else:
                    self.assertIn("/native-v2/native_compare", shell)


if __name__ == "__main__":
    unittest.main()
