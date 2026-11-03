from pathlib import Path
import unittest

from prepare_cpu_decode_deep_stack import specialize


class CpuDecodeDeepStackTests(unittest.TestCase):
    def test_only_prefix_and_stack_depth_change(self):
        source = Path(__file__).with_name("run_cpu_decode_sample_v2.py").read_text()
        result = specialize(source)
        restored = result.replace('VARIANT = "cpu-decode-deep-stack-v1"', 'VARIANT = "cpu-decode-sample-v2"')
        restored = restored.replace('"dwarf,32768"', '"dwarf,8192"')
        self.assertEqual(restored, source)
        self.assertIn('"-e", "cpu-clock:u"', result)
        self.assertIn("diagnostic_only=True", result)
        self.assertIn("parent.owned_model_pid(container_id, start)", result)


if __name__ == "__main__":
    unittest.main()
