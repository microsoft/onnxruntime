import json
import unittest

from run_decode_gpu_tail_profile_32k_v1 import TEMPLATE, VARIANT, gpu_events, parent, profile_command


class LongContextProfileTests(unittest.TestCase):
    def test_only_fresh_diagnostic_prefix_repetitions_and_profile_scope_change(self):
        template = json.loads((parent.ROOT / (TEMPLATE + ".guard.json")).read_text())["command"]
        command = profile_command(template)
        self.assertEqual(command[:5], template[:5])
        self.assertIn("--contexts 32768", command[-1])
        self.assertIn("--repetitions 1", command[-1])
        self.assertIn("--profile-decode-after 252", command[-1])
        self.assertIn("--webgpu-validation-mode full", command[-1])
        self.assertNotIn(TEMPLATE, command[-1])
        restored = command[-1].split(" --webgpu-validation-mode full", 1)[0]
        restored = restored.replace(VARIANT, TEMPLATE).replace("--repetitions 1", "--repetitions 3")
        self.assertEqual(restored, template[-1])

    def test_strict_native_gpu_event_filter(self):
        good = dict(cat="Kernel", pid=-1, tid=-1, ph="X", name="node&program", dur=1)
        self.assertEqual(gpu_events([good, dict(cat="Node", dur=100)]), [good])
        with self.assertRaises(RuntimeError):
            gpu_events([dict(good, dur=0)])
        with self.assertRaises(RuntimeError):
            gpu_events([dict(good, dur=float("nan"))])
        with self.assertRaises(RuntimeError):
            gpu_events([dict(good, pid=1)])


if __name__ == "__main__":
    unittest.main()
