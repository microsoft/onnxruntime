import json
import unittest

from prepare_webgpu_int4_metrics import ROOT
from run_decode_gpu_tail_profile_v1 import TEMPLATE, gpu_events, profile_command


class DecodeGpuTailProfileTests(unittest.TestCase):
    def test_exact_launch_changes_preserve_runtime_selectors_and_placement(self):
        template = json.loads((ROOT / (TEMPLATE + ".guard.json")).read_text())["command"]
        command = profile_command(template)
        self.assertEqual(command[:5], template[:5])
        self.assertIn("--contexts 2048 --tokens 256 --repetitions 1", command[-1])
        self.assertIn("--webgpu-validation-mode full", command[-1])
        self.assertIn("--profile-decode-after 252", command[-1])
        self.assertIn("ORT_WEBGPU_QMOE_FC2_PRIVATE_A=0", command[-1])
        self.assertIn("ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH=0", command[-1])
        self.assertIn("ORT_GENAI_WEBGPU_GREEDY_ARGMAX=1", command[-1])
        self.assertIn("ORT_GENAI_WEBGPU_GPU_SLICE_COPY=1", command[-1])
        self.assertNotIn(TEMPLATE, command[-1])
        self.assertEqual(command[-1].split("; exec python3")[0],
                         template[-1].split("; exec python3")[0].replace(TEMPLATE, "decode-gpu-tail-profile-v1"))

    def test_rejects_changed_template(self):
        with self.assertRaises(RuntimeError):
            profile_command(["taskset", "-c", "16-23", "bash", "-lc", "not the template"])

    def test_filters_real_gpu_events_and_rejects_bad_durations(self):
        event = dict(cat="Kernel", pid=-1, tid=-1, ph="X", name="native shader", dur=17.0)
        cpu = dict(event, pid=123, tid=123)
        self.assertEqual(gpu_events([event, cpu]), [event])
        with self.assertRaises(RuntimeError):
            gpu_events([cpu])
        for duration in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(RuntimeError):
                gpu_events([dict(event, dur=duration)])


if __name__ == "__main__":
    unittest.main()
