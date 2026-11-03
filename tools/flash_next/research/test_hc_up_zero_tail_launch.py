import math
from pathlib import Path
import unittest

from hc_up_zero_tail_transform import FLAG, NAME
from run_hc_up_zero_tail_v1 import BUILD, command_for, matched, measured_gpu


class HCUpZeroTailLaunchTests(unittest.TestCase):
    def test_actual_events_are_paired_with_exactly_sixteen_excluded_warmup_pairs(self):
        events = [{"cat": "Kernel", "name": "HCUpPreMixF16&" + NAME,
                   "pid": -1, "tid": -1, "ph": "X", "dur": index + 1,
                   "args": {"cache_key": "key:128,1,1:shader"}} for index in range(288)]
        self.assertEqual(measured_gpu(events, True), list(range(34, 289, 2)))
        for change in ({"dur": 0}, {"dur": math.nan}, {"pid": 1}, {"name": "HCUpPreMixPacked2F16"},
                       {"args": {"cache_key": "key:80,1,1:shader"}}):
            invalid = events.copy()
            invalid[20] = {**invalid[20], **change}
            with self.assertRaises(RuntimeError):
                measured_gpu(invalid, True)
        with self.assertRaises(RuntimeError):
            measured_gpu(events[:-1], True)

    def test_operator_gate_requires_both_strict_one_percent_and_120_improvements(self):
        before = after = [12.0] * 128
        self.assertTrue(matched(before, [11.0] * 128, after)["operator_gate_pass"])
        self.assertFalse(matched(before, [12.0 / 1.009] * 128, after)["operator_gate_pass"])
        candidate = [11.0] * 119 + [13.0] * 9
        self.assertGreater(matched(before, candidate, after)["matched_gpu_speed_ratio"], 1.01)
        self.assertFalse(matched(before, candidate, after)["operator_gate_pass"])
        with self.assertRaises(ValueError):
            matched(before, [0] * 128, after)

    def test_fixed_private_selector_and_unchanged_original_carrier_are_explicit(self):
        prefix = Path(__file__).parent / "hc-up-zero-tail-test-only"
        fixture = BUILD.parent / "actual-fixtures-v1/anchor0-rows1.onnx"
        command = command_for(BUILD / "libonnxruntime_providers_webgpu.so",
                              "compare", prefix, 1, fixture)
        self.assertIn(f"{FLAG}=1", command[-1])
        self.assertIn("ORT_WEBGPU_HC_UP_PREMIX=1", command[-1])
        self.assertIn("ORT_WEBGPU_HC_UP_PARALLEL2=1", command[-1])
        self.assertIn("ORT_TEST_HC_UP_NATIVE_PROFILE=1", command[-1])
        self.assertIn("ORT_TEST_HC_UP_FP32_ACCUMULATION=0", command[-1])
        self.assertIn("hc_up_compare /workspace/", command[-1])
        self.assertIn(" compare /workspace/", command[-1])
        with self.assertRaises(ValueError):
            command_for(BUILD / "libonnxruntime_providers_webgpu.so", "compare", prefix, 3, fixture)


if __name__ == "__main__":
    unittest.main()
