from pathlib import Path
import unittest

from hc_up_packed80_transform import FLAG, NAME
from run_hc_up_packed80_v1 import BUILD, command_for, matched, measured_gpu


class HCUpPacked80LaunchTests(unittest.TestCase):
    def test_actual_selected_and_default_programs_require_different_exact_workgroups(self):
        for selected, name, threads in ((True, NAME, 80), (False, "HCUpPreMixPacked2F16", 128)):
            events = [{"cat": "Kernel", "name": "HCUpPreMixF16&" + name,
                       "pid": -1, "tid": -1, "ph": "X", "dur": index + 1,
                       "args": {"cache_key": f"key:{threads},1,1:shader"}} for index in range(288)]
            self.assertEqual(measured_gpu(events, selected), list(range(34, 289, 2)))
            events[0]["args"]["cache_key"] = f"key:{128 if selected else 80},1,1:shader"
            with self.assertRaises(RuntimeError):
                measured_gpu(events, selected)

    def test_scope_and_strict_thresholds_are_unchanged(self):
        self.assertFalse(matched([12.0] * 128, [12.0 / 1.009] * 128, [12.0] * 128)["operator_gate_pass"])
        self.assertFalse(matched([12.0] * 128, [11.0] * 119 + [13.0] * 9,
                                 [12.0] * 128)["operator_gate_pass"])
        native = (Path(__file__).parent / "hc_up_packed80_native.cpp").read_text()
        self.assertNotIn("ORT_WEBGPU_HC_UP_ZERO_TAIL", native)
        self.assertIn(FLAG, native)
        self.assertIn("round < 144", native)
        self.assertIn("repeat < 2", native)
        prefix = Path(__file__).parent / "packed80-test-only"
        fixture = BUILD.parent / "unused-test-only.onnx"
        command = command_for(BUILD / "libonnxruntime_providers_webgpu.so", "compare", prefix, 1, fixture)
        self.assertIn(f"{FLAG}=1", command[-1])
        self.assertNotIn("ORT_WEBGPU_HC_UP_ZERO_TAIL", command[-1])


if __name__ == "__main__":
    unittest.main()
