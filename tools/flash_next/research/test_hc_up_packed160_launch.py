from pathlib import Path
import unittest

from hc_up_packed160_transform import FLAG, NAME
from run_hc_up_packed160_v1 import BUILD, command_for, matched, measured_gpu


class HCUpPacked160LaunchTests(unittest.TestCase):
    def test_real_selected_160_and_original_128_thread_events_are_required(self):
        for selected, name, threads in ((True, NAME, 160), (False, "HCUpPreMixPacked2F16", 128)):
            events = [{"cat": "Kernel", "name": "HCUpPreMixF16&" + name,
                       "pid": -1, "tid": -1, "ph": "X", "dur": index + 1,
                       "args": {"cache_key": f"key:{threads},1,1:shader"}} for index in range(288)]
            self.assertEqual(measured_gpu(events, selected), list(range(34, 289, 2)))
            events[0]["args"]["cache_key"] = f"key:{128 if selected else 160},1,1:shader"
            with self.assertRaises(RuntimeError):
                measured_gpu(events, selected)

    def test_strict_threshold_and_fixed_selector_cannot_admit_the_held_80_result(self):
        self.assertFalse(matched([13.0] * 128, [12.0] * 78 + [13.0] * 50,
                                 [13.0] * 128)["operator_gate_pass"])
        prefix = Path(__file__).parent / "packed160-test-only"
        command = command_for(BUILD / "libonnxruntime_providers_webgpu.so", "compare", prefix, 1,
                              BUILD.parent / "unused-test-only.onnx")
        self.assertIn(f"{FLAG}=1", command[-1])
        self.assertNotIn("ORT_WEBGPU_HC_UP_PACKED80", command[-1])
        native = (Path(__file__).parent / "hc_up_packed160_native.cpp").read_text()
        self.assertIn(FLAG, native)
        self.assertNotIn("ORT_WEBGPU_HC_UP_PACKED80", native)
        self.assertIn("round < 144", native)
        self.assertIn("repeat < 2", native)


if __name__ == "__main__":
    unittest.main()
