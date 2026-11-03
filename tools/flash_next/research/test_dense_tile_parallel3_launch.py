import shlex
import unittest

from run_dense_tile_parallel3_v1 import (
    BINARY, BUILD, DATA, FIXTURES, FLAG, NAME, ROOT, command_for, container, matched, measured_gpu,
)


class DenseTileParallelLaunchTests(unittest.TestCase):
    def test_native_cli_order_and_supported_flag_wiring(self):
        fixture = FIXTURES / "k2560-n2560.onnx"
        provider = BUILD / "libonnxruntime_providers_webgpu.so"
        reference = DATA / "reference.f16"
        prefix = ROOT / "unused-test-only"
        command = command_for("compare", provider, fixture, reference, prefix, 1)
        self.assertIn(f"{FLAG}=1", command[-1])
        for flag in ("DENSE_COLUMNS8_DOUBLE_BUFFER", "DENSE_BLOCK32", "GENERIC_MATMUL_REPLAY"):
            self.assertIn("ORT_WEBGPU_" + flag + "=1", command[-1])
        self.assertEqual(shlex.split(command[-1].split("exec ", 1)[1]), [
            container(BINARY), "compare", container(provider), container(fixture),
            container(reference), container(ROOT / "unused-test-only-profile"),
        ])

    def test_gpu_filter_demands_actual_selected_geometry_and_strict_consistency(self):
        for selected, threads in ((False, 128), (True, 384)):
            name = NAME if selected else "DenseColumns8DoubleBuffer"
            events = [{
                "cat": "Kernel", "name": "node&" + name, "pid": -1, "tid": -1,
                "ph": "X", "dur": 20, "args": {"cache_key": f":{threads},1,1:"},
            } for _ in range(144)]
            self.assertEqual(len(measured_gpu(events, selected)), 128)
            events[0]["args"]["cache_key"] = ":256,1,1:"
            with self.assertRaises(RuntimeError):
                measured_gpu(events, selected)
        self.assertTrue(matched([100] * 128, [80] * 128, [100] * 128)["operator_gate_pass"])
        self.assertFalse(matched([100] * 128, [80] * 119 + [100] * 9, [100] * 128)["operator_gate_pass"])
        self.assertFalse(matched([100] * 128, [99.5] * 128, [100] * 128)["operator_gate_pass"])


if __name__ == "__main__":
    unittest.main()
