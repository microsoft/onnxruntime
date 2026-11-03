import unittest

from run_hc_owned_pair34_v1 import BENCHMARK, BUILD, FLAG, NAME, ROOT, command_for, container, measured_gpu


class PairLaunchTests(unittest.TestCase):
    def test_isolated_launches_and_original_provider_controls(self):
        prefix = ROOT / "owned-pair-test"
        for candidate in (False, True):
            command = command_for(BENCHMARK, prefix, int(candidate), candidate, ["fixture", "report", "-"])
            self.assertEqual(command[:5], ["taskset", "-c", "0-7", "bash", "-lc"])
            self.assertIn(f"{FLAG}={int(candidate)}", command[-1])
            self.assertEqual(container(BUILD / "libonnxruntime_providers_webgpu.so") in command[-1], candidate)
            self.assertIn("ORT_WEBGPU_QMOE_FC2_PRIVATE_A=0", command[-1])
            self.assertIn("ORT_WEBGPU_STATIC_NODE_REPLAY=1", command[-1])
        self.assertIn(f"{FLAG}=2", command_for(BENCHMARK, prefix, 2, True, ["fixture"])[-1])
        with self.assertRaises(ValueError):
            command_for(BENCHMARK, prefix, 3, True, [])

    def test_only_exact_actual_gpu_program_and_workgroup_are_accepted(self):
        for flag, name in ((False, "HCPostNormProjectionF16"), (True, NAME)):
            events = [dict(cat="Kernel", pid=-1, tid=-1, ph="X", dur=1, name="node&" + name,
                           args={"cache_key": f"{name}[hint]:256,1,1:type"}) for _ in range(144)]
            self.assertEqual(len(measured_gpu(events, flag)), 128)
            with self.assertRaises(RuntimeError):
                measured_gpu(events[:-1], flag)
            events[0]["args"]["cache_key"] = "program[hint]:128,1,1:type"
            with self.assertRaises(RuntimeError):
                measured_gpu(events, flag)


if __name__ == "__main__":
    unittest.main()
