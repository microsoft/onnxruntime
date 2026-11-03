import json
from pathlib import Path
import unittest

from qualify_quickgelu_mul_long_context import ROOT
from run_qmoe_prefill_rows4_standard import TEMPLATE
from run_row4_default_cpu_policy import cpu_command, verify_cpu_policy


class DefaultCpuPolicyTests(unittest.TestCase):
    def test_controls_preserve_default_workers_and_only_candidate_changes_affinity(self):
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        provider = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/"
                        "libonnxruntime_providers_webgpu.so")
        for placement in ("baseline", "local8"):
            command = cpu_command(template, provider, ROOT / "default-policy-unit-test", 2048, True, placement)
            self.assertIn("--tokens 256 --repetitions 3", command[-1])
            self.assertIn("--chunk 1024", command[-1])
            self.assertEqual("--inherit-worker-cpu-affinity" in command[-1], placement == "local8")
            self.assertEqual(command[:3] == ["taskset", "-c", "48-55"], placement == "local8")

    def test_default_control_requires_original_pins_and_no_override(self):
        raw = dict(cpu_affinity=list(range(96)), thread_cpu_affinity={"1": list(range(96)), "2": [1]})
        verify_cpu_policy(raw, "baseline")
        with self.assertRaises(AssertionError):
            verify_cpu_policy(dict(raw, worker_affinity_override={"target_cpus": list(range(96))}), "baseline")
        with self.assertRaises(AssertionError):
            verify_cpu_policy(dict(raw, thread_cpu_affinity={"1": list(range(96))}), "baseline")

    def test_candidate_requires_exact_all_worker_mask(self):
        mask = list(range(48, 56))
        raw = dict(cpu_affinity=mask, thread_cpu_affinity={"1": mask, "2": mask},
                   worker_affinity_override={"target_cpus": mask})
        verify_cpu_policy(raw, "local8")
        with self.assertRaises(AssertionError):
            verify_cpu_policy(dict(raw, thread_cpu_affinity={"1": mask, "2": [1]}), "local8")


if __name__ == "__main__":
    unittest.main()
