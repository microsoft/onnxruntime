import json
import unittest

from run_genai_allocator_workers import OLD_RUNTIME, ROOT, RUNTIME, TEMPLATE, verify_workers, worker_command
from run_qmoe_prefill_rows4_standard import DIRECTORY


class AllocatorWorkersTests(unittest.TestCase):
    def test_short_diagnostic_preserves_provider_and_affinity(self):
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for mode in ("control", "single"):
            with self.subTest(mode=mode):
                command = worker_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                         ROOT / f"allocator-test-{mode}", mode)
                self.assertIn(f"ORT_GENAI_WEBGPU_ALLOCATOR_SINGLE_THREAD={int(mode == 'single')}; ", command[-1])
                self.assertNotIn(OLD_RUNTIME, command[-1])
                self.assertIn(str(RUNTIME).replace("/home/sayanshaw/repos", "/workspace"), command[-1])
                self.assertIn("--tokens 36 --repetitions 1", command[-1])
                self.assertNotIn("--inherit-worker-cpu-affinity", command[-1])
                self.assertNotIn("--compare-", command[-1])
                self.assertIn("--webgpu-validation-mode full", command[-1])

    def test_exact_default_pins_and_removed_pins(self):
        default = {"cpu_affinity": list(range(96)), "thread_cpu_affinity":
                   {str(cpu): [cpu] for cpu in range(1, 96)}}
        default["thread_cpu_affinity"]["main"] = list(range(96))
        self.assertEqual(verify_workers(default, "control")["individually_pinned_threads"], 95)
        with self.assertRaises(AssertionError):
            verify_workers(default, "single")
        single = {"cpu_affinity": list(range(96)), "thread_cpu_affinity": {"main": list(range(96))}}
        self.assertEqual(verify_workers(single, "single")["individually_pinned_threads"], 0)
        with self.assertRaises(AssertionError):
            verify_workers(single, "control")

    def test_affinity_override_and_wrong_worker_mask_rejected(self):
        for raw in ({"cpu_affinity": list(range(96)), "thread_cpu_affinity": {"main": list(range(96))},
                     "worker_affinity_override": {}},
                    {"cpu_affinity": list(range(96)), "thread_cpu_affinity": {"main": [48, 49]}},
                    {"cpu_affinity": list(range(96)), "thread_cpu_affinity": {}}):
            with self.subTest(raw=raw), self.assertRaises(AssertionError):
                verify_workers(raw, "single")


if __name__ == "__main__":
    unittest.main()
