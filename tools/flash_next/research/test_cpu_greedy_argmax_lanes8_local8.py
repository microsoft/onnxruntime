import ast
import json
from pathlib import Path
import unittest

from prepare_cpu_greedy_argmax_lanes8_local8 import screen, standard


ROOT = Path(__file__).parent


class CpuGreedyLocal8Tests(unittest.TestCase):
    def test_generated_proof_sources(self):
        for kind, transform in (("standard", standard), ("screen", screen)):
            original = (ROOT / f"run_cpu_greedy_argmax_lanes8_{kind}.py").read_text()
            generated = (ROOT / f"run_cpu_greedy_argmax_lanes8_local8_{kind}.py").read_text()
            self.assertEqual(ast.dump(ast.parse(generated)), ast.dump(ast.parse(transform(original))))

    def test_matched_local8_commands_and_original_runtime_controls(self):
        from run_cpu_greedy_argmax_lanes8_local8_standard import DIRECTORY, FLAG, OLD_RUNTIME, RUNTIME, TEMPLATE, model_command
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (False, True):
                for mode in ("control", "lanes8"):
                    command = model_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                            ROOT / "cpu-argmax-local8-command-test", context, full, mode)
                    self.assertEqual(command[:3], ["taskset", "-c", "48-55"])
                    self.assertEqual(command[-1].count("--inherit-worker-cpu-affinity"), 1)
                    self.assertIn("--tokens 256 --repetitions 3", command[-1])
                    self.assertIn(f"--contexts {context}", command[-1])
                    self.assertIn(f"export {FLAG}={int(mode == 'lanes8')}; ", command[-1])
                    expected = OLD_RUNTIME if mode == "control" else str(RUNTIME).replace("/home/sayanshaw/repos", "/workspace")
                    self.assertIn(expected, command[-1])

    def test_worker_mask_exact_not_merely_contained(self):
        from run_cpu_greedy_argmax_lanes8_local8_standard import verify_workers
        cpus = list(range(48, 56))
        raw = dict(cpu_affinity=cpus, worker_affinity_override=dict(target_cpus=cpus),
                   thread_cpu_affinity={"1": cpus, "2": cpus})
        self.assertEqual(verify_workers(raw, "control")["individually_pinned_threads"], 0)
        raw["thread_cpu_affinity"]["2"] = [48]
        with self.assertRaises(AssertionError):
            verify_workers(raw, "control")


if __name__ == "__main__":
    unittest.main()
