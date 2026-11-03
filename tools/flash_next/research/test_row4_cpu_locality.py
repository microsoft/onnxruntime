import ast
import math
import unittest
from pathlib import Path
from unittest import mock

import cpu_local_windows
from qualify_row4_cpu_locality import compare_samples
from run_row4_cpu_locality import CPU_MASKS, evidence_prefix, placement_command
from qualify_quickgelu_mul_long_context import ROOT


def samples(tps, ttft=3.5):
    return [{"prompt_sha256": f"prompt{i}", "output_token_ids": [1, 2, i],
             "decode_tps": tps, "ttft_seconds": ttft} for i in range(3)]


class CpuLocalityTests(unittest.TestCase):
    def test_all_existing_workers_inherit_mask_and_failure_is_explicit(self):
        masks = {0: {48, 49}, 1: {48, 49}, 2: {1}, 3: {60}}
        def set_mask(tid, mask):
            masks[tid] = set(mask)
        with mock.patch.object(cpu_local_windows, "Path") as path, \
                mock.patch.object(cpu_local_windows.os, "sched_getaffinity", side_effect=lambda tid: masks[tid]), \
                mock.patch.object(cpu_local_windows.os, "sched_setaffinity", side_effect=set_mask):
            path.return_value.iterdir.return_value = [Path(str(tid)) for tid in (1, 2, 3)]
            result = cpu_local_windows.inherit_worker_affinity()
        self.assertEqual(result["previous_thread_masks"]["2"], [1])
        self.assertEqual(result["target_cpus"], [48, 49])
        self.assertTrue(all(masks[tid] == {48, 49} for tid in (1, 2, 3)))
        with mock.patch.object(cpu_local_windows, "Path") as path, \
                mock.patch.object(cpu_local_windows.os, "sched_getaffinity", side_effect=lambda tid: {48} if tid == 0 else {1}), \
                mock.patch.object(cpu_local_windows.os, "sched_setaffinity"):
            path.return_value.iterdir.return_value = [Path("2")]
            with self.assertRaisesRegex(RuntimeError, "did not inherit"):
                cpu_local_windows.inherit_worker_affinity()

    def test_affinity_capture_is_on_the_natural_standard_completion_path(self):
        tree = ast.parse((ROOT / "benchmark_webgpu_int4.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        standard_try = [node for node in ast.walk(main) if isinstance(node, ast.Try)
                        and any(isinstance(stmt, ast.Assign) and ast.unparse(stmt.targets[0]) == "result['summary']"
                                for stmt in node.body)]
        self.assertEqual(len(standard_try), 1)
        captures = [stmt for stmt in standard_try[0].body if isinstance(stmt, ast.If)
                    and ast.unparse(stmt.test) == "hasattr(os, 'sched_getaffinity')"]
        self.assertEqual(len(captures), 1)
        self.assertIn("result['cpu_affinity']", ast.unparse(captures[0]))
        self.assertIn("result['thread_cpu_affinity']", ast.unparse(captures[0]))

    def test_process_wrapper_preserves_command_and_baseline(self):
        command = ["bash", "-lc", "exec python3 benchmark.py"]
        self.assertEqual(placement_command(command, "baseline"), command)
        self.assertEqual(placement_command(command, "local"), ["taskset", "-c", "48-95", *command])
        self.assertEqual(placement_command(command, "local8"), ["taskset", "-c", "48-55", *command])
        self.assertEqual(command, ["bash", "-lc", "exec python3 benchmark.py"])
        with self.assertRaises(ValueError):
            placement_command(command, "invalid")
        self.assertEqual(CPU_MASKS["local"], list(range(48, 96)))
        self.assertEqual(CPU_MASKS["local8"], list(range(48, 56)))

    def test_controls_modes_contexts_and_trials_have_distinct_evidence(self):
        prefixes = {evidence_prefix(placement, context, full, trial)
                    for placement in CPU_MASKS for context in (2048, 8192, 32768)
                    for full in (False, True) for trial in (0, 1, 2)}
        self.assertEqual(len(prefixes), 54)

    def test_bracketed_gain_requires_all_three_requests_and_ttft_preservation(self):
        result = compare_samples(samples(90), samples(80), samples(82))
        self.assertTrue(result["supported"])
        self.assertAlmostEqual(result["matched_request_decode_ratio"], 90 / math.sqrt(80 * 82))
        self.assertFalse(compare_samples(samples(79), samples(80), samples(82))["supported"])
        self.assertFalse(compare_samples(samples(80), samples(80), samples(80))["supported"])
        mixed = samples(90)
        mixed[0]["decode_tps"] = 79
        self.assertFalse(compare_samples(mixed, samples(80), samples(82))["supported"])
        self.assertFalse(compare_samples(samples(90, 3.6), samples(80), samples(82))["supported"])

    def test_invalid_and_mismatching_measurements_fail_explicitly(self):
        for value in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(AssertionError):
                compare_samples(samples(value), samples(80), samples(82))
        mismatch = samples(90)
        mismatch[0]["output_token_ids"] = [999]
        with self.assertRaises(AssertionError):
            compare_samples(mismatch, samples(80), samples(82))
        with self.assertRaises(AssertionError):
            compare_samples(samples(90)[:2], samples(80), samples(82))


if __name__ == "__main__":
    unittest.main()
