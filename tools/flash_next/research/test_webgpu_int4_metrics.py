import ast
import copy
import gc
import resource
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from prepare_webgpu_int4_metrics import benchmark_source, guard_source
from report_webgpu_int4_metrics import metrics, percentile, table


class MetricsTests(unittest.TestCase):
    def test_actual_request_boundaries_and_unchanged_ids(self):
        source = ast.parse(benchmark_source())
        request = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "run_request")
        class Generator:
            def __init__(self, *args):
                self.ids = []
            def append_tokens(self, prompt):
                self.prompt = list(prompt)
            def is_done(self):
                return False
            def generate_next_token(self):
                self.ids.append(len(self.ids) + 5)
            def get_next_tokens(self):
                return self.ids[-1:]
            def get_sequence(self, index):
                return self.prompt + self.ids
        clock = Mock(side_effect=[10, 13, 14, 16, 19])
        environment = dict(time=SimpleNamespace(perf_counter=clock), resource=resource, gc=gc, token_hash=str)
        exec(compile(ast.Module(body=[request], type_ignores=[]), "<measured-request>", "exec"), environment)
        og = SimpleNamespace(GeneratorParams=lambda model: SimpleNamespace(set_search_options=Mock()),
                             Generator=Generator)
        result = environment["run_request"](og, None, SimpleNamespace(decode=str), [1, 2], 3, 1024)
        self.assertEqual(result["output_token_ids"], [5, 6, 7])
        self.assertEqual(result["prefill_seconds"], 3)
        self.assertEqual(result["ttft_seconds"], 4)
        self.assertEqual(result["decode_seconds"], 5)
        self.assertEqual(result["request_seconds"], 9)
        self.assertAlmostEqual(result["prefill_tps"], 2 / 3)
        self.assertAlmostEqual(result["decode_tps"], 2 / 5)

    def fixtures(self):
        samples = [dict(warmup=False, repetition=i, context=2048, input_tokens=2048,
                        output_tokens=3, output_token_ids=[1, 2, 3], prefill_seconds=i + 0.5,
                        ttft_seconds=i + 1, decode_seconds=2, request_seconds=i + 3, decode_tps=1)
                   for i in range(3)]
        warmup = dict(samples[0], warmup=True, repetition=-1, ttft_seconds=1000)
        return (dict(status="completed", samples=[warmup, *samples]),
                dict(status="completed", exit_code=0, gpu_memory_samples=4, peak_gpu_used_bytes=72 * 1024**3,
                     gpu_memory_sampling_interval_seconds=0.5, gpu_memory_scope="test"))

    def test_warmup_exclusion_quantiles_rates_and_gib(self):
        raw, guard = self.fixtures()
        row = metrics(raw, guard, 2048, 3)
        self.assertEqual(row["requests_done"], "3/3")
        self.assertEqual(row["ttft_p50_seconds"], 2)
        self.assertAlmostEqual(row["ttft_p95_seconds"], 2.9)
        self.assertAlmostEqual(row["prefill_tps"], 6144 / 4.5)
        self.assertEqual(row["decode_tps"], 1)
        self.assertEqual(row["peak_gpu_gib"], 72)
        self.assertIn("| 2K | 3/3 |", table([row]))

    def test_invalid_or_missing_measurements_fail_explicitly(self):
        raw, guard = self.fixtures()
        for field, value in (("prefill_seconds", float("nan")), ("prefill_seconds", 10),
                             ("request_seconds", 100), ("decode_tps", 99)):
            broken = copy.deepcopy(raw)
            broken["samples"][1][field] = value
            with self.assertRaises(ValueError):
                metrics(broken, guard, 2048, 3)
        with self.assertRaises(ValueError):
            metrics(raw, guard, 2048, 10)
        with self.assertRaises(ValueError):
            metrics(raw, dict(guard, gpu_memory_samples=1), 2048, 3)
        with self.assertRaises(ValueError):
            percentile([], 0.95)

    def test_generated_guard_uses_used_bytes_and_resets_after_idle(self):
        source = guard_source()
        tree = ast.parse(source)
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "used_gpu_bytes")
        import ctypes
        from nvml_monitor import MemoryInfo
        def query(handle, pointer):
            ctypes.cast(pointer, ctypes.POINTER(MemoryInfo)).contents.used = 12345
            return 0
        monitor = SimpleNamespace(_handle=lambda uuid: 1, _check=Mock(),
                                  library=SimpleNamespace(nvmlDeviceGetMemoryInfo=query))
        environment = dict(ctypes=ctypes, MemoryInfo=MemoryInfo)
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "<gpu-memory-helper>", "exec"), environment)
        self.assertEqual(environment["used_gpu_bytes"](monitor, "uuid"), 12345)
        monitor._check.assert_called_once_with(0, "nvmlDeviceGetMemoryInfo")
        self.assertLess(source.index('state["gpu_snapshot"]'), source.index('state["baseline_gpu_used_bytes"]'))
        self.assertIn('max(state["peak_gpu_used_bytes"]', source)
        self.assertIn("stop_owned()", source)

    def test_reference_subset_recomputes_its_own_summary(self):
        from run_webgpu_int4_metrics import reference_subset
        raw, guard = self.fixtures()
        raw["samples"][0] = dict(raw["samples"][1], warmup=True, repetition=-1)
        raw["summary"] = {"2048:configured": {"decode_tps": 99}}
        subset = reference_subset(raw, 2048)
        self.assertEqual(len(subset["samples"]), 4)
        self.assertEqual(subset["summary"]["2048:configured"]["decode_tps"], 1)
        self.assertEqual(raw["summary"]["2048:configured"]["decode_tps"], 99)


if __name__ == "__main__":
    unittest.main()
