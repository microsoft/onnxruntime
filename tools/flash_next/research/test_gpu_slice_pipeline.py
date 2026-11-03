import ast
from pathlib import Path
import unittest

from run_gpu_slice_pipeline import compare, percentile


ROOT = Path(__file__).parent


def sample(ttft, decode):
    return dict(ttft_seconds=ttft, decode_tps=decode, decode_seconds=255 / decode, output_tokens=256)


class SlicePipelineTests(unittest.TestCase):
    def test_ttft_gain_requires_more_than_one_percent(self):
        control = [sample(4, 82)] * 3
        self.assertTrue(compare([sample(3.95, 82)] * 3, control, control)["supported"])
        self.assertFalse(compare([sample(3.96, 82)] * 3, control, control)["supported"])

    def test_each_request_must_improve_ttft(self):
        control = [sample(4, 82)] * 3
        self.assertFalse(compare([sample(3, 82), sample(3, 82), sample(4.01, 82)],
                                 control, control)["supported"])

    def test_decode_regression_cannot_be_hidden_by_ttft(self):
        control = [sample(4, 82)] * 3
        self.assertTrue(compare([sample(3, 80.37)] * 3, control, control)["supported"])
        self.assertFalse(compare([sample(3, 80.35)] * 3, control, control)["supported"])

    def test_invalid_decode_denominator_rejects(self):
        control = [sample(4, 82)] * 3
        invalid = sample(3, 82)
        invalid["output_tokens"] = 255
        with self.assertRaises(RuntimeError):
            compare([invalid] * 3, control, control)

    def test_three_request_percentiles_use_linear_interpolation(self):
        self.assertEqual(percentile([3, 1, 2], 0.5), 2)
        self.assertAlmostEqual(percentile([3, 1, 2], 0.95), 2.9)

    def test_benchmark_keeps_verified_gpu_ready_request(self):
        def request(path):
            tree = ast.parse(path.read_text())
            return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_request"))

        self.assertEqual(request(ROOT / "benchmark_gpu_slice.py"),
                         request(ROOT / "benchmark_gpu_greedy_argmax_ready.py"))

    def test_copy_excludes_lazy_cpu_authoritative_buffers(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-gpu-greedy-argmax/build-v6-r2/ep_webgpu_interface.cpp").read_text()
        begin = source.index("dynamic_cast<WebGPUMemory*>(&source)")
        end = source.index("GraphExecutor::Execute(config, params);", begin)
        copy = source[begin:end]
        self.assertIn("source.size_in_bytes_ / 4 <= std::numeric_limits<uint32_t>::max()", copy)
        self.assertIn('"session.disable_cpu_ep_fallback"', copy)
        self.assertIn('TensorSpec(&start, Ort::TypeToTensorType<int64_t>', copy)


if __name__ == "__main__":
    unittest.main()
