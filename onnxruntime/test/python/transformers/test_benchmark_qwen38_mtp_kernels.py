# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


def load_benchmark():
    path = Path(__file__).with_name("benchmark_qwen38_mtp_kernels.py")
    spec = importlib.util.spec_from_file_location("benchmark_qwen38_mtp_kernels", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"cupy": mock.Mock()}):
        spec.loader.exec_module(module)
    return module


class TestBenchmarkCudaSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = load_benchmark()

    def test_templated_device_function_preserves_declaration(self):
        source = "template <bool Native = false>\n__device__ float decode(int x) { return x; }\nvoid host() {}"
        actual = self.benchmark.device_function(source, "decode")
        self.assertTrue(actual.startswith("template <bool Native = false>"))
        self.assertNotIn("void host", actual)

    def test_non_template_device_function_stays_non_template(self):
        source = "template <typename T>\nvoid unrelated() {}\n__device__ float decode(int x) { return x; }"
        actual = self.benchmark.device_function(source, "decode")
        self.assertTrue(actual.startswith("__device__"))

    def test_global_function_stops_before_host_helpers(self):
        source = "template <typename T>\n__global__ void kernel(T x) { if (x) { x++; } }\nbool host() { return true; }"
        actual = self.benchmark.device_function(source, "kernel", "__global__")
        self.assertTrue(actual.startswith("template <typename T>"))
        self.assertNotIn("bool host", actual)

    def test_missing_function_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "Missing device function"):
            self.benchmark.device_function("", "missing")

    def test_nvfp4_generated_source_excludes_host_inspection(self):
        source = (self.benchmark.ROOT / "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.cu").read_text()
        for element in ("half", "__nv_bfloat16"):
            for fused, bias, wide, reuse in (
                (False, False, True, False),
                (True, True, False, False),
                (True, False, False, True),
            ):
                with self.subTest(element=element, fused=fused, bias=bias, wide=wide, reuse=reuse):
                    with mock.patch.object(
                        self.benchmark, "compile_device_code", return_value=object()
                    ) as compile_code:
                        self.benchmark.compile_nvfp4(source, element, fused, bias, wide, reuse)
                    generated, _ = compile_code.call_args.args
                    self.assertIn("template <bool NativeDecode = false>", generated)
                    self.assertIn("__global__ void MoeGemvFp4RawKPackedKernel", generated)
                    self.assertNotIn("bool Fp4MoeGemvSupportsNativeScale", generated)
                    self.assertNotIn("bool Fp4MoeGemvKernelUsesNativeScale", generated)
                    self.assertNotIn("cudaFuncGetAttributes", generated)


if __name__ == "__main__":
    unittest.main()
