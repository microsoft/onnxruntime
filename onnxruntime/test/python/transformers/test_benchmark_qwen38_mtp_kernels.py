# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import sys
import unittest.mock
from pathlib import Path


def load_benchmark():
    path = Path(__file__).with_name("benchmark_qwen38_mtp_kernels.py")
    spec = importlib.util.spec_from_file_location("benchmark_qwen38_mtp_kernels", path)
    module = importlib.util.module_from_spec(spec)
    with unittest.mock.patch.dict(sys.modules, {"cupy": unittest.mock.Mock()}):
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
        source = """
template <bool NativeDecode = false>
__device__ float DecodeE4M3Fn(unsigned char code) { return code; }
template <typename T, bool FusedSwiGlu, bool EnableBias>
__device__ void StorePair() {}
// NVFP4 schema weights
template <typename T, bool FusedSwiGlu, bool EnableBias, int KLanes = 8,
          bool ReuseExpertRows = false, bool NativeScale = false>
__global__ void MoeGemvFp4RawKPackedKernel() { if (NativeScale) { return; } }
bool Fp4MoeGemvSupportsNativeScale() { return cudaFuncGetAttributes(); }
bool Fp4MoeGemvKernelUsesNativeScale() { return false; }
template <typename T>
__device__ __forceinline__ float PackedLowToFloat(uint32_t word) { return word; }
template <typename T, bool FusedSwiGlu, bool EnableBias, int KLanes, int Warps, int Batch, int MinBlocks>
__global__ void MoeGemvFp4DecodeKernel() {}
template <typename T, bool FusedSwiGlu>
void LaunchMoeGemvFp4RawNPacked() {}
"""
        headers = {
            "common.h": "enum class ActivationType { Identity };\nstruct ActivationParams {\n  ActivationParams() {} };",
            "details.h": "template <typename AType, bool PairInterleaved = false>\nstruct Converter {\n};",
            "interleaved_numeric_conversion.h": "CUTLASS_DEVICE int fp4_e2m1x8_uninterleave(int value) { return value; }",
        }
        for element in ("half", "__nv_bfloat16"):
            for fused, bias, wide, reuse, decode in (
                (False, False, True, False, False),
                (True, True, False, False, False),
                (True, False, False, True, False),
                (True, False, False, False, True),
                (False, True, True, False, True),
            ):
                with self.subTest(element=element, fused=fused, bias=bias, wide=wide, reuse=reuse, decode=decode):
                    with (
                        unittest.mock.patch.object(Path, "read_text", lambda path: headers[path.name]),
                        unittest.mock.patch.object(
                            self.benchmark, "compile_device_code", return_value=object()
                        ) as compile_code,
                    ):
                        self.benchmark.compile_nvfp4(source, element, fused, bias, wide, reuse, decode=decode)
                    generated, _ = compile_code.call_args.args
                    self.assertIn("template <bool NativeDecode = false>", generated)
                    if decode:
                        self.assertIn("__global__ void MoeGemvFp4DecodeKernel", generated)
                    else:
                        self.assertIn("__global__ void MoeGemvFp4RawKPackedKernel", generated)
                    self.assertNotIn("bool Fp4MoeGemvSupportsNativeScale", generated)
                    self.assertNotIn("bool Fp4MoeGemvKernelUsesNativeScale", generated)
                    self.assertNotIn("cudaFuncGetAttributes", generated)


if __name__ == "__main__":
    unittest.main()
