// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS) && !defined(ORT_MINIMAL_BUILD) && \
    (!defined(ORT_USE_EP_API_ADAPTERS) || defined(ORT_WEBGPU_INTERNAL_TEST_LIBRARY_PATH))

#include "gtest/gtest.h"

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/contrib_ops/matmul_2bits_test_helper.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#if defined(ORT_USE_EP_API_ADAPTERS)
#include "test/providers/webgpu/test_plugin.h"
#else
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#endif

namespace onnxruntime::test {
namespace {

using TestOptions2Bits = MatMul2BitsTestParams;

template <typename T1>
void RunTest2Bits(const MatMul2BitsTestParams& opts) {
  SCOPED_TRACE(opts);
  ASSERT_FALSE(opts.has_typed_zero_point && !opts.has_zero_point);
  auto ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(ep, nullptr);
  OpTester test("MatMulNBits", 1, kMSDomain);
  PopulateMatMul2BitsTest<T1>(test, opts);
  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(session_options).ConfigEp(std::move(ep)).RunWithConfig();
}

void RunSharedWebGpu2BitsCases(MatMul2BitsTestGroup group, int64_t accuracy_level) {
  ForEachMatMul2BitsTestCase(group, accuracy_level, [](const MatMul2BitsTestParams& params) {
    RunTest2Bits<float>(params);
  });
}

}  // namespace

// Preserve the WebGPU executions formerly selected by the shared float tests.
TEST(MatMul2BitsWebGpu, Shared_Float32_2b_Accuracy0) {
  RunSharedWebGpu2BitsCases(MatMul2BitsTestGroup::Basic, 0);
}

TEST(MatMul2BitsWebGpu, Shared_Float32_2b_Accuracy4) {
  RunSharedWebGpu2BitsCases(MatMul2BitsTestGroup::Basic, 4);
}

TEST(MatMul2BitsWebGpu, Shared_Float32_2b_BlkLen64_Accuracy4) {
  RunSharedWebGpu2BitsCases(MatMul2BitsTestGroup::Block64, 4);
}

TEST(MatMul2BitsWebGpu, Shared_Float32_2b_BlkLen64_Accuracy0) {
  RunSharedWebGpu2BitsCases(MatMul2BitsTestGroup::Block64, 0);
}

TEST(MatMul2BitsWebGpu, Shared_Float32_2b_BlkLen128_Accuracy4) {
  RunSharedWebGpu2BitsCases(MatMul2BitsTestGroup::Block128, 4);
}

TEST(MatMul2BitsWebGpu, Shared_Float32_2b_BlkLen128_Accuracy0) {
  RunSharedWebGpu2BitsCases(MatMul2BitsTestGroup::Block128, 0);
}

TEST(MatMul2BitsWebGpu, Shared_Float32_2b_BlkLen32_Accuracy4) {
  RunSharedWebGpu2BitsCases(MatMul2BitsTestGroup::Block32, 4);
}

TEST(MatMul2BitsWebGpu, Shared_Float32_2b_BlkLen32_Accuracy0) {
  RunSharedWebGpu2BitsCases(MatMul2BitsTestGroup::Block32, 0);
}

namespace {

// Runs a 2-bit MatMulNBits test only on WebGPU EP with a scalar reference.
// The test quantizes random weights to 2 bits, dequantizes to compute
// expected output via matmul on CPU, then compares WebGPU output.
template <typename T>
void RunWebGpu2BitsTest(int64_t M, int64_t N, int64_t K, int64_t block_size,
                        bool has_zero_point, float abs_error = 0.1f, float rel_error = 0.02f) {
  TestOptions2Bits opts{};
  opts.M = M;
  opts.N = N;
  opts.K = K;
  opts.block_size = block_size;
  opts.has_zero_point = has_zero_point;
  opts.output_abs_error = abs_error;
  opts.output_rel_error = rel_error;

  RunTest2Bits<T>(opts);
}

}  // namespace

// WebGPU 2-bit tests: symmetric (no zero points)
TEST(MatMul2BitsWebGpu, Float32_Symmetric_Small) {
  RunWebGpu2BitsTest<float>(1, 32, 32, 16, false);
  RunWebGpu2BitsTest<float>(1, 32, 32, 32, false);
  RunWebGpu2BitsTest<float>(1, 32, 16, 16, false);
}

TEST(MatMul2BitsWebGpu, Float32_Symmetric_Medium) {
  RunWebGpu2BitsTest<float>(1, 288, 16, 16, false);
  RunWebGpu2BitsTest<float>(4, 32, 32, 16, false);
  RunWebGpu2BitsTest<float>(4, 288, 16, 16, false);
  RunWebGpu2BitsTest<float>(100, 32, 32, 16, false);
  RunWebGpu2BitsTest<float>(100, 288, 16, 16, false);
}

// WebGPU 2-bit tests: asymmetric (with zero points) — the primary accuracy concern
TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_Small) {
  RunWebGpu2BitsTest<float>(1, 1, 16, 16, true);
  RunWebGpu2BitsTest<float>(1, 2, 16, 16, true);
  RunWebGpu2BitsTest<float>(1, 32, 16, 16, true);
  RunWebGpu2BitsTest<float>(1, 32, 32, 16, true);
  RunWebGpu2BitsTest<float>(1, 32, 32, 32, true);
}

TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_Medium) {
  RunWebGpu2BitsTest<float>(1, 288, 16, 16, true);
  RunWebGpu2BitsTest<float>(4, 32, 32, 16, true);
  RunWebGpu2BitsTest<float>(4, 288, 16, 16, true);
  RunWebGpu2BitsTest<float>(100, 32, 32, 16, true);
  RunWebGpu2BitsTest<float>(100, 288, 16, 16, true);
}

TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_BlockSize32) {
  // blockSize=32 triggers the Intel Gen12 optimized path on matching hardware.
  RunWebGpu2BitsTest<float>(1, 32, 32, 32, true);
  RunWebGpu2BitsTest<float>(4, 32, 32, 32, true);
  RunWebGpu2BitsTest<float>(100, 32, 32, 32, true);
}

TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_BlockSize128) {
  RunWebGpu2BitsTest<float>(1, 32, 16, 128, true);
  RunWebGpu2BitsTest<float>(4, 32, 16, 128, true);
  RunWebGpu2BitsTest<float>(100, 32, 16, 128, true);
}

// BlockSize=64 tests — covers nBlocksPerCol not a multiple of 4 (padding edge case).
// These match configurations found in real 2-bit quantized transformer models.
TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_BlockSize64) {
  RunWebGpu2BitsTest<float>(1, 32, 64, 64, true);
  RunWebGpu2BitsTest<float>(1, 32, 128, 64, true);
  RunWebGpu2BitsTest<float>(1, 192, 384, 64, true, 0.3f, 0.05f);
  RunWebGpu2BitsTest<float>(1, 384, 1024, 64, true, 0.5f, 0.05f);
}

TEST(MatMul2BitsWebGpu, Float32_Symmetric_BlockSize64) {
  RunWebGpu2BitsTest<float>(1, 32, 64, 64, false);
  RunWebGpu2BitsTest<float>(1, 32, 128, 64, false);
  RunWebGpu2BitsTest<float>(1, 192, 384, 64, false, 0.3f, 0.05f);
}

// Larger K tests — exercises multi-word (multiple u32) extraction per block,
// verifying the Q2 nibble-spread and A-offset tracking across passes.
TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_LargerK) {
  RunWebGpu2BitsTest<float>(1, 32, 64, 32, true);
  RunWebGpu2BitsTest<float>(1, 32, 128, 32, true);
  RunWebGpu2BitsTest<float>(1, 32, 256, 32, true, 0.3f, 0.05f);
}

// DP4A path tests (accuracy_level=4) — exercises the 1024-entry LUT / dequantization
// path for 2-bit weights with zero_points.
// DP4A constraints: accuracy_level==4, block_size%32==0, K%128==0, N%16==0.
// Skipped when the adapter lacks Subgroups support or is Apple (Metal),
// because the DP4A kernel would silently fall back to the default path.
TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_DP4A) {
  // Ensure the WebGPU context is initialized so we can query adapter capabilities.
  auto ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(ep, nullptr);
#if defined(ORT_USE_EP_API_ADAPTERS)
  if (!GetWebGpuTestPlugin().Inspect(*ep).supports_dp4a) {
#else
  if (!contrib::webgpu::HasDP4ADeviceSupport(ep->GetDeviceId())) {
#endif
    GTEST_SKIP() << "DP4A requires Subgroups support on a non-Apple adapter";
  }

  TestOptions2Bits opts{};
  opts.accuracy_level = 4;
  opts.has_zero_point = true;
  opts.output_abs_error = 0.1f;
  opts.output_rel_error = 0.02f;

  // M=1, N=16, K=128, block_size=32 — minimal DP4A-eligible shape
  opts.M = 1;
  opts.N = 16;
  opts.K = 128;
  opts.block_size = 32;
  RunTest2Bits<float>(opts);

  // M=1, N=32, K=256, block_size=32 — larger K
  opts.M = 1;
  opts.N = 32;
  opts.K = 256;
  opts.block_size = 32;
  opts.output_abs_error = 0.3f;
  opts.output_rel_error = 0.05f;
  RunTest2Bits<float>(opts);

  // M=4 (rows), N=32, K=128, block_size=32
  opts.M = 4;
  opts.N = 32;
  opts.K = 128;
  opts.block_size = 32;
  opts.output_abs_error = 0.1f;
  opts.output_rel_error = 0.02f;
  RunTest2Bits<float>(opts);

  // M=1, N=16, K=128, block_size=128 — full-block
  opts.M = 1;
  opts.N = 16;
  opts.K = 128;
  opts.block_size = 128;
  RunTest2Bits<float>(opts);
}

}  // namespace onnxruntime::test

#endif  // WebGPU with contrib ops and native or private plugin test support
