// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <cstdint>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

void RunFloat16NarrowDP4AAccumulationTest(int64_t bits) {
  ConfigOptions config_options;
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableMatmulFp32Accumulation,
                                                 webgpu::options::kEnableMatmulFp32Accumulation_OFF));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  if (!provider) {
    GTEST_SKIP() << "WebGPU execution provider is unavailable.";
  }
  const auto& context = webgpu::WebGpuContextFactory::GetContext(provider->GetDeviceId());
  if (!context.DeviceHasFeature(wgpu::FeatureName::ShaderF16)) {
    GTEST_SKIP() << "WebGPU adapter does not support shader-f16.";
  }
  // Match DP4A dispatch: Apple adapters deliberately use other MatMulNBits paths.
  if (!context.DeviceHasFeature(wgpu::FeatureName::Subgroups) ||
      context.AdapterInfo().vendor == std::string_view{"apple"}) {
    GTEST_SKIP() << "WebGPU DP4A path is unavailable.";
  }

  // N=32 excludes subgroup-matrix dispatch. M=4 and K=4096 select the narrow
  // K-parallel DP4A path, including its unconditional FP32 accumulation.
  constexpr int64_t M = 4, N = 32, K = 4096, block_size = 32;
  constexpr int64_t blocks = K / block_size;
  const int64_t values_per_byte = 8 / bits;
  const int64_t blob_size = block_size / values_per_byte;
  const int32_t zero_point = 1 << (bits - 1);
  InlinedVector<MLFloat16> a(M * K);
  InlinedVector<uint8_t> packed_weights(N * blocks * blob_size, 0);
  InlinedVector<MLFloat16> scales(N * blocks, MLFloat16(16.0f));
  InlinedVector<MLFloat16> expected(M * N);

  // +/-127 quantizes exactly with activation scale 1. Each 32-element block
  // contributes +/-65024, which fits FP16, but the first two K iterations sum
  // to +/-130048. Later blocks cancel that sum; one omitted negative weight
  // leaves exactly +/-2032. FP16 partial sums overflow before cancellation.
  for (int64_t m = 0; m < M; ++m) {
    const float sign = m % 2 == 0 ? 1.0f : -1.0f;
    for (int64_t k = 0; k < K; ++k) {
      a[m * K + k] = MLFloat16(sign * 127.0f);
    }
    for (int64_t n = 0; n < N; ++n) {
      expected[m * N + n] = MLFloat16(sign * 2032.0f);
    }
  }
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      const int32_t weight = k == K - 1 ? 0 : (k < K / 2 ? 1 : -1);
      const auto quantized = static_cast<uint8_t>(zero_point + weight);
      packed_weights[(n * K + k) / values_per_byte] |=
          static_cast<uint8_t>(quantized << ((k % values_per_byte) * bits));
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("bits", bits);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("accuracy_level", 4);
  test.AddInput<MLFloat16>("A", {M, K}, a.data(), a.size());
  test.AddInput<uint8_t>("B", {N, blocks, blob_size}, packed_weights.data(), packed_weights.size(), true);
  test.AddInput<MLFloat16>("scales", {N, blocks}, scales.data(), scales.size(), true);
  test.AddOutput<MLFloat16>("Y", {M, N}, expected.data(), expected.size());
  test.SetOutputAbsErr("Y", 0.0f);
  test.SetOutputRelErr("Y", 0.0f);

  SessionOptions options;
  options.use_per_session_threads = false;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}

}  // namespace

TEST(WebGpuMatMulNBitsTest, Float16_Int2_NarrowDP4AAccumulation) {
  RunFloat16NarrowDP4AAccumulationTest(2);
}

TEST(WebGpuMatMulNBitsTest, Float16_Int4_NarrowDP4AAccumulation) {
  RunFloat16NarrowDP4AAccumulationTest(4);
}

TEST(WebGpuMatMulNBitsTest, Float16_Int8_NarrowDP4AAccumulation) {
  RunFloat16NarrowDP4AAccumulationTest(8);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
