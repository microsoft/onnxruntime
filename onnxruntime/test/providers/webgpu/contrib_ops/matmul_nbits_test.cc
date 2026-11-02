// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

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

template <typename T>
void RunPerChannelDP4ATest(int64_t bits, int64_t M, int64_t N, int64_t K, bool has_bias,
                           bool has_zero_points = false) {
  SCOPED_TRACE(::testing::Message() << "bits=" << bits << " M=" << M << " N=" << N << " K=" << K
                                    << " bias=" << has_bias << " zero_points=" << has_zero_points);
  auto provider = DefaultWebGpuExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "WebGPU execution provider is unavailable.";
  }
  const auto& context = webgpu::WebGpuContextFactory::GetContext(provider->GetDeviceId());
  if constexpr (std::is_same_v<T, MLFloat16>) {
    if (!context.DeviceHasFeature(wgpu::FeatureName::ShaderF16)) {
      GTEST_SKIP() << "WebGPU adapter does not support shader-f16.";
    }
  }
  if (!context.DeviceHasFeature(wgpu::FeatureName::Subgroups) ||
      context.AdapterInfo().vendor == std::string_view{"apple"}) {
    GTEST_SKIP() << "WebGPU DP4A path is unavailable.";
  }

  const int64_t values_per_byte = 8 / bits;
  const int32_t zero_point = 1 << (bits - 1);
  const int32_t q_max = zero_point - 1;
  InlinedVector<float> a(M * K);
  InlinedVector<int32_t> b_q(N * K);
  InlinedVector<uint8_t> packed_weights(N * K / values_per_byte, 0);
  InlinedVector<float> scales(N);
  InlinedVector<float> bias(N);
  InlinedVector<uint8_t> zero_points(N);

  // Deterministic, aperiodic values; the row maxima land on distinct activations so the
  // per-token quantization has no exact .5 ties except by coincidence.
  uint32_t seed = 12345u;
  auto next = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<float>((seed >> 8) & 0xFFFF) / 65535.0f;  // [0, 1]
  };
  for (int64_t m = 0; m < M; ++m) {
    const float row_scale = 0.25f + 2.0f * next();
    for (int64_t k = 0; k < K; ++k) {
      a[m * K + k] = (next() - 0.5f) * row_scale;
    }
  }
  for (int64_t n = 0; n < N; ++n) {
    scales[n] = (0.5f + next()) / static_cast<float>(q_max) / std::sqrt(static_cast<float>(K)) * 8.0f;
    bias[n] = has_bias ? (next() - 0.5f) : 0.0f;
    // Asymmetric weights: a per-column zero point near the midpoint; the 4-bit value sits in the low nibble.
    const int32_t column_zero_point =
        has_zero_points ? zero_point + static_cast<int32_t>(std::lround((next() * 2.0f - 1.0f) * bits)) : zero_point;
    zero_points[n] = static_cast<uint8_t>(column_zero_point);
    for (int64_t k = 0; k < K; ++k) {
      const int32_t stored =
          has_zero_points
              ? static_cast<int32_t>(std::lround(next() * static_cast<float>((1 << bits) - 1)))
              : static_cast<int32_t>(std::lround((next() * 2.0f - 1.0f) * static_cast<float>(q_max))) + zero_point;
      b_q[n * K + k] = stored - column_zero_point;
      packed_weights[(n * K + k) / values_per_byte] |=
          static_cast<uint8_t>(stored << ((k % values_per_byte) * bits));
    }
  }
  if constexpr (std::is_same_v<T, MLFloat16>) {
    // The kernel sees the f16-rounded tensors; quantize the reference from the same values.
    auto round_f16 = [](float v) { return MLFloat16(v).ToFloat(); };
    std::transform(a.begin(), a.end(), a.begin(), round_f16);
    std::transform(scales.begin(), scales.end(), scales.begin(), round_f16);
    std::transform(bias.begin(), bias.end(), bias.begin(), round_f16);
  }

  // Reference: int8 A (absmax / 127 per A block), exact int32 dot per block, f32 rescale per block.
  // Zero points disable the per-channel path, so that case uses the blockwise kernel's 128-element A blocks.
  const int64_t a_block = has_zero_points ? 128 : K;
  InlinedVector<float> expected(M * N);
  InlinedVector<int32_t> a_q(K);
  InlinedVector<float> scale_a(K / a_block);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t blk = 0; blk < K / a_block; ++blk) {
      float block_max = 0.0f;
      for (int64_t k = blk * a_block; k < (blk + 1) * a_block; ++k) {
        block_max = std::max(block_max, std::abs(a[m * K + k]));
      }
      scale_a[blk] = block_max / 127.0f;
      for (int64_t k = blk * a_block; k < (blk + 1) * a_block; ++k) {
        a_q[k] = block_max > 0.0f ? static_cast<int32_t>(std::nearbyint(a[m * K + k] / block_max * 127.0f)) : 0;
      }
    }
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t blk = 0; blk < K / a_block; ++blk) {
        int64_t acc = 0;
        for (int64_t k = blk * a_block; k < (blk + 1) * a_block; ++k) {
          acc += static_cast<int64_t>(a_q[k]) * b_q[n * K + k];
        }
        sum += static_cast<float>(acc) * scale_a[blk];
      }
      expected[m * N + n] = sum * scales[n] + bias[n];
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("bits", bits);
  test.AddAttribute<int64_t>("block_size", K);
  test.AddAttribute<int64_t>("accuracy_level", 4);
  auto add_t = [&](const char* name, const std::vector<int64_t>& dims, const InlinedVector<float>& values,
                   bool is_initializer) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      InlinedVector<MLFloat16> converted(values.size());
      std::transform(values.begin(), values.end(), converted.begin(), [](float v) { return MLFloat16(v); });
      test.AddInput<MLFloat16>(name, dims, converted.data(), converted.size(), is_initializer);
    } else {
      test.AddInput<float>(name, dims, values.data(), values.size(), is_initializer);
    }
  };
  add_t("A", {M, K}, a, false);
  test.AddInput<uint8_t>("B", {N, 1, K / values_per_byte}, packed_weights.data(), packed_weights.size(), true);
  add_t("scales", {N, 1}, scales, true);
  if (has_zero_points) {
    test.AddInput<uint8_t>("zero_points", {N, 1}, zero_points.data(), zero_points.size(), true);
  } else {
    test.AddOptionalInputEdge<uint8_t>();
  }
  test.AddOptionalInputEdge<int32_t>();  // g_idx
  if (has_bias) {
    add_t("bias", {N}, bias, true);
  } else {
    test.AddOptionalInputEdge<T>();
  }
  if constexpr (std::is_same_v<T, MLFloat16>) {
    InlinedVector<MLFloat16> expected_t(expected.size());
    std::transform(expected.begin(), expected.end(), expected_t.begin(), [](float v) { return MLFloat16(v); });
    test.AddOutput<MLFloat16>("Y", {M, N}, expected_t.data(), expected_t.size());
  } else {
    test.AddOutput<float>("Y", {M, N}, expected.data(), expected.size());
  }
  // A pack4x8snorm tie flips one activation by 1, i.e. up to |q_b| * scale_a * scale_b per output.
  test.SetOutputAbsErr("Y", 0.02f);
  test.SetOutputRelErr("Y", 0.02f);

  SessionOptions options;
  options.use_per_session_threads = false;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}

template <typename T>
void RunPerChannelDP4ATests(int64_t bits) {
  // Partial M and N tiles with the smallest K the DP4A path accepts.
  RunPerChannelDP4ATest<T>(bits, 100, 80, 128, false);
  // Prefill-sized K with bias; 577 rows make the 8-bit grouped tile order end in a short group.
  RunPerChannelDP4ATest<T>(bits, 577, 144, 3840, true);
  // block_size == K with zero points must fall back to the blockwise kernel.
  RunPerChannelDP4ATest<T>(bits, 100, 80, 256, false, true);
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

TEST(WebGpuMatMulNBitsTest, Float16_Int8_PerChannelDP4A) {
  RunPerChannelDP4ATests<MLFloat16>(8);
}

TEST(WebGpuMatMulNBitsTest, Float16_Int4_PerChannelDP4A) {
  RunPerChannelDP4ATests<MLFloat16>(4);
}

TEST(WebGpuMatMulNBitsTest, Float32_Int8_PerChannelDP4A) {
  RunPerChannelDP4ATests<float>(8);
}

TEST(WebGpuMatMulNBitsTest, Float32_Int4_PerChannelDP4A) {
  RunPerChannelDP4ATests<float>(4);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
