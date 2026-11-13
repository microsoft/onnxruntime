// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"

#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/common/span_utils.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

void RunMatMulNBitsTileTest(int64_t bits, int64_t M, int64_t N, int64_t K,
                            bool has_zero_points, bool has_bias, int64_t batch_count = 1) {
  SCOPED_TRACE(testing::Message() << "bits=" << bits << ", M=" << M << ", N=" << N << ", K=" << K
                                  << ", zero_points=" << has_zero_points << ", bias=" << has_bias);

  auto provider = DefaultWebGpuExecutionProvider();
  ASSERT_NE(provider, nullptr);
  const auto& context = webgpu::WebGpuContextFactory::GetContext(provider->GetDeviceId());
  if (!context.DeviceHasFeature(wgpu::FeatureName::ShaderF16)) {
    GTEST_SKIP() << "WebGPU adapter does not support shader-f16";
  }

  constexpr int64_t block_size = 32;
  const int64_t blocks = K / block_size;
  const int64_t values_per_byte = 8 / bits;
  const int64_t blob_size = block_size / values_per_byte;
  const int64_t zero_point_bytes = (blocks + values_per_byte - 1) / values_per_byte;
  const int32_t default_zero_point = 1 << (bits - 1);

  RandomValueGenerator random{1234};
  const auto a = FloatsToMLFloat16s(random.Gaussian<float>(AsSpan({batch_count * M, K}), 0.0f, 0.25f));
  const auto weights = random.Uniform<int32_t>(AsSpan({N, K}), 0, (1 << bits) - 1);
  const auto bias = FloatsToMLFloat16s(random.Uniform<float>(AsSpan({N}), -0.5f, 0.5f));
  InlinedVector<uint8_t> packed_weights(N * blocks * blob_size, 0);
  InlinedVector<uint8_t> packed_zero_points(N * zero_point_bytes, 0);
  InlinedVector<MLFloat16> scales(N * blocks);
  InlinedVector<float> dequantized_weights(N * K);

  // Construct the quantized inputs locally; the reference uses the unpacked values
  // so packing/layout errors are not shared between the kernel and the reference.
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t block = 0; block < blocks; ++block) {
      const int64_t scale_index = n * blocks + block;
      scales[scale_index] = MLFloat16(static_cast<float>(1 + scale_index % 4) /
                                      (bits == 4 ? 128.0f : 2048.0f));
      const int32_t zero_point = has_zero_points
                                     ? default_zero_point + static_cast<int32_t>(scale_index % 5) - 2
                                     : default_zero_point;
      packed_zero_points[n * zero_point_bytes + block / values_per_byte] |=
          static_cast<uint8_t>(zero_point << ((block % values_per_byte) * bits));
      for (int64_t k = block * block_size; k < (block + 1) * block_size; ++k) {
        const int64_t index = n * K + k;
        packed_weights[index / values_per_byte] |=
            static_cast<uint8_t>(weights[index] << ((k % values_per_byte) * bits));
        dequantized_weights[index] = static_cast<float>(weights[index] - zero_point) * scales[scale_index].ToFloat();
      }
    }
  }

  InlinedVector<MLFloat16> expected(batch_count * M * N);
  for (int64_t m = 0; m < batch_count * M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += a[m * K + k].ToFloat() * dequantized_weights[n * K + k];
      }
      expected[m * N + n] = MLFloat16(sum + (has_bias ? bias[n].ToFloat() : 0.0f));
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("bits", bits);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("accuracy_level", 4);
  const TensorShapeVector a_shape = batch_count == 1 ? TensorShapeVector{M, K}
                                                     : TensorShapeVector{batch_count, M, K};
  const TensorShapeVector y_shape = batch_count == 1 ? TensorShapeVector{M, N}
                                                     : TensorShapeVector{batch_count, M, N};
  test.AddInput<MLFloat16>("A", a_shape, a);
  test.AddInput<uint8_t>("B", {N, blocks, blob_size}, packed_weights.data(), packed_weights.size(), true);
  test.AddInput<MLFloat16>("scales", {N, blocks}, scales.data(), scales.size(), true);
  if (has_zero_points) {
    test.AddInput<uint8_t>("zero_points", {N, zero_point_bytes}, packed_zero_points.data(), packed_zero_points.size(), true);
  } else {
    test.AddOptionalInputEdge<uint8_t>();
  }
  test.AddOptionalInputEdge<int32_t>();  // g_idx
  if (has_bias) {
    test.AddInput<MLFloat16>("bias", {N}, bias, true);
  }
  test.AddOutput<MLFloat16>("Y", y_shape, expected.data(), expected.size());
  test.SetOutputAbsErr("Y", 0.1f);
  test.SetOutputRelErr("Y", 0.02f);

  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}

void RunMatMulNBitsTileTests(int64_t bits) {
  for (bool has_zero_points : {false, true}) {
    for (bool has_bias : {false, true}) {
      // AMD D3D12 wave64: small-M/BK=64, four-wave/BK=128 with an M tail,
      // and two-wave/BK=64. Other adapters validate their selected WebGPU path.
      // N=128 checks valid output from both column subgroups of the wide tile.
      RunMatMulNBitsTileTest(bits, 32, 128, 64, has_zero_points, has_bias);
      RunMatMulNBitsTileTest(bits, 400, 64, 128, has_zero_points, has_bias);
      RunMatMulNBitsTileTest(bits, 640, 64, 128, has_zero_points, has_bias);

      // N=64 needs only one column wave; N=192 retains the wide tile's partial N edge.
      RunMatMulNBitsTileTest(bits, 32, 64, 64, has_zero_points, has_bias);
      RunMatMulNBitsTileTest(bits, 32, 192, 64, has_zero_points, has_bias);

      // Initial BK=128 does not divide K: reduce it to 64 for K=192 and
      // to 32 for K=96 so no iteration reads beyond the prepacked K dimension.
      RunMatMulNBitsTileTest(bits, 400, 64, 192, has_zero_points, has_bias);
      RunMatMulNBitsTileTest(bits, 400, 64, 96, has_zero_points, has_bias);

      // Two-wave BK=128 and wide BK=32, both with partial 16-row prepack blocks.
      RunMatMulNBitsTileTest(bits, 257, 64, 128, has_zero_points, has_bias);
      RunMatMulNBitsTileTest(bits, 33, 128, 32, has_zero_points, has_bias);

      // The single-wave tile must also handle reduced BK with a partial prepack block.
      RunMatMulNBitsTileTest(bits, 33, 64, 32, has_zero_points, has_bias);
    }
  }
}

template <typename T = MLFloat16>
void RunSmallMSimtTest(const std::vector<int64_t>& a_shape, int64_t N, bool expect_simt,
                       bool has_zero_points = false, bool has_bias = false, int64_t block_size = 32, int64_t bits = 4,
                       bool random_values = false) {
  const int64_t K = a_shape.back();
  const int64_t rows = TensorShape(a_shape).Size() / K;
  SCOPED_TRACE(testing::Message() << "A=" << TensorShape(a_shape) << ", N=" << N
                                  << ", block_size=" << block_size << ", bits=" << bits
                                  << ", type=" << DataTypeImpl::GetType<T>());
  auto provider = DefaultWebGpuExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "WebGPU EP is unavailable";
  }
  const auto& context = webgpu::WebGpuContextFactory::GetContext(provider->GetDeviceId());
  if (!context.DeviceHasFeature(wgpu::FeatureName::ShaderF16)) {
    GTEST_SKIP() << "WebGPU adapter does not support shader-f16";
  }

  const int64_t blocks = (K + block_size - 1) / block_size;
  const int64_t values_per_byte = 8 / bits;
  const int64_t blob_size = block_size / values_per_byte;
  const int64_t zero_bytes = (blocks + values_per_byte - 1) / values_per_byte;
  const int default_zero = 1 << (bits - 1);
  InlinedVector<T> a(rows * K);
  InlinedVector<uint8_t> b(N * blocks * blob_size);
  InlinedVector<T> scales(N * blocks);
  InlinedVector<uint8_t> zeros(N * zero_bytes, 0);
  InlinedVector<T> bias(N);
  for (int64_t i = 0; i < rows * K; ++i) {
    a[i] = T(static_cast<float>(i % 17 - 8) / 16.0f);
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<uint8_t>((i * 29 + 7) % 256);
  }
  for (int64_t col = 0; col < N; ++col) {
    bias[col] = T(static_cast<float>(col % 7 - 3) / 16.0f);
    for (int64_t block = 0; block < blocks; ++block) {
      scales[col * blocks + block] = T(static_cast<float>(1 + block % 3) / (bits == 8 ? 2048.0f : 128.0f));
      const int zero = default_zero + static_cast<int>((col + block) % 3) - 1;
      zeros[col * zero_bytes + block / values_per_byte] |=
          static_cast<uint8_t>(zero << (bits * (block % values_per_byte)));
    }
  }
  if (random_values) {
    RandomValueGenerator random{2814};
    const auto values = random.Gaussian<float>(AsSpan({rows, K}), 0.0f, 0.25f);
    const auto scale_values = random.Uniform<float>(AsSpan({N, blocks}), 0.001f, 0.017f);
    std::transform(values.begin(), values.end(), a.begin(), [](float v) { return T(v); });
    std::transform(scale_values.begin(), scale_values.end(), scales.begin(), [](float v) { return T(v); });
  }
  // Match weight dequantization precision, then independently decode and accumulate in FP64.
  InlinedVector<T> expected(rows * N);
  for (int64_t row = 0; row < rows; ++row) {
    // The activation pattern repeats every 17 rows, including in the large-batch case.
    if (!random_values && row >= 17) {
      std::copy_n(expected.begin() + (row % 17) * N, N, expected.begin() + row * N);
      continue;
    }
    for (int64_t col = 0; col < N; ++col) {
      double sum = has_bias ? static_cast<float>(bias[col]) : 0.0;
      for (int64_t inner = 0; inner < K; ++inner) {
        const auto packed = b[(col * blocks + inner / block_size) * blob_size +
                              (inner % block_size) / values_per_byte];
        const int weight = (packed >> (bits * (inner % values_per_byte))) & ((1 << bits) - 1);
        const int zero = has_zero_points
                             ? (zeros[col * zero_bytes + (inner / block_size) / values_per_byte] >>
                                (bits * ((inner / block_size) % values_per_byte))) &
                                   ((1 << bits) - 1)
                             : default_zero;
        const T dequantized(static_cast<float>(weight - zero) *
                            static_cast<float>(scales[col * blocks + inner / block_size]));
        sum += static_cast<double>(static_cast<float>(a[row * K + inner])) * static_cast<float>(dequantized);
      }
      expected[row * N + col] = T(static_cast<float>(sum));
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("bits", bits);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("accuracy_level", 4);
  test.AddInput<T>("A", a_shape, a.data(), a.size());
  test.AddInput<uint8_t>("B", {N, blocks, blob_size}, b.data(), b.size(), true);
  test.AddInput<T>("scales", {N, blocks}, scales.data(), scales.size(), true);
  if (has_zero_points) {
    test.AddInput<uint8_t>("zero_points", {N, zero_bytes}, zeros.data(), zeros.size(), true);
  } else {
    test.AddOptionalInputEdge<uint8_t>();
  }
  test.AddOptionalInputEdge<int32_t>();
  if (has_bias) {
    test.AddInput<T>("bias", {N}, bias.data(), bias.size(), true);
  }
  auto output_shape = a_shape;
  output_shape.back() = N;
  test.AddOutput<T>("Y", output_shape, expected.data(), expected.size());
  test.SetOutputTolerance(expect_simt ? (random_values ? 0.002f : 0.0f) : 0.1f, 0.0f);

  const auto severity = logging::LoggingManager::DefaultLogger().GetSeverity();
  logging::LoggingManager::SetDefaultLoggerSeverity(logging::Severity::kINFO);
  auto restore_severity = gsl::finally([severity] {
    logging::LoggingManager::SetDefaultLoggerSeverity(severity);
  });
  SessionOptions options;
  options.session_log_severity_level = static_cast<int>(logging::Severity::kINFO);
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  testing::internal::CaptureStderr();
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
  const auto log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("Starting program \""), std::string::npos) << "No shader dispatch was observed.";
  EXPECT_EQ(log.find("Starting program \"MatMulNBitsSmallMSimt") != std::string::npos, expect_simt) << log;
}

}  // namespace

TEST(WebGpuMatMulNBitsTest, SmallMSimtRowsAndOptionalInputs) {
  for (int64_t M = 1; M <= 9; ++M) {
    RunSmallMSimtTest({M, 4096}, 4096, M >= 2 && M <= 8, M % 2 == 0, M % 3 == 0);
  }
}

TEST(WebGpuMatMulNBitsTest, SmallMSimtThresholdsAndTails) {
  RunSmallMSimtTest({4, 4095}, 4096, false);
  RunSmallMSimtTest({4, 4096}, 4095, false);
  RunSmallMSimtTest({3, 4097}, 4101, true, true, true);
  RunSmallMSimtTest({5, 4128}, 4097, true, true, true);
  RunSmallMSimtTest({4, 4096}, 4096, false, true, true, 64);
}

TEST(WebGpuMatMulNBitsTest, SmallMSimtNonDyadicScales) {
  RunSmallMSimtTest({6, 4128}, 4101, true, true, true, 32, 4, true);
}

TEST(WebGpuMatMulNBitsTest, SmallMSimtBatchesAreNotMatrixRows) {
  RunSmallMSimtTest({3, 4, 4096}, 4096, true, true, true);
  RunSmallMSimtTest({2, 8, 4096}, 4096, true);
  RunSmallMSimtTest({2, 2, 3, 4096}, 4096, true, true);
  RunSmallMSimtTest({4, 1, 4096}, 4096, false);
  RunSmallMSimtTest({2, 9, 4096}, 4096, false);
  RunSmallMSimtTest({129, 2, 4096}, 4096, true);
}

TEST(WebGpuMatMulNBitsTest, SmallMSimtNarrowProjectionFallback) {
  for (int64_t M : {2, 4, 8}) {
    RunSmallMSimtTest({M, 5120}, 1024, false, true, true);
  }
}

TEST(WebGpuMatMulNBitsTest, SmallMSimtOtherQuantizationFallbacks) {
  RunSmallMSimtTest<float>({4, 4096}, 4096, false, true, true);
  RunSmallMSimtTest({4, 4096}, 4096, false, true, true, 32, 2);
  RunSmallMSimtTest({4, 4096}, 4096, false, true, true, 32, 8);
}

TEST(WebGpuMatMulNBitsTest, Float16_4Bits_SubgroupMatrixTiles) {
  RunMatMulNBitsTileTests(4);
}

TEST(WebGpuMatMulNBitsTest, Float16_8Bits_SubgroupMatrixTiles) {
  RunMatMulNBitsTileTests(8);
}

TEST(WebGpuMatMulNBitsTest, DP4AQualcommWave128Tiles) {
  auto provider = DefaultWebGpuExecutionProvider();
  ASSERT_NE(provider, nullptr);
  const auto& context = webgpu::WebGpuContextFactory::GetContext(provider->GetDeviceId());
  if (context.AdapterInfo().vendor != std::string_view{"qualcomm"} ||
      !context.DeviceHasFeature(wgpu::FeatureName::SubgroupSizeControl) ||
      context.AdapterInfo().subgroupMinSize > 128 || context.AdapterInfo().subgroupMaxSize < 128) {
    GTEST_SKIP() << "Requires Qualcomm 128-lane subgroup size control";
  }
  for (bool has_zero_points : {false, true}) {
    for (bool has_bias : {false, true}) {
      // Exercise both column subgroups and partially filled M and N tiles.
      RunMatMulNBitsTileTest(4, 128, 32, 128, has_zero_points, has_bias);
      RunMatMulNBitsTileTest(4, 129, 48, 256, has_zero_points, has_bias, 2);
      RunMatMulNBitsTileTest(4, 257, 16, 128, has_zero_points, has_bias);
    }
  }
}

TEST(WebGpuMatMulNBitsTest, DP4AActivationScaleReuse) {
  auto provider = DefaultWebGpuExecutionProvider();
  ASSERT_NE(provider, nullptr);
  const auto& context = webgpu::WebGpuContextFactory::GetContext(provider->GetDeviceId());
  if (!context.DeviceHasFeature(wgpu::FeatureName::Subgroups) ||
      context.AdapterInfo().vendor == std::string_view{"apple"}) {
    GTEST_SKIP() << "Requires the DP4A path";
  }

  constexpr int64_t batch_count = 2;
  constexpr int64_t N = 48;
  constexpr int64_t K = 384;
  constexpr int64_t block_size = 32;
  for (int64_t M : {65, 129}) {
    for (int64_t bits : {4, 8}) {
      SCOPED_TRACE(testing::Message() << "M=" << M << ", bits=" << bits);
      InlinedVector<float> a(batch_count * M * K);
      InlinedVector<float> expected(batch_count * M * N);
      for (int64_t batch = 0; batch < batch_count; ++batch) {
        for (int64_t row = 0; row < M; ++row) {
          for (int64_t k = 0; k < K; ++k) {
            // Powers of two give distinct scales across blocks, rows, and batches.
            a[(batch * M + row) * K + k] = static_cast<float>(1 << ((batch * 3 + row % 3 + (k / 128) * 2) % 7));
          }
          for (int64_t n = 0; n < N; ++n) {
            const int64_t k = (n % (K / block_size)) * block_size + block_size - 1;
            expected[(batch * M + row) * N + n] = a[(batch * M + row) * K + k];
          }
        }
      }

      // Each column selects the last activation of one K tile, including all
      // four tiles sharing a scale and the transitions to the next scale.
      const int64_t values_per_byte = 8 / bits;
      InlinedVector<uint8_t> weights(N * K / values_per_byte, bits == 4 ? 0x88 : 0x80);
      for (int64_t n = 0; n < N; ++n) {
        const int64_t k = (n % (K / block_size)) * block_size + block_size - 1;
        weights[(n * K + k) / values_per_byte] |= static_cast<uint8_t>(1 << ((k % values_per_byte) * bits));
      }
      InlinedVector<float> scales(N * K / block_size, 1.0f);
      OpTester test("MatMulNBits", 1, kMSDomain);
      test.AddAttribute<int64_t>("K", K);
      test.AddAttribute<int64_t>("N", N);
      test.AddAttribute<int64_t>("bits", bits);
      test.AddAttribute<int64_t>("block_size", block_size);
      test.AddAttribute<int64_t>("accuracy_level", 4);
      test.AddInput<float>("A", {batch_count, M, K}, a.data(), a.size());
      test.AddInput<uint8_t>("B", {N, K / block_size, block_size / values_per_byte}, weights.data(), weights.size(), true);
      test.AddInput<float>("scales", {N, K / block_size}, scales.data(), scales.size(), true);
      test.AddOutput<float>("Y", {batch_count, M, N}, expected.data(), expected.size());
      test.SetOutputAbsErr("Y", 0.0001f);
      test.SetOutputRelErr("Y", 0.00001f);
      SessionOptions options;
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
      test.Config(options).ConfigEp(DefaultWebGpuExecutionProvider()).RunWithConfig();
    }
  }
}

TEST(WebGpuMatMulNBitsTest, DP4AQuantizationLargeSubgroups) {
  auto provider = DefaultWebGpuExecutionProvider();
  ASSERT_NE(provider, nullptr);
  const auto& context = webgpu::WebGpuContextFactory::GetContext(provider->GetDeviceId());
  if (!context.DeviceHasFeature(wgpu::FeatureName::ShaderF16) ||
      !context.DeviceHasFeature(wgpu::FeatureName::Subgroups) ||
      context.AdapterInfo().subgroupMinSize <= 32) {
    GTEST_SKIP() << "Requires subgroups wider than a 128-value quantization block";
  }

  constexpr int64_t M = 5;
  constexpr int64_t N = 16;
  for (int64_t K : {128, 256}) {
    SCOPED_TRACE(K);
    InlinedVector<MLFloat16> a(M * K, MLFloat16(0.0f));
    InlinedVector<MLFloat16> expected(M * N);
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t block = 0; block < K / 128; ++block) {
        const bool large = (m * (K / 128) + block) % 2 != 0;
        a[m * K + block * 128] = MLFloat16(large ? 64.0f : 0.5f);
        a[m * K + block * 128 + 127] = MLFloat16(large ? 127.0f : 1.0f);
      }
      for (int64_t n = 0; n < N; ++n) {
        expected[m * N + n] = a[m * K];
      }
    }
    // Select the first activation. A reduction across the whole subgroup would
    // mix the small and large blocks and quantize 0.5 to 1 instead of about 0.504.
    InlinedVector<uint8_t> weights(N * K / 2, 0x88);
    for (int64_t n = 0; n < N; ++n) {
      weights[n * K / 2] = 0x89;
    }
    InlinedVector<MLFloat16> scales(N * K / 32, MLFloat16(1.0f));
    OpTester test("MatMulNBits", 1, kMSDomain);
    test.AddAttribute<int64_t>("K", K);
    test.AddAttribute<int64_t>("N", N);
    test.AddAttribute<int64_t>("bits", 4);
    test.AddAttribute<int64_t>("block_size", 32);
    test.AddAttribute<int64_t>("accuracy_level", 4);
    test.AddInput<MLFloat16>("A", {M, K}, a.data(), a.size());
    test.AddInput<uint8_t>("B", {N, K / 32, 16}, weights.data(), weights.size(), true);
    test.AddInput<MLFloat16>("scales", {N, K / 32}, scales.data(), scales.size(), true);
    test.AddOutput<MLFloat16>("Y", {M, N}, expected.data(), expected.size());
    test.SetOutputAbsErr("Y", 0.01f);
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    test.Config(options).ConfigEp(DefaultWebGpuExecutionProvider()).RunWithConfig();
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
