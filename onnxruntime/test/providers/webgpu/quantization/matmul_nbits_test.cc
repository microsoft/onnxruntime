// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <cstdint>
#include <utility>

#include "gtest/gtest.h"

#include "core/common/inlined_containers.h"
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
                            bool has_zero_points, bool has_bias) {
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
  const auto a = FloatsToMLFloat16s(random.Gaussian<float>(AsSpan({M, K}), 0.0f, 0.25f));
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

  InlinedVector<MLFloat16> expected(M * N);
  for (int64_t m = 0; m < M; ++m) {
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
  test.AddInput<MLFloat16>("A", {M, K}, a);
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
  test.AddOutput<MLFloat16>("Y", {M, N}, expected.data(), expected.size());
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
      RunMatMulNBitsTileTest(bits, 32, 64, 64, has_zero_points, has_bias);
      RunMatMulNBitsTileTest(bits, 400, 64, 128, has_zero_points, has_bias);
      RunMatMulNBitsTileTest(bits, 640, 64, 128, has_zero_points, has_bias);
    }
  }
}

}  // namespace

TEST(WebGpuMatMulNBitsTest, Float16_4Bits_SubgroupMatrixTiles) {
  RunMatMulNBitsTileTests(4);
}

TEST(WebGpuMatMulNBitsTest, Float16_8Bits_SubgroupMatrixTiles) {
  RunMatMulNBitsTileTests(8);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
