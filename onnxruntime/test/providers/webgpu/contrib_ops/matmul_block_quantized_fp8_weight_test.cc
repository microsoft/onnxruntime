// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/float8.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#if defined(USE_WEBGPU) && !defined(DISABLE_FLOAT8_TYPES)
namespace {
void RunFp8MatMul(const std::vector<int64_t>& a_dims, int64_t n, int64_t block_size,
                  bool with_a_scale, bool with_bias) {
  auto ep = DefaultWebGpuExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }

  const auto k = a_dims.back();
  int64_t m = 1;
  for (size_t i = 0; i + 1 < a_dims.size(); ++i) {
    m *= a_dims[i];
  }
  const auto blocks = (k + block_size - 1) / block_size;
  std::vector<MLFloat16> a;
  std::vector<Float8E4M3FN> b;
  std::vector<float> scales;
  std::vector<MLFloat16> biases;
  for (int64_t i = 0; i < m * k; ++i) {
    a.emplace_back(0.125f * static_cast<float>((i % 11) - 5));
  }
  for (int64_t i = 0; i < n; ++i) {
    biases.emplace_back(static_cast<float>((i % 5) - 2) * 0.25f);
    for (int64_t j = 0; j < blocks; ++j) {
      scales.push_back(0.125f * static_cast<float>(1 + (i + j) % 4));
    }
    for (int64_t j = 0; j < k; ++j) {
      b.emplace_back(0.25f * static_cast<float>(((i * 3 + j) % 9) - 4), true);
    }
  }

  std::vector<MLFloat16> expected;
  expected.reserve(m * n);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int64_t j = 0; j < k; ++j) {
        float activation = a[row * k + j].ToFloat();
        if (with_a_scale) {
          activation = MLFloat16(Float8E4M3FN(activation / 0.375f, true).ToFloat() * 0.375f).ToFloat();
        }
        const auto weight = MLFloat16(b[col * k + j].ToFloat() *
                                      scales[col * blocks + j / block_size])
                                .ToFloat();
        sum += activation * weight;
      }
      expected.emplace_back(MLFloat16(sum + (with_bias ? biases[col].ToFloat() : 0.0f)));
    }
  }

  std::vector<int64_t> output_dims(a_dims.begin(), a_dims.end());
  output_dims.back() = n;
  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute("block_size", block_size);
  test.AddInput<MLFloat16>("A", a_dims, a);
  test.AddInput<Float8E4M3FN>("B", {n, k}, b, /*is_initializer=*/true);
  test.AddInput<float>("b_scale", {n, blocks}, scales, /*is_initializer=*/true);
  if (with_a_scale) {
    test.AddInput<float>("a_scale", {}, {0.375f}, /*is_initializer=*/true);
  } else {
    test.AddOptionalInputEdge<float>();
  }
  if (with_bias) {
    test.AddInput<MLFloat16>("bias", {n}, biases);
  }
  test.AddOutput<MLFloat16>("Y", output_dims, expected);
  test.SetOutputTolerance(0.03f);
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(ep));
  test.Run(options, OpTester::ExpectResult::kExpectSuccess, {}, {}, nullptr, &providers);
}
}  // namespace

TEST(MatMulBlockQuantizedFp8WeightTest, DecodeRankOneOddBytes) {
  RunFp8MatMul({3}, 1, 2, false, false);
}

TEST(MatMulBlockQuantizedFp8WeightTest, DecodeWithBlockScalesAndBias) {
  RunFp8MatMul({1, 33}, 19, 16, false, true);
}

TEST(MatMulBlockQuantizedFp8WeightTest, ActivationQdqBatched) {
  RunFp8MatMul({2, 3, 33}, 19, 16, true, true);
}

TEST(MatMulBlockQuantizedFp8WeightTest, MatrixTailWithBlockScales) {
  RunFp8MatMul({9, 33}, 19, 16, false, true);
}

TEST(MatMulBlockQuantizedFp8WeightTest, MatrixWithActivationQdq) {
  RunFp8MatMul({8, 32}, 16, 32, true, false);
}

TEST(MatMulBlockQuantizedFp8WeightTest, SubnormalSignedWeightsAndBias) {
  auto ep = DefaultWebGpuExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  const std::vector<Float8E4M3FN> b{
      Float8E4M3FN(uint8_t{0x01}, Float8E4M3FN::FromBits()),
      Float8E4M3FN(uint8_t{0x81}, Float8E4M3FN::FromBits()),
      Float8E4M3FN(0.5625f, true),
      Float8E4M3FN(-1.0f, true)};
  float expected = 0.25f;
  for (const auto weight : b) {
    expected += weight.ToFloat();
  }
  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute("block_size", int64_t{4});
  test.AddInput<MLFloat16>("A", {1, 4}, std::vector<MLFloat16>(4, MLFloat16(1.0f)));
  test.AddInput<Float8E4M3FN>("B", {1, 4}, b, true);
  test.AddInput<float>("b_scale", {1, 1}, {1.0f});
  test.AddOptionalInputEdge<float>();
  test.AddInput<MLFloat16>("bias", {1}, {MLFloat16(0.25f)});
  test.AddOutput<MLFloat16>("Y", {1, 1}, {MLFloat16(expected)});
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(ep));
  test.Run(options, OpTester::ExpectResult::kExpectSuccess, {}, {}, nullptr, &providers);
}

TEST(MatMulBlockQuantizedFp8WeightTest, ActivationQdqTieAndSaturation) {
  auto ep = DefaultWebGpuExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute("block_size", int64_t{1});
  test.AddInput<MLFloat16>("A", {7, 1},
                           ToFloat16({0.3984375f, -0.3984375f, 0.4453125f, -0.4453125f,
                                      200.0f, -200.0f, 0.0f}));
  test.AddInput<Float8E4M3FN>("B", {1, 1}, {Float8E4M3FN(1.0f, true)}, true);
  test.AddInput<float>("b_scale", {1, 1}, {1.0f});
  test.AddInput<float>("a_scale", {}, {0.375f});
  test.AddOutput<MLFloat16>("Y", {7, 1},
                            ToFloat16({0.375f, -0.375f, 0.46875f, -0.46875f,
                                       168.0f, -168.0f, 0.0f}));
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(ep));
  test.Run(options, OpTester::ExpectResult::kExpectSuccess, {}, {}, nullptr, &providers);
}

TEST(MatMulBlockQuantizedFp8WeightTest, EmptyReductionWithBias) {
  RunFp8MatMul({2, 0}, 3, 16, false, true);
}

TEST(MatMulBlockQuantizedFp8WeightTest, EmptyOutput) {
  RunFp8MatMul({0, 16}, 3, 16, false, false);
}

TEST(MatMulBlockQuantizedFp8WeightTest, InvalidScaleShape) {
  auto ep = DefaultWebGpuExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute("block_size", int64_t{16});
  test.AddInput<MLFloat16>("A", {1, 17}, std::vector<MLFloat16>(17, MLFloat16(1.0f)));
  test.AddInput<Float8E4M3FN>("B", {1, 17}, std::vector<Float8E4M3FN>(17, Float8E4M3FN(1.0f, true)));
  test.AddInput<float>("b_scale", {1, 1}, {1.0f});
  test.AddOutput<MLFloat16>("Y", {1, 1}, {MLFloat16(17.0f)});
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(ep));
  test.Run(options, OpTester::ExpectResult::kExpectFailure, "b_scale must have shape", {}, nullptr, &providers);
}

#endif

}  // namespace test
}  // namespace onnxruntime
