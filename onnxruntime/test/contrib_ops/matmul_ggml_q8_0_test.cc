// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#if defined(USE_CUDA)
namespace onnxruntime::test {
namespace {

std::vector<uint8_t> PackQ8(int64_t n, int64_t blocks) {
  constexpr std::array<uint16_t, 8> scales{0x2c00, 0x3401, 0xb001, 0x0000,
                                           0x0001, 0x0400, 0x3080, 0x3555};
  std::vector<uint8_t> bytes(static_cast<size_t>(n * blocks * 34));
  for (int64_t block = 0; block < n * blocks; ++block) {
    auto* p = bytes.data() + block * 34;
    const auto d = scales[static_cast<size_t>(block) % scales.size()];
    p[0] = static_cast<uint8_t>(d);
    p[1] = static_cast<uint8_t>(d >> 8);
    for (int j = 0; j < 32; ++j) {
      p[2 + j] = static_cast<uint8_t>(block * 32 + j);
    }
  }
  return bytes;
}

template <typename T>
void RunQ8(const std::vector<int64_t>& shape, int64_t n, int64_t blocks, bool use_bias) {
  const int64_t m = std::accumulate(shape.begin(), shape.end() - 1, int64_t{1},
                                    std::multiplies<int64_t>{});
  const int64_t k = blocks * 32;
  const auto bytes = PackQ8(n, blocks);
  std::vector<T> a(static_cast<size_t>(m * k));
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = T(static_cast<float>(static_cast<int>(i % 31) - 15) / 32.0f);
  }
  std::vector<T> bias(static_cast<size_t>(n));
  for (int64_t col = 0; col < n; ++col) {
    bias[static_cast<size_t>(col)] = T(static_cast<float>(col % 7 - 3) / 16.0f);
  }
  std::vector<T> expected;
  expected.reserve(static_cast<size_t>(m * n));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float sum = 0;
      for (int64_t kk = 0; kk < k; ++kk) {
        const auto* p = bytes.data() + (col * blocks + kk / 32) * 34;
        const float d = MLFloat16::FromBits(static_cast<uint16_t>(p[0] | (p[1] << 8))).ToFloat();
        const int raw = p[2 + kk % 32];
        const int q = raw < 128 ? raw : raw - 256;
        const T weight(d * static_cast<float>(q));
        sum = std::fma(static_cast<float>(a[static_cast<size_t>(row * k + kk)]),
                       static_cast<float>(weight), sum);
      }
      const T rounded(sum);
      expected.emplace_back(use_bias ? static_cast<float>(rounded) + static_cast<float>(bias[static_cast<size_t>(col)])
                                     : static_cast<float>(rounded));
    }
  }
  OpTester test("MatMulGGMLQ8_0", 1, kMSDomain);
  test.AddInput<T>("A", shape, a);
  test.AddInput<uint8_t>("B", {n, blocks, 34}, bytes);
  if (use_bias) {
    test.AddInput<T>("bias", {n}, bias);
  }
  auto output_shape = shape;
  output_shape.back() = n;
  test.AddOutput<T>("Y", output_shape, expected);
  test.SetOutputTolerance(std::is_same_v<T, float> ? 0.0005f : 0.03125f,
                          std::is_same_v<T, float> ? 0.00002f : 0.008f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

}  // namespace

TEST(MatMulGGMLQ8_0, FloatSignedExtremaAndScales) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  RunQ8<float>({32}, 9, 1, true);
  RunQ8<float>({3, 5, 96}, 7, 3, false);
}

TEST(MatMulGGMLQ8_0, Float16DecodeAndPrefill) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  RunQ8<MLFloat16>({1, 64}, 9, 2, false);
  RunQ8<MLFloat16>({3, 7, 96}, 13, 3, true);
}

TEST(MatMulGGMLQ8_0, BFloat16DecodeAndPrefill) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  RunQ8<BFloat16>({1, 64}, 9, 2, true);
  RunQ8<BFloat16>({17, 96}, 13, 3, false);
}

TEST(MatMulGGMLQ8_0, EmptyDimensionsAndContractionAllTypes) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  const auto run = []<typename T>() {
    RunQ8<T>({0, 32}, 3, 1, true);
    RunQ8<T>({2, 32}, 0, 1, false);
    RunQ8<T>({3, 0}, 5, 0, true);
  };
  run.template operator()<float>();
  run.template operator()<MLFloat16>();
  run.template operator()<BFloat16>();
}

TEST(MatMulGGMLQ8_0, RoundsCompleteWeightNotScaleOrFinalOnly) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  // Non-BF16-representable source scale; all q values distinguish scale-only rounding.
  std::vector<uint8_t> bytes(8 * 34, 0);
  for (int col = 0; col < 8; ++col) {
    bytes[col * 34] = 0x55;
    bytes[col * 34 + 1] = 0x35;
    for (int j = 0; j < 32; ++j) bytes[col * 34 + 2 + j] = static_cast<uint8_t>(col * 32 + j);
  }
  const auto run = [&]<typename T>() {
    std::vector<T> a(32 * 32, T(0.0f));
    std::vector<T> expected(32 * 8);
    for (int j = 0; j < 32; ++j) {
      a[j * 32 + j] = T(1.0f);
      for (int col = 0; col < 8; ++col) {
        const int raw = col * 32 + j;
        expected[j * 8 + col] = T(0.333251953125f * static_cast<float>(raw < 128 ? raw : raw - 256));
      }
    }
    OpTester test("MatMulGGMLQ8_0", 1, kMSDomain);
    test.AddInput<T>("A", {32, 32}, a);
    test.AddInput<uint8_t>("B", {8, 1, 34}, bytes);
    test.AddOutput<T>("Y", {32, 8}, expected);
    test.SetOutputTolerance(0, 0);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  };
  run.template operator()<float>();
  run.template operator()<MLFloat16>();
  run.template operator()<BFloat16>();
}

TEST(MatMulGGMLQ8_0, WeightRoundingCancellationAndOutputBeforeBias) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  const auto run = []<typename T>(float half_ulp) {
    std::vector<uint8_t> bytes(34, 0);
    bytes[0] = 0x55;
    bytes[1] = 0x35;
    bytes[2] = 127;
    bytes[3] = 125;
    std::vector<T> a(32, T(0.0f));
    a[0] = T(125.0f);
    a[1] = T(-127.0f);
    const T expected(125.0f * static_cast<float>(T(127.0f * 0.333251953125f)) -
                     127.0f * static_cast<float>(T(125.0f * 0.333251953125f)));
    if constexpr (!std::is_same_v<T, float>) {
      ASSERT_NE(static_cast<float>(expected), 0.0f);
    }
    const auto check = [&](T result, bool bias) {
      OpTester test("MatMulGGMLQ8_0", 1, kMSDomain);
      test.AddInput<T>("A", {1, 32}, a);
      test.AddInput<uint8_t>("B", {1, 1, 34}, bytes);
      if (bias) test.AddInput<T>("bias", {1}, {T(-1.0f)});
      test.AddOutput<T>("Y", {1, 1}, {result});
      test.SetOutputTolerance(0, 0);
      std::vector<std::unique_ptr<IExecutionProvider>> providers;
      providers.push_back(DefaultCudaExecutionProvider());
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
    };
    check(expected, false);
    bytes.assign(34, 1);
    bytes[0] = 0;
    bytes[1] = 0x3c;  // Unit reconstructed weights.
    a.assign(32, T(0.0f));
    a[0] = T(1.0f);
    a[1] = T(half_ulp);
    check(T(0.0f), true);
  };
  run.template operator()<float>(0x1p-24f);
  run.template operator()<MLFloat16>(0x1p-11f);
  run.template operator()<BFloat16>(0x1p-8f);
}

TEST(MatMulGGMLQ8_0, RejectsInvalidShapesAndTypes) {
  for (const auto& shape : std::vector<std::vector<int64_t>>{{1, 1, 33}, {1, 34}}) {
    OpTester test("MatMulGGMLQ8_0", 1, kMSDomain);
    test.AddInput<float>("A", {1, 32}, std::vector<float>(32));
    const auto count = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>{});
    test.AddInput<uint8_t>("B", shape, std::vector<uint8_t>(static_cast<size_t>(count)));
    test.AddOutput<float>("Y", {1, 1}, {0});
    test.Run(OpTester::ExpectResult::kExpectFailure, "B must have shape");
  }
  OpTester test("MatMulGGMLQ8_0", 1, kMSDomain);
  test.AddInput<double>("A", {1, 32}, std::vector<double>(32));
  test.AddInput<uint8_t>("B", {1, 1, 34}, std::vector<uint8_t>(34));
  test.AddOutput<double>("Y", {1, 1}, {0});
  test.Run(OpTester::ExpectResult::kExpectFailure, "invalid");
}

TEST(MatMulGGMLQ8_0, RejectsBiasScalarAndContractionMismatch) {
  OpTester bias("MatMulGGMLQ8_0", 1, kMSDomain);
  bias.AddInput<float>("A", {1, 32}, std::vector<float>(32));
  bias.AddInput<uint8_t>("B", {1, 1, 34}, std::vector<uint8_t>(34));
  bias.AddInput<float>("bias", {2}, {0, 0});
  bias.AddOutput<float>("Y", {1, 1}, {0});
  bias.Run(OpTester::ExpectResult::kExpectFailure, "bias must have shape");
  OpTester scalar("MatMulGGMLQ8_0", 1, kMSDomain);
  scalar.AddInput<float>("A", {}, {0});
  scalar.AddInput<uint8_t>("B", {1, 1, 34}, std::vector<uint8_t>(34));
  scalar.AddOutput<float>("Y", {1}, {0});
  scalar.Run(OpTester::ExpectResult::kExpectFailure, "A must have rank");
  OpTester contraction("MatMulGGMLQ8_0", 1, kMSDomain);
  contraction.AddInput<float>("A", {1, 31}, std::vector<float>(31));
  contraction.AddInput<uint8_t>("B", {1, 1, 34}, std::vector<uint8_t>(34));
  contraction.AddOutput<float>("Y", {1, 1}, {0});
  contraction.Run(OpTester::ExpectResult::kExpectFailure, "A's last dimension");
}

TEST(MatMulGGMLQ8_0, RuntimeShapeValidation) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  OpTester test("MatMulGGMLQ8_0", 1, kMSDomain);
  test.AddShapeToTensorData(false);
  test.AddInput<float>("A", {1, 31}, std::vector<float>(31));
  test.AddInput<uint8_t>("B", {1, 1, 34}, std::vector<uint8_t>(34));
  test.AddOutput<float>("Y", {1, 1}, {0});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "A's last dimension", {}, nullptr, &providers);
}

TEST(MatMulGGMLQ8_0, NoCpuFallback) {
  OpTester test("MatMulGGMLQ8_0", 1, kMSDomain);
  test.AddInput<float>("A", {1, 32}, std::vector<float>(32));
  test.AddInput<uint8_t>("B", {1, 1, 34}, std::vector<uint8_t>(34));
  test.AddOutput<float>("Y", {1, 1}, {0});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "Could not find an implementation", {}, nullptr, &providers);
}

}  // namespace onnxruntime::test
#endif
