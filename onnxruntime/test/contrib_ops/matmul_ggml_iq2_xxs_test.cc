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
#include "contrib_ops/cuda/quantization/iq2_xxs_grid.h"

namespace onnxruntime::test {
namespace {

// Scalar reference follows canonical dequantize_row_iq2_xxs, independently of CUDA lane mapping:
// ggml-org/llama.cpp@b76b13379d3b7d67ab0bc424516bb6fd2522bd8e,
// ggml/src/ggml-quants.c:2489-2514. See ContribOperators.md for the pinned URL.
std::vector<float> Decode(const std::vector<uint8_t>& bytes) {
  std::vector<float> values;
  values.reserve(bytes.size() / 66 * 256);
  for (size_t block = 0; block < bytes.size() / 66; ++block) {
    const auto* p = bytes.data() + block * 66;
    const float d = MLFloat16::FromBits(static_cast<uint16_t>(p[0] | (p[1] << 8))).ToFloat();
    for (int group = 0; group < 8; ++group) {
      const auto* record = p + 2 + group * 8;
      const uint32_t aux = record[4] | (uint32_t{record[5]} << 8) |
                           (uint32_t{record[6]} << 16) | (uint32_t{record[7]} << 24);
      const float scale = d * (0.5f + (aux >> 28)) * 0.25f;
      for (int vector = 0; vector < 4; ++vector) {
        unsigned sign_bits = (aux >> (7 * vector)) & 127;
        unsigned parity = 0;
        for (int bit = 0; bit < 7; ++bit) {
          parity ^= (sign_bits >> bit) & 1;
        }
        sign_bits |= parity << 7;
        const uint64_t grid = contrib::cuda::kIQ2XXSGrid[record[vector]];
        for (int j = 0; j < 8; ++j) {
          const float magnitude = static_cast<float>((grid >> (8 * j)) & 255);
          values.push_back(scale * magnitude * ((sign_bits & (1u << j)) ? -1.0f : 1.0f));
        }
      }
    }
  }
  return values;
}

std::vector<uint8_t> Pack(int64_t n, int64_t k_blocks) {
  constexpr std::array<uint16_t, 8> scale_bits{0x2c00, 0x3401, 0xb001, 0x0000,
                                               0x0001, 0x0400, 0x3080, 0x3555};
  std::vector<uint8_t> bytes(static_cast<size_t>(n * k_blocks * 66));
  for (int64_t block = 0; block < n * k_blocks; ++block) {
    auto* p = bytes.data() + block * 66;
    const auto d = scale_bits[static_cast<size_t>(block) % scale_bits.size()];
    p[0] = static_cast<uint8_t>(d);
    p[1] = static_cast<uint8_t>(d >> 8);
    for (int group = 0; group < 8; ++group) {
      auto* record = p + 2 + group * 8;
      uint32_t aux = static_cast<uint32_t>((block + group) % 16) << 28;
      for (int vector = 0; vector < 4; ++vector) {
        record[vector] = static_cast<uint8_t>(block * 32 + group * 4 + vector);
        aux |= static_cast<uint32_t>((block * 32 + group * 4 + vector) % 128) << (7 * vector);
      }
      for (int j = 0; j < 4; ++j) {
        record[4 + j] = static_cast<uint8_t>(aux >> (8 * j));
      }
    }
  }
  return bytes;
}

template <typename T>
void RunCase(const std::vector<int64_t>& shape, int64_t n, int64_t k_blocks, bool use_bias) {
  const int64_t m = std::accumulate(shape.begin(), shape.end() - 1, int64_t{1},
                                    std::multiplies<int64_t>{});
  const int64_t k = k_blocks * 256;
  const auto bytes = Pack(n, k_blocks);
  const auto weights = Decode(bytes);
  std::vector<T> a(static_cast<size_t>(m * k));
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = T(static_cast<float>(static_cast<int>(i % 31) - 15) / 32.0f);
  }
  std::vector<T> bias(static_cast<size_t>(n));
  for (int64_t col = 0; col < n; ++col) {
    bias[static_cast<size_t>(col)] = T(static_cast<float>(col % 7) / 16.0f);
  }
  std::vector<T> expected;
  expected.reserve(static_cast<size_t>(m * n));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float sum = 0;
      for (int64_t kk = 0; kk < k; ++kk) {
        sum = std::fma(static_cast<float>(a[static_cast<size_t>(row * k + kk)]),
                       static_cast<float>(T(weights[static_cast<size_t>(col * k + kk)])), sum);
      }
      const T rounded(sum);
      expected.emplace_back(use_bias ? static_cast<float>(rounded) + static_cast<float>(bias[static_cast<size_t>(col)])
                                     : static_cast<float>(rounded));
    }
  }
  OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  test.AddInput<T>("A", shape, a);
  test.AddInput<uint8_t>("B", {n, k_blocks, 66}, bytes);
  if (use_bias) {
    test.AddInput<T>("bias", {n}, bias);
  }
  auto output_shape = shape;
  output_shape.back() = n;
  test.AddOutput<T>("Y", output_shape, expected);
  // FP32 accumulation order differs between the scalar reference and warp tree reduction.
  test.SetOutputTolerance(std::is_same_v<T, float> ? 0.0005f : 0.03125f,
                          std::is_same_v<T, float> ? 0.00002f : 0.008f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

}  // namespace

TEST(MatMulGGMLIQ2XXS, DecodeFloatAllGridSignsScales) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  // 256 columns x two superblocks visit every grid and sign index and all local scales.
  RunCase<float>({1, 512}, 256, 2, false);
}

TEST(MatMulGGMLIQ2XXS, PrefillFloatOddLeadingDimensions) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunCase<float>({3, 5, 768}, 7, 3, true);
}

TEST(MatMulGGMLIQ2XXS, DecodeFloat16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunCase<MLFloat16>({512}, 9, 2, true);
}

TEST(MatMulGGMLIQ2XXS, PrefillFloat16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunCase<MLFloat16>({17, 512}, 13, 2, false);
}

TEST(MatMulGGMLIQ2XXS, DecodeBFloat16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunCase<BFloat16>({1, 512}, 9, 2, true);
}

TEST(MatMulGGMLIQ2XXS, PrefillBFloat16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunCase<BFloat16>({3, 7, 512}, 13, 2, true);
}

TEST(MatMulGGMLIQ2XXS, EmptyDimensionsAndEmptyContraction) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunCase<float>({0, 256}, 3, 1, false);
  RunCase<float>({2, 256}, 0, 1, false);
  RunCase<float>({3, 0}, 5, 0, true);
  RunCase<MLFloat16>({0, 256}, 3, 1, true);
  RunCase<MLFloat16>({2, 256}, 0, 1, false);
  RunCase<MLFloat16>({3, 0}, 5, 0, true);
  RunCase<BFloat16>({0, 256}, 3, 1, true);
  RunCase<BFloat16>({2, 256}, 0, 1, false);
  RunCase<BFloat16>({3, 0}, 5, 0, true);
}

TEST(MatMulGGMLIQ2XXS, EveryGridElementInCanonicalOrder) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  constexpr int64_t n = 256;
  std::vector<uint8_t> bytes(n * 66, 0);
  std::vector<float> a(8 * 256, 0);
  std::vector<float> expected(8 * n);
  for (int j = 0; j < 8; ++j) {
    a[j * 256 + j] = 1;
    for (int col = 0; col < n; ++col) {
      bytes[col * 66 + 1] = 0x48;  // d=8, local_scale=0 -> scale=1.
      bytes[col * 66 + 2] = static_cast<uint8_t>(col);
      expected[j * n + col] = static_cast<float>((contrib::cuda::kIQ2XXSGrid[col] >> (8 * j)) & 255);
    }
  }
  OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  test.AddInput<float>("A", {8, 256}, a);
  test.AddInput<uint8_t>("B", {n, 1, 66}, bytes);
  test.AddOutput<float>("Y", {8, n}, expected);
  test.SetOutputTolerance(0, 0);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(MatMulGGMLIQ2XXS, RoundsReconstructedWeightsBeforeMultiplication) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  std::vector<uint8_t> bytes(66, 0);
  bytes[0] = 0x55;
  bytes[1] = 0x35;  // d=0.333251953125
  bytes[2] = 14;    // First vector is [8,43,25,25,8,8,8,8].
  // The FP32 weights cancel exactly; native reconstructed weights must not.
  const auto run = [&]<typename T>() {
    std::vector<T> a(256, T(0.0f));
    a[1] = T(25.0f);
    a[2] = T(-43.0f);
    OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
    test.AddInput<T>("A", {1, 256}, a);
    test.AddInput<uint8_t>("B", {1, 1, 66}, bytes);
    const T expected(25.0f * static_cast<float>(T(43.0f * 0.333251953125f / 8.0f)) -
                     43.0f * static_cast<float>(T(25.0f * 0.333251953125f / 8.0f)));
    if constexpr (!std::is_same_v<T, float>) {
      ASSERT_NE(static_cast<float>(expected), 0.0f);
    }
    test.AddOutput<T>("Y", {1, 1}, {expected});
    test.SetOutputTolerance(0, 0);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  };
  run.template operator()<float>();
  run.template operator()<MLFloat16>();
  run.template operator()<BFloat16>();
}

TEST(MatMulGGMLIQ2XXS, RoundsMatMulOutputBeforeBias) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  // d=1 and grid0=8 produce exactly unit weights.
  std::vector<uint8_t> bytes(66, 0);
  bytes[1] = 0x3c;
  const auto run = [&]<typename T>(float half_ulp) {
    std::vector<T> a(256, T(0.0f));
    a[0] = T(1.0f);
    a[8] = T(half_ulp);
    OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
    test.AddInput<T>("A", {1, 256}, a);
    test.AddInput<uint8_t>("B", {1, 1, 66}, bytes);
    test.AddInput<T>("bias", {1}, {T(-1.0f)});
    test.AddOutput<T>("Y", {1, 1}, {T(0.0f)});
    test.SetOutputTolerance(0, 0);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  };
  run.template operator()<float>(0x1p-24f);
  run.template operator()<MLFloat16>(0x1p-11f);
  run.template operator()<BFloat16>(0x1p-8f);
}

TEST(MatMulGGMLIQ2XXS, RejectsInvalidShapes) {
  for (const auto& shape : std::vector<std::vector<int64_t>>{{1, 1, 65}, {1, 66}}) {
    OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
    test.AddInput<float>("A", {1, 256}, std::vector<float>(256));
    const auto count = std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                       std::multiplies<int64_t>{});
    test.AddInput<uint8_t>("B", shape, std::vector<uint8_t>(static_cast<size_t>(count)));
    test.AddOutput<float>("Y", {1, 1}, {0});
    test.Run(OpTester::ExpectResult::kExpectFailure, "B must have shape");
  }
  OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  test.AddInput<float>("A", {1, 255}, std::vector<float>(255));
  test.AddInput<uint8_t>("B", {1, 1, 66}, std::vector<uint8_t>(66));
  test.AddOutput<float>("Y", {1, 1}, {0});
  test.Run(OpTester::ExpectResult::kExpectFailure, "A's last dimension");
}

TEST(MatMulGGMLIQ2XXS, RejectsInvalidBiasAndScalarActivation) {
  OpTester bias_test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  bias_test.AddInput<float>("A", {1, 256}, std::vector<float>(256));
  bias_test.AddInput<uint8_t>("B", {1, 1, 66}, std::vector<uint8_t>(66));
  bias_test.AddInput<float>("bias", {2}, {0, 0});
  bias_test.AddOutput<float>("Y", {1, 1}, {0});
  bias_test.Run(OpTester::ExpectResult::kExpectFailure, "bias must have shape");
  OpTester scalar_test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  scalar_test.AddInput<float>("A", {}, {0});
  scalar_test.AddInput<uint8_t>("B", {1, 1, 66}, std::vector<uint8_t>(66));
  scalar_test.AddOutput<float>("Y", {1}, {0});
  scalar_test.Run(OpTester::ExpectResult::kExpectFailure, "A must have rank");
}

TEST(MatMulGGMLIQ2XXS, RuntimeShapeValidation) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  test.AddShapeToTensorData(false);
  test.AddInput<float>("A", {1, 255}, std::vector<float>(255));
  test.AddInput<uint8_t>("B", {1, 1, 66}, std::vector<uint8_t>(66));
  test.AddOutput<float>("Y", {1, 1}, {0});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "A's last dimension", {}, nullptr, &providers);
}

TEST(MatMulGGMLIQ2XXS, RejectsUnsupportedActivationType) {
  OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  test.AddInput<double>("A", {1, 256}, std::vector<double>(256));
  test.AddInput<uint8_t>("B", {1, 1, 66}, std::vector<uint8_t>(66));
  test.AddOutput<double>("Y", {1, 1}, {0});
  test.Run(OpTester::ExpectResult::kExpectFailure, "invalid");
}

TEST(MatMulGGMLIQ2XXS, NoCpuFallback) {
  OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  test.AddInput<float>("A", {1, 256}, std::vector<float>(256));
  test.AddInput<uint8_t>("B", {1, 1, 66}, std::vector<uint8_t>(66));
  test.AddOutput<float>("Y", {1, 1}, {0});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "Could not find an implementation",
           {}, nullptr, &providers);
}

}  // namespace onnxruntime::test
#endif
