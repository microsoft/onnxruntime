// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "core/platform/env.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#if defined(USE_CUDA)
#include <cuda.h>
#endif

#if defined(USE_CUDA) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
namespace onnxruntime::test {
namespace {

float DecodeScale(uint8_t code) {
  const int exponent = (code >> 3) & 15;
  const int fraction = code & 7;
  const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -9)
                                        : std::ldexp(1.0f + fraction / 8.0f, exponent - 7);
  return (code & 128) ? -magnitude : magnitude;
}

template <typename T>
void Check(const std::vector<int64_t>& shape, int64_t n, const std::vector<T>& a,
           const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales,
           float global, const std::vector<T>& expected) {
  const int64_t k = shape.back();
  OpTester test("MatMulNVFP4A16", 1, kMSDomain);
  test.AddInput<T>("A", shape, a);
  test.AddInput<uint8_t>("B", {n, k / 2}, packed);
  test.AddInput<uint8_t>("weight_scale", {n, k / 16}, scales);
  test.AddInput<float>("weight_scale_2", {}, {global});
  auto output_shape = shape;
  output_shape.back() = n;
  test.AddOutput<T>("Y", output_shape, expected);
  test.SetOutputTolerance(0, 0);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename T>
void RunCase(const std::vector<int64_t>& shape, int64_t n) {
  constexpr std::array<float, 8> magnitudes{0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  constexpr std::array<uint8_t, 8> scale_codes{0x3a, 0x34, 0x42, 0x2c, 0x3c, 0x44, 0x32, 0x36};
  constexpr float global = 0.33333f;
  const int64_t k = shape.back();
  const int64_t m = std::accumulate(shape.begin(), shape.end() - 1, int64_t{1},
                                    std::multiplies<int64_t>{});
  std::vector<uint8_t> packed(static_cast<size_t>(n * k / 2));
  std::vector<uint8_t> scales(static_cast<size_t>(n * k / 16));
  std::vector<T> local(static_cast<size_t>(n * k));
  for (size_t i = 0; i < scales.size(); ++i) scales[i] = scale_codes[i % scale_codes.size()];
  for (int64_t col = 0; col < n; ++col) {
    for (int64_t kk = 0; kk < k; ++kk) {
      const int code = static_cast<int>((col * 7 + kk) % 16);
      packed[static_cast<size_t>((col * k + kk) / 2)] |= static_cast<uint8_t>(code << (4 * (kk % 2)));
      local[static_cast<size_t>(col * k + kk)] = T(
          magnitudes[code & 7] * ((code & 8) ? -1.0f : 1.0f) *
          DecodeScale(scales[static_cast<size_t>(col * (k / 16) + kk / 16)]));
    }
  }
  std::vector<T> a(static_cast<size_t>(m * k));
  for (size_t i = 0; i < a.size(); ++i) a[i] = T(static_cast<float>(static_cast<int>(i % 31) - 15) / 32.0f);
  std::vector<T> expected;
  expected.reserve(static_cast<size_t>(m * n));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float sum = 0;
      for (int64_t kk = 0; kk < k; ++kk) {
        sum = std::fma(static_cast<float>(a[static_cast<size_t>(row * k + kk)]),
                       static_cast<float>(local[static_cast<size_t>(col * k + kk)]), sum);
      }
      expected.emplace_back(sum * global);
    }
  }
  Check(shape, n, a, packed, scales, global, expected);
}

template <typename T>
void CheckProductPrecision(int64_t k, float large_activation, float activation_ulp, bool overflow) {
  std::vector<T> a(static_cast<size_t>(k), T(0.0f));
  std::vector<uint8_t> packed(static_cast<size_t>(k / 2), 0);
  a[0] = T(overflow ? large_activation : 1.0f + activation_ulp);
  packed[0] = overflow ? 0x07 : 0x77;
  if (!overflow) a[1] = T(-1.0f);
  // E2M1 6 times E4M3 0.125 is exactly 0.75 in both activation types.
  // Multiplying A by 6 first can overflow or lose the residual before scaling.
  const T expected(0.75f * (overflow ? large_activation : activation_ulp));
  Check<T>({1, k}, 1, a, packed, std::vector<uint8_t>(static_cast<size_t>(k / 16), 0x20),
           1.0f, {expected});
}

}  // namespace

TEST(MatMulNVFP4A16, BFloat16DecodeAndGeneralPrefill) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  for (int64_t m : {1, 15, 65}) RunCase<BFloat16>({m, 128}, 7);
  RunCase<BFloat16>({1, 16}, 3);
  RunCase<BFloat16>({15, 16}, 3);
  RunCase<BFloat16>({3, 5, 128}, 9);
  RunCase<BFloat16>({128}, 8);
}

TEST(MatMulNVFP4A16, Float16DecodeAndGeneralPrefill) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  for (int64_t m : {1, 15, 65}) RunCase<MLFloat16>({m, 128}, 7);
  RunCase<MLFloat16>({1, 16}, 3);
  RunCase<MLFloat16>({15, 16}, 3);
  RunCase<MLFloat16>({3, 5, 128}, 9);
}

TEST(MatMulNVFP4A16, SmallKProductsAvoidActivationTypeOverflow) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  for (int64_t k : {32, 64, 96}) {
    CheckProductPrecision<MLFloat16>(k, 16384.0f, 0x1p-10f, true);
    CheckProductPrecision<BFloat16>(k, 0x1p126f, 0x1p-7f, true);
  }
}

TEST(MatMulNVFP4A16, SmallKProductsAvoidActivationTypeRounding) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  for (int64_t k : {32, 64, 96}) {
    CheckProductPrecision<MLFloat16>(k, 16384.0f, 0x1p-10f, false);
    CheckProductPrecision<BFloat16>(k, 0x1p126f, 0x1p-7f, false);
  }
}

TEST(MatMulNVFP4A16, ProductsStayFP32WhenMmaDisabled) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  // The selector is process-cached. Run this test in a fresh process with the
  // override already set, rather than mutating it after another test dispatched.
  if (Env::Default().GetEnvironmentVar("ORT_FP4_GEMV_MMA") != "0") {
    GTEST_SKIP() << "Run in a fresh process with ORT_FP4_GEMV_MMA=0.";
  }
  for (bool overflow : {false, true}) {
    CheckProductPrecision<MLFloat16>(128, 16384.0f, 0x1p-10f, overflow);
    CheckProductPrecision<BFloat16>(128, 0x1p126f, 0x1p-7f, overflow);
  }
}

TEST(MatMulNVFP4A16, LocalScaleIncludesSubnormalZeroAndNegative) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  const auto run = [](auto type) {
    using T = decltype(type);
    const std::vector<uint8_t> scales{0x01, 0x00, 0xb8, 0x7e};
    const std::vector<uint8_t> packed(4 * 8, 0x22);
    const std::vector<T> a(65 * 16, T(1.0f));
    std::vector<T> expected;
    for (int row = 0; row < 65; ++row) {
      for (uint8_t code : scales) expected.emplace_back(16.0f * DecodeScale(code) * 0.125f);
    }
    Check<T>({65, 16}, 4, a, packed, scales, 0.125f, expected);
  };
  run(MLFloat16{});
  run(BFloat16{});
}

TEST(MatMulNVFP4A16, DoesNotRoundAccumulatorBeforeGlobalScale) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  const auto run = [](auto type, float half_ulp) {
    using T = decltype(type);
    for (int64_t m : {1, 15, 65}) {
      std::vector<T> a(static_cast<size_t>(m * 128), T(0.0f));
      for (int64_t row = 0; row < m; ++row) {
        a[static_cast<size_t>(row * 128)] = T(1.0f);
        a[static_cast<size_t>(row * 128 + 8)] = T(half_ulp);
      }
      const T expected((1.0f + half_ulp) * 0.75f);
      ASSERT_NE(static_cast<float>(expected), static_cast<float>(T(0.75f)));
      Check<T>({m, 128}, 1, a, std::vector<uint8_t>(64, 0x22),
               std::vector<uint8_t>(8, 0x38), 0.75f, std::vector<T>(static_cast<size_t>(m), expected));
    }
  };
  run(MLFloat16{}, 0x1p-11f);
  run(BFloat16{}, 0x1p-8f);
}

TEST(MatMulNVFP4A16, CancellationAvoidsGlobalWeightRoundingAndOverflow) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  const auto run = [](auto type) {
    using T = decltype(type);
    for (int64_t m : {1, 15, 65}) {
      std::vector<T> a(static_cast<size_t>(m * 128), T(0.0f));
      for (int64_t row = 0; row < m; ++row) {
        a[static_cast<size_t>(row * 128)] = T(1.0f);
        a[static_cast<size_t>(row * 128 + 8)] = T(-1.0f);
        a[static_cast<size_t>(row * 128 + 16)] = T(1.0f / 256.0f);
      }
      Check<T>({m, 128}, 1, a, std::vector<uint8_t>(64, 0x22),
               std::vector<uint8_t>(8, 0x38), 100000.0f,
               std::vector<T>(static_cast<size_t>(m), T(100000.0f / 256.0f)));
    }
  };
  run(MLFloat16{});
  run(BFloat16{});
}

TEST(MatMulNVFP4A16, EmptyDimensionsAndContraction) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  const auto run = [](auto type) {
    using T = decltype(type);
    RunCase<T>({0, 128}, 7);
    RunCase<T>({2, 128}, 0);
    RunCase<T>({3, 0}, 5);
  };
  run(MLFloat16{});
  run(BFloat16{});
}

TEST(MatMulNVFP4A16, RejectsShapeAndScaleMismatches) {
  const auto run = [](std::vector<int64_t> a_shape, std::vector<int64_t> b_shape,
                      std::vector<int64_t> scale_shape, std::vector<int64_t> global_shape,
                      const char* error) {
    const auto size = [](const auto& shape) {
      return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>{});
    };
    OpTester test("MatMulNVFP4A16", 1, kMSDomain);
    test.AddInput<BFloat16>("A", a_shape, std::vector<BFloat16>(static_cast<size_t>(size(a_shape))));
    test.AddInput<uint8_t>("B", b_shape, std::vector<uint8_t>(static_cast<size_t>(size(b_shape))));
    test.AddInput<uint8_t>("weight_scale", scale_shape, std::vector<uint8_t>(static_cast<size_t>(size(scale_shape))));
    test.AddInput<float>("weight_scale_2", global_shape, std::vector<float>(static_cast<size_t>(size(global_shape)), 1));
    test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.0f)});
    test.Run(OpTester::ExpectResult::kExpectFailure, error);
  };
  run({}, {1, 8}, {1, 1}, {}, "A must have rank");
  run({1, 16}, {8}, {1, 1}, {}, "B must have shape");
  run({1, 18}, {1, 9}, {1, 2}, {}, "K must be divisible");
  run({1, 32}, {1, 8}, {1, 1}, {}, "A's last dimension");
  run({1, 16}, {1, 8}, {2, 1}, {}, "weight_scale must have shape");
  run({1, 16}, {1, 8}, {1}, {}, "weight_scale must have shape");
  run({1, 16}, {1, 8}, {1, 2}, {}, "weight_scale must have shape");
  run({1, 16}, {1, 8}, {1, 1}, {2}, "weight_scale_2 must contain");
}

TEST(MatMulNVFP4A16, RuntimeShapeValidation) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 or newer is required.";
  OpTester test("MatMulNVFP4A16", 1, kMSDomain);
  test.AddShapeToTensorData(false);
  test.AddInput<BFloat16>("A", {1, 32}, std::vector<BFloat16>(32));
  test.AddInput<uint8_t>("B", {1, 8}, std::vector<uint8_t>(8));
  test.AddInput<uint8_t>("weight_scale", {1, 1}, {0x38});
  test.AddInput<float>("weight_scale_2", {}, {1});
  test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.0f)});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "A's last dimension", {}, nullptr, &providers);
}

TEST(MatMulNVFP4A16, NoCpuFallback) {
  OpTester test("MatMulNVFP4A16", 1, kMSDomain);
  test.AddInput<BFloat16>("A", {1, 16}, std::vector<BFloat16>(16));
  test.AddInput<uint8_t>("B", {1, 8}, std::vector<uint8_t>(8));
  test.AddInput<uint8_t>("weight_scale", {1, 1}, {0x38});
  test.AddInput<float>("weight_scale_2", {}, {1});
  test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.0f)});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "Could not find an implementation", {}, nullptr, &providers);
}

}  // namespace onnxruntime::test
#endif
