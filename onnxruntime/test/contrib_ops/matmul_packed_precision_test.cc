// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
namespace onnxruntime::test {
namespace {

constexpr std::array<const char*, 3> kPrecisionPolicies{nullptr, "0", "1"};

void RunPackedCuda(OpTester& test, const char* policy = nullptr, float relative_tolerance = 0.f) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  SessionOptions options;
  if (policy != nullptr) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(
        kOrtSessionOptionsCudaMatMulPackedAllowUnroundedGemv, policy));
  }
  test.SetOutputTolerance(0, relative_tolerance);
  test.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename T>
void TestNvfp4GlobalScale(int n = 17) {
  constexpr int k = 128;
  constexpr std::array<float, 16> values{
      0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
      0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  constexpr float global_scale = .333333f;
  for (int rows : {1, 7, 8, 9, 17, 129}) {
    if (n >= 65536 && rows > 8) continue;
    SCOPED_TRACE(testing::Message() << "rows=" << rows);
    std::vector<T> a(rows * k), expected(rows * n);
    std::vector<uint8_t> packed(n * (k / 64) * 36);
    std::vector<float> weights(n * k);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < k; ++col) {
        a[row * k + col] = T(.125f + static_cast<float>((col + row) % 23) / 256.f);
      }
    }
    for (int col = 0; col < n; ++col) {
      for (int block = 0; block < k / 64; ++block) {
        auto* bytes = packed.data() + (col * (k / 64) + block) * 36;
        for (int group = 0; group < 4; ++group) {
          const uint8_t scale = static_cast<uint8_t>(0x38 + (col + block + group) % 7);
          bytes[group] = scale;
          const float d = std::ldexp(1.f + static_cast<float>(scale & 7) / 8.f,
                                     static_cast<int>(scale >> 3) - 7);
          for (int element = 0; element < 8; ++element) {
            const int lo = (col + group + element) % 16;
            const int hi = (col + block + element + 5) % 16;
            bytes[4 + group * 8 + element] = static_cast<uint8_t>(lo | (hi << 4));
            weights[col * k + block * 64 + group * 16 + element] = T(values[lo] * d).ToFloat();
            weights[col * k + block * 64 + group * 16 + element + 8] = T(values[hi] * d).ToFloat();
          }
        }
      }
    }
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < n; ++col) {
        float sum = 0.f;
        for (int reduction = 0; reduction < k; ++reduction) {
          sum += a[row * k + reduction].ToFloat() * weights[col * k + reduction];
        }
        expected[row * n + col] = T(sum * global_scale);
      }
    }
    for (bool initializer : {false, true}) {
      for (const char* policy : kPrecisionPolicies) {
        OpTester test("MatMulPacked", 1, kMSDomain);
        test.AddAttribute("K", int64_t{k});
        test.AddAttribute("N", int64_t{n});
        test.AddAttribute("packed_type", int64_t{40});
        test.AddInput<T>("A", {1, rows, k}, a);
        test.AddInput<uint8_t>("B", {n, (k / 64) * 36}, packed, initializer);
        test.AddInput<float>("global_scale", {}, {global_scale}, initializer);
        test.AddOutput<T>("Y", {1, rows, n}, expected);
        // BF16 prefill (M > 8) applies global_scale in the cuBLAS GEMM epilogue, which can round one unit in the
        // last place away from an FP32 multiply followed by the cast. Decode kernels and FP32 prefill stay exact.
        const float relative_tolerance = std::is_same_v<T, BFloat16> && rows > 8 ? 0x1p-7f : 0.f;
        RunPackedCuda(test, policy, relative_tolerance);
      }
    }
  }
}

TEST(MatMulPackedPrecisionTest, Nvfp4Bf16GlobalScaleBeforeCast) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestNvfp4GlobalScale<BFloat16>();
}

TEST(MatMulPackedPrecisionTest, Nvfp4Fp16GlobalScaleBeforeCast) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestNvfp4GlobalScale<MLFloat16>();
}

TEST(MatMulPackedPrecisionTest, Nvfp4GlobalScaleAcrossPrefillTiles) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestNvfp4GlobalScale<BFloat16>(4099);
  TestNvfp4GlobalScale<MLFloat16>(4099);
}

TEST(MatMulPackedPrecisionTest, Nvfp4GlobalScaleLargeNDefaultDecode) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestNvfp4GlobalScale<BFloat16>(65536);
}

TEST(MatMulPackedPrecisionTest, Nvfp4UnsignedScaleKnownAnswers) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int k = 64;
  constexpr int n = 17;
  for (int rows : {1, 8, 9}) {
    std::vector<BFloat16> a(rows * k, BFloat16(0.f)), expected(rows * n);
    std::vector<uint8_t> packed(n * 36, 0x11);
    for (int col = 0; col < n; ++col) {
      const std::array<uint8_t, 4> scales{0x00, 0x01, 0x7e, 0x7f};
      for (int group = 0; group < 4; ++group) packed[col * 36 + group] = scales[group];
    }
    for (int row = 0; row < rows; ++row) {
      const int group = row % 4;
      a[row * k + group * 16] = BFloat16(1.f);
      const std::array<float, 4> values{0.f, 0x1p-10f, 224.f, 0.f};
      for (int col = 0; col < n; ++col) expected[row * n + col] = BFloat16(values[group]);
    }
    for (const char* policy : kPrecisionPolicies) {
      OpTester test("MatMulPacked", 1, kMSDomain);
      test.AddAttribute("K", int64_t{k});
      test.AddAttribute("N", int64_t{n});
      test.AddAttribute("packed_type", int64_t{40});
      test.AddInput<BFloat16>("A", {rows, k}, a);
      test.AddInput<uint8_t>("B", {n, 36}, packed);
      test.AddOutput<BFloat16>("Y", {rows, n}, expected);
      RunPackedCuda(test, policy);
    }
  }
}

TEST(MatMulPackedPrecisionTest, Nvfp4Bf16ProductionProjectionShapes) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr std::array<float, 16> values{
      0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
      0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  constexpr std::array<int, 8> active{0, 17, 63, 128, 255, 512, 1535, 2047};
  constexpr float global_scale = -.333333f;
  for (const auto& shape : {std::array<int, 2>{18560, 4096}, {4096, 8192}, {35, 192}, {8191, 640}, {8192, 128}, {8193, 640}}) {
    const int n = shape[0], k = shape[1];
    const int row_bytes = (k / 64) * 36;
    std::vector<uint8_t> packed(n * row_bytes);
    for (int col = 0; col < n; ++col) {
      for (int block = 0; block < k / 64; ++block) {
        auto* bytes = packed.data() + col * row_bytes + block * 36;
        for (int group = 0; group < 4; ++group) {
          // Unsigned contract scales, including zero and the 0x7f sentinel.
          bytes[group] = static_cast<uint8_t>((col + block * 7 + group * 19) % 128);
          for (int j = 0; j < 8; ++j) {
            bytes[4 + group * 8 + j] = static_cast<uint8_t>(col * 17 + block * 13 + group * 11 + j * 29);
          }
        }
      }
    }
    for (int rows : {1, 3, 7, 8}) {
      SCOPED_TRACE(testing::Message() << "n=" << n << " k=" << k << " rows=" << rows);
      std::vector<BFloat16> a(rows * k, BFloat16(0.f)), expected(rows * n);
      for (int row = 0; row < rows; ++row) {
        for (int index : active) {
          if (index < k) a[row * k + index] = BFloat16(static_cast<float>((index + row) % 9 - 4) / 8.f);
        }
        for (int col = 0; col < n; ++col) {
          float sum = 0.f;
          for (int index : active) {
            if (index >= k) continue;
            const auto* bytes = packed.data() + col * row_bytes + (index / 64) * 36;
            const int group = (index % 64) / 16;
            const uint8_t scale = bytes[group];
            const int exponent = (scale >> 3) & 15, fraction = scale & 7;
            float d = exponent == 0
                          ? std::ldexp(static_cast<float>(fraction), -9)
                          : std::ldexp(1.f + static_cast<float>(fraction) / 8.f, exponent - 7);
            if (scale == 127) d = 0.f;
            const int position = index % 16;
            const int code = (bytes[4 + group * 8 + position % 8] >> (4 * (position / 8))) & 15;
            const float operand = BFloat16(d * values[code]).ToFloat();
            sum += a[row * k + index].ToFloat() * operand;
          }
          // Sparse dyadic products are exactly representable in FP32, so this
          // independent reference is insensitive to the MMA reduction order.
          expected[row * n + col] = BFloat16(sum * global_scale);
        }
      }
      for (bool initializer : {false, true}) {
        OpTester test("MatMulPacked", 1, kMSDomain);
        test.AddAttribute("K", static_cast<int64_t>(k));
        test.AddAttribute("N", static_cast<int64_t>(n));
        test.AddAttribute("packed_type", int64_t{40});
        test.AddInput<BFloat16>("A", {rows, k}, a);
        test.AddInput<uint8_t>("B", {n, row_bytes}, packed, initializer);
        test.AddInput<float>("global_scale", {}, {global_scale}, initializer);
        test.AddOutput<BFloat16>("Y", {rows, n}, expected);
        RunPackedCuda(test);
      }
    }
  }
}

TEST(MatMulPackedPrecisionTest, Nvfp4Bf16PackedOperandExtremes) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int k = 128, n = 35, row_bytes = 72;
  constexpr std::array<float, 16> values{
      0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
      0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  constexpr std::array<uint8_t, 8> scales{0, 1, 7, 8, 56, 125, 126, 127};
  constexpr std::array<float, 8> decoded{0.f, 0x1p-9f, 7.f * 0x1p-9f, 0x1p-6f, 1.f, 416.f, 448.f, 0.f};
  std::vector<uint8_t> packed(n * row_bytes);
  for (int col = 0; col < n; ++col) {
    for (int block = 0; block < k / 64; ++block) {
      auto* bytes = packed.data() + col * row_bytes + block * 36;
      for (int group = 0; group < 4; ++group) {
        bytes[group] = scales[col % scales.size()];
        for (int j = 0; j < 8; ++j) {
          bytes[4 + group * 8 + j] = static_cast<uint8_t>(col * 17 + block * 13 + group * 11 + j * 29);
        }
      }
    }
  }
  // The second sweep observes all eight high-nibble lanes, without introducing
  // multi-product sums whose FP32 order might differ between GPU architectures.
  for (int first : {0, 8}) {
    for (int rows : {1, 7, 8}) {
      SCOPED_TRACE(testing::Message() << "first=" << first << " rows=" << rows);
      std::vector<BFloat16> a(rows * k, BFloat16(0.f)), expected(rows * n);
      for (int row = 0; row < rows; ++row) {
        const int index = row * 17 + first;
        a[row * k + index] = BFloat16(1.f);
        for (int col = 0; col < n; ++col) {
          const auto* bytes = packed.data() + col * row_bytes + (index / 64) * 36;
          const int group = (index % 64) / 16, position = index % 16;
          const int code = (bytes[4 + group * 8 + position % 8] >> (4 * (position / 8))) & 15;
          float sum = 0.f;
          sum += BFloat16(decoded[col % decoded.size()] * values[code]).ToFloat();
          expected[row * n + col] = BFloat16(sum);
        }
      }
      for (bool initializer : {false, true}) {
        OpTester test("MatMulPacked", 1, kMSDomain);
        test.AddAttribute("K", int64_t{k});
        test.AddAttribute("N", int64_t{n});
        test.AddAttribute("packed_type", int64_t{40});
        test.AddInput<BFloat16>("A", {rows, k}, a);
        test.AddInput<uint8_t>("B", {n, row_bytes}, packed, initializer);
        test.AddOutput<BFloat16>("Y", {rows, n}, expected);
        RunPackedCuda(test);
      }
    }
  }
}

template <typename T>
void TestCompleteOperandRounding() {
  constexpr int k = 256;
  constexpr int n = 17;
  const MLFloat16 source_scale(.019f);
  for (int packed_type : {8, 16}) {
    const int block_size = packed_type == 8 ? 32 : 256;
    const int block_bytes = packed_type == 8 ? 34 : 66;
    const int row_bytes = (k / block_size) * block_bytes;
    std::vector<uint8_t> packed(n * row_bytes);
    for (int col = 0; col < n; ++col) {
      auto* bytes = packed.data() + col * row_bytes;
      bytes[0] = static_cast<uint8_t>(source_scale.val);
      bytes[1] = static_cast<uint8_t>(source_scale.val >> 8);
      bytes[2] = packed_type == 8 ? 43 : 1;
      if (packed_type == 8) bytes[3] = 255;
    }
    const float d = source_scale.ToFloat() * (packed_type == 8 ? 1.f : .125f);
    const float w0 = d * 43.f;
    const float w1 = packed_type == 8 ? -d : d * 8.f;
    const float a0 = packed_type == 8 ? 1.f : 8.f;
    const float a1 = packed_type == 8 ? 43.f : -43.f;
    const T rounded(a0 * T(w0).ToFloat() + a1 * T(w1).ToFloat());
    ASSERT_NE(rounded.ToFloat(), 0.f);
    for (int rows : {1, 7, 8, 9, 17}) {
      for (const char* policy : kPrecisionPolicies) {
        SCOPED_TRACE(testing::Message() << "type=" << packed_type << " rows=" << rows
                                        << " policy=" << (policy != nullptr ? policy : "default"));
        std::vector<T> a(rows * k, T(0.f));
        for (int row = 0; row < rows; ++row) {
          a[row * k] = T(a0);
          a[row * k + 1] = T(a1);
        }
        const bool unrounded_gemv = policy != nullptr && policy[0] == '1' && rows <= 8;
        const T result = unrounded_gemv ? T(0.f) : rounded;
        OpTester test("MatMulPacked", 1, kMSDomain);
        test.AddAttribute("K", int64_t{k});
        test.AddAttribute("N", int64_t{n});
        test.AddAttribute("packed_type", static_cast<int64_t>(packed_type));
        test.AddInput<T>("A", {rows, k}, a);
        test.AddInput<uint8_t>("B", {n, row_bytes}, packed);
        test.AddOutput<T>("Y", {rows, n}, std::vector<T>(rows * n, result));
        RunPackedCuda(test, policy);
      }
    }
  }
}

TEST(MatMulPackedPrecisionTest, CompleteBf16OperandRoundingIsDefaultWithGemvOptOut) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestCompleteOperandRounding<BFloat16>();
}

TEST(MatMulPackedPrecisionTest, CompleteFp16OperandRoundingIsDefaultWithGemvOptOut) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestCompleteOperandRounding<MLFloat16>();
}

TEST(MatMulPackedPrecisionTest, InvalidGemvPrecisionPolicyFailsInitialization) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (const char* policy : {"", "2", "true"}) {
    SCOPED_TRACE(testing::Message() << "policy=" << policy);
    OpTester test("MatMulPacked", 1, kMSDomain);
    test.AddAttribute("K", int64_t{32});
    test.AddAttribute("N", int64_t{1});
    test.AddAttribute("packed_type", int64_t{1});
    test.AddInput<MLFloat16>("A", {1, 32}, std::vector<MLFloat16>(32, MLFloat16(0.f)));
    test.AddInput<uint8_t>("B", {1, 64}, std::vector<uint8_t>(64, 0));
    test.AddOutput<MLFloat16>("Y", {1, 1}, {MLFloat16(0.f)});
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(
        kOrtSessionOptionsCudaMatMulPackedAllowUnroundedGemv, policy));
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(options, OpTester::ExpectResult::kExpectFailure,
             std::string(kOrtSessionOptionsCudaMatMulPackedAllowUnroundedGemv) + " must be 0 or 1",
             {}, nullptr, &providers);
  }
}

TEST(MatMulPackedPrecisionTest, RoundedBf16DecodeNonzeroAcrossDispatchPaths) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  const MLFloat16 scale(.019f);
  const BFloat16 expected(BFloat16(scale.ToFloat() * 43.f).ToFloat() -
                          43.f * BFloat16(scale.ToFloat()).ToFloat());
  ASSERT_NE(expected.ToFloat(), 0.f);
  // Also exercise this with 75-virtual PTX on SM80+: hardware-only MMA dispatch returns zeros.
  for (int k : {32, 64}) {
    const int row_bytes = (k / 32) * 34;
    for (int n : {17, 65536}) {
      std::vector<uint8_t> packed(n * row_bytes, 0);
      for (int col = 0; col < n; ++col) {
        auto* bytes = packed.data() + col * row_bytes;
        bytes[0] = static_cast<uint8_t>(scale.val);
        bytes[1] = static_cast<uint8_t>(scale.val >> 8);
        bytes[2] = 43;
        bytes[3] = 255;
      }
      for (int rows : {1, 7, 8}) {
        SCOPED_TRACE(testing::Message() << "k=" << k << " n=" << n << " rows=" << rows);
        std::vector<BFloat16> a(rows * k, BFloat16(0.f));
        for (int row = 0; row < rows; ++row) {
          a[row * k] = BFloat16(1.f);
          a[row * k + 1] = BFloat16(43.f);
        }
        OpTester test("MatMulPacked", 1, kMSDomain);
        test.AddAttribute("K", static_cast<int64_t>(k));
        test.AddAttribute("N", static_cast<int64_t>(n));
        test.AddAttribute("packed_type", int64_t{8});
        test.AddInput<BFloat16>("A", {rows, k}, a);
        test.AddInput<uint8_t>("B", {n, row_bytes}, packed);
        test.AddOutput<BFloat16>("Y", {rows, n}, std::vector<BFloat16>(rows * n, expected));
        RunPackedCuda(test);
      }
    }
  }
}

TEST(MatMulPackedPrecisionTest, RoundedBf16ContiguousKCancellation) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int k = 256;
  constexpr int n = 17;
  constexpr int row_bytes = (k / 32) * 34;
  std::vector<uint8_t> packed(n * row_bytes, 0);
  const MLFloat16 scale(1.f);
  for (int row = 0; row < n; ++row) {
    for (int block = 0; block < k / 32; ++block) {
      auto* bytes = packed.data() + row * row_bytes + block * 34;
      bytes[0] = static_cast<uint8_t>(scale.val);
      bytes[1] = static_cast<uint8_t>(scale.val >> 8);
      bytes[2] = 1;
    }
  }
  for (int rows : {1, 7, 8}) {
    std::vector<BFloat16> a(rows * k, BFloat16(0.f));
    for (int row = 0; row < rows; ++row) {
      // Separate large-value MMA partials from unit terms. Lanes 20/28 also
      // combine the units before the large values in the SIMT reduction.
      a[row * k] = BFloat16(0x1p25f);
      a[row * k + 64] = BFloat16(-0x1p25f);
      a[row * k + 160] = BFloat16(1.f);
      a[row * k + 224] = BFloat16(1.f);
    }
    OpTester test("MatMulPacked", 1, kMSDomain);
    test.AddAttribute("K", int64_t{k});
    test.AddAttribute("N", int64_t{n});
    test.AddAttribute("packed_type", int64_t{8});
    test.AddInput<BFloat16>("A", {rows, k}, a);
    test.AddInput<uint8_t>("B", {n, row_bytes}, packed);
    test.AddOutput<BFloat16>("Y", {rows, n}, std::vector<BFloat16>(rows * n, BFloat16(2.f)));
    RunPackedCuda(test);
  }
}

TEST(MatMulPackedPrecisionTest, Q8Bf16VectorRecordWindowsAndStrideFallback) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (const auto& shape : {std::array<int, 2>{17, 192}, {17, 256}, {8193, 256}}) {
    const auto [n, k] = shape;
    const int row_bytes = (k / 32) * 34;
    std::vector<uint8_t> packed(n * row_bytes);
    std::vector<BFloat16> weights(n * k);
    for (int col = 0; col < n; ++col) {
      for (int block = 0; block < k / 32; ++block) {
        auto* bytes = packed.data() + col * row_bytes + block * 34;
        const MLFloat16 scale((col + block) % 2 ? -.019f : .027f);
        bytes[0] = static_cast<uint8_t>(scale.val);
        bytes[1] = static_cast<uint8_t>(scale.val >> 8);
        for (int j = 0; j < 32; ++j) {
          const int code = (col * 17 + block * 13 + j * 29) % 256 - 128;
          bytes[j + 2] = static_cast<uint8_t>(code);
          weights[col * k + block * 32 + j] = BFloat16(scale.ToFloat() * static_cast<float>(code));
        }
      }
    }
    for (int rows : {1, 3, 8}) {
      SCOPED_TRACE(testing::Message() << "N=" << n << " K=" << k << " rows=" << rows);
      std::vector<BFloat16> a(rows * k, BFloat16(0.f)), expected(rows * n);
      for (int row = 0; row < rows; ++row) {
        // Sparse dyadic activations make these small FP32 sums exact irrespective
        // of MMA's internal summation order, while sampling both record halves.
        for (int block = 0; block < k / 32; ++block) {
          a[row * k + block * 32 + (block * 5 + row * 17) % 32] = BFloat16(.125f);
        }
        for (int col = 0; col < n; ++col) {
          float sum = 0.f;
          for (int block = 0; block < k / 32; ++block) {
            sum += .125f * weights[col * k + block * 32 + (block * 5 + row * 17) % 32].ToFloat();
          }
          expected[row * n + col] = BFloat16(sum);
        }
      }
      OpTester test("MatMulPacked", 1, kMSDomain);
      test.AddAttribute("K", static_cast<int64_t>(k));
      test.AddAttribute("N", static_cast<int64_t>(n));
      test.AddAttribute("packed_type", int64_t{8});
      test.AddInput<BFloat16>("A", {rows, k}, a);
      test.AddInput<uint8_t>("B", {n, row_bytes}, packed);
      test.AddOutput<BFloat16>("Y", {rows, n}, expected);
      RunPackedCuda(test);
    }
  }
}

TEST(MatMulPackedPrecisionTest, Iq2Bf16ProductionWidthsPreserveCompleteOperands) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int k = 4096;
  constexpr int row_bytes = (k / 256) * 66;
  constexpr std::array<int, 8> active{0, 31, 255, 511, 1023, 1535, 2047, 4095};
  for (int n : {4096, 5376, 5377}) {
    std::vector<uint8_t> packed(n * row_bytes);
    for (int col = 0; col < n; ++col) {
      for (int block = 0; block < k / 256; ++block) {
        auto* bytes = packed.data() + col * row_bytes + block * 66;
        const MLFloat16 scale(.019f * (block & 1 ? -1.f : 1.f) + static_cast<float>(col % 17 + 1) / 4096.f);
        bytes[0] = static_cast<uint8_t>(scale.val);
        bytes[1] = static_cast<uint8_t>(scale.val >> 8);
        for (int record = 0; record < 8; ++record) {
          const uint32_t signs_scale =
              (static_cast<uint32_t>((col + block + record) & 15) << 28) |
              ((col + record * 13) & 127u) |
              (((col + record * 17 + 1) & 127u) << 7) |
              (((col + record * 23 + 2) & 127u) << 14) |
              (((col + record * 31 + 3) & 127u) << 21);
          for (int group = 0; group < 4; ++group) {
            bytes[2 + record * 8 + group] = static_cast<uint8_t>(col * 17 + block * 13 + record * 11 + group * 29);
            bytes[6 + record * 8 + group] = static_cast<uint8_t>(signs_scale >> (8 * group));
          }
        }
      }
    }
    for (int rows : {1, 3, 8}) {
      SCOPED_TRACE(testing::Message() << "n=" << n << " rows=" << rows);
      std::vector<BFloat16> a(rows * k, BFloat16(0.f)), expected(rows * n);
      for (int row = 0; row < rows; ++row) {
        for (int index : active) a[row * k + index] = BFloat16(static_cast<float>((index + row) % 3 - 1));
        for (int col = 0; col < n; ++col) {
          float sum = 0.f;
          for (int index : active) {
            const auto* bytes = packed.data() + col * row_bytes + (index / 256) * 66;
            const auto scale = MLFloat16::FromBits(
                static_cast<uint16_t>(bytes[0] | (static_cast<uint16_t>(bytes[1]) << 8)));
            const int record = (index % 256) / 32;
            const auto* codes = bytes + 2 + record * 8;
            const uint32_t signs_scale = static_cast<uint32_t>(codes[4]) |
                                         (static_cast<uint32_t>(codes[5]) << 8) |
                                         (static_cast<uint32_t>(codes[6]) << 16) |
                                         (static_cast<uint32_t>(codes[7]) << 24);
            const int group = (index % 32) / 8, position = index % 8;
            const uint32_t seven = (signs_scale >> (7 * group)) & 127u;
            uint32_t parity = seven;
            parity ^= parity >> 4;
            parity ^= parity >> 2;
            parity ^= parity >> 1;
            const uint32_t signs = seven | ((parity & 1u) << 7);
            const uint64_t grid = contrib::gguf_quant::iq2xxs_grid[codes[group]];
            const float d = scale.ToFloat() * (.5f + static_cast<float>(signs_scale >> 28)) * .25f;
            const float operand = BFloat16(d * static_cast<float>((grid >> (8 * position)) & 255u) *
                                           ((signs & (1u << position)) ? -1.f : 1.f))
                                      .ToFloat();
            sum += a[row * k + index].ToFloat() * operand;
          }
          // Eight dyadic products fit exactly in FP32 independently of the MMA
          // partition/reduction order.
          expected[row * n + col] = BFloat16(sum);
        }
      }
      for (bool initializer : {false, true}) {
        OpTester test("MatMulPacked", 1, kMSDomain);
        test.AddAttribute("K", int64_t{k});
        test.AddAttribute("N", static_cast<int64_t>(n));
        test.AddAttribute("packed_type", int64_t{16});
        test.AddInput<BFloat16>("A", {rows, k}, a);
        test.AddInput<uint8_t>("B", {n, row_bytes}, packed, initializer);
        test.AddOutput<BFloat16>("Y", {rows, n}, expected);
        RunPackedCuda(test);
      }
    }
  }
}

TEST(MatMulPackedPrecisionTest, Iq2Bf16OneHotSweepsEveryBlockPosition) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int k = 256, n = 35, rows = 8, row_bytes = 66;
  std::mt19937 rng(34002);
  std::vector<uint8_t> packed(n * row_bytes);
  for (int col = 0; col < n; ++col) {
    auto* bytes = packed.data() + col * row_bytes;
    for (int byte = 2; byte < row_bytes; ++byte) bytes[byte] = static_cast<uint8_t>(rng());
    const MLFloat16 scale(.019f + static_cast<float>(col % 17) / 4096.f);
    bytes[0] = static_cast<uint8_t>(scale.val);
    bytes[1] = static_cast<uint8_t>(scale.val >> 8);
  }
  for (int base = 0; base < k; base += rows) {
    SCOPED_TRACE(testing::Message() << "position_base=" << base);
    std::vector<BFloat16> a(rows * k, BFloat16(0.f)), expected(rows * n);
    for (int row = 0; row < rows; ++row) {
      const int index = base + row;
      a[row * k + index] = BFloat16(1.f);
      for (int col = 0; col < n; ++col) {
        const auto* bytes = packed.data() + col * row_bytes;
        const float scale = MLFloat16::FromBits(
                                static_cast<uint16_t>(bytes[0] | (static_cast<uint16_t>(bytes[1]) << 8)))
                                .ToFloat();
        const auto* record = bytes + 2 + (index / 32) * 8;
        uint32_t signs_scale = 0;
        for (int byte = 0; byte < 4; ++byte) signs_scale |= static_cast<uint32_t>(record[4 + byte]) << (8 * byte);
        const int group = (index % 32) / 8, position = index % 8;
        const uint32_t seven = (signs_scale >> (7 * group)) & 127u;
        int sign = position == 7 ? 0 : (seven >> position) & 1u;
        if (position == 7) {
          for (int bit = 0; bit < 7; ++bit) sign ^= (seven >> bit) & 1u;
        }
        const uint64_t grid = contrib::gguf_quant::iq2xxs_grid[record[group]];
        const float d = scale * (.5f + static_cast<float>(signs_scale >> 28)) * .25f;
        expected[row * n + col] =
            BFloat16(d * static_cast<float>((grid >> (8 * position)) & 255u) * (sign ? -1.f : 1.f));
      }
    }
    OpTester test("MatMulPacked", 1, kMSDomain);
    test.AddAttribute("K", int64_t{k});
    test.AddAttribute("N", int64_t{n});
    test.AddAttribute("packed_type", int64_t{16});
    test.AddInput<BFloat16>("A", {rows, k}, a);
    test.AddInput<uint8_t>("B", {n, row_bytes}, packed);
    test.AddOutput<BFloat16>("Y", {rows, n}, expected);
    RunPackedCuda(test);
  }
}

TEST(MatMulPackedPrecisionTest, ActivationTypeRelu2AndSiluAcrossDispatchPaths) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int k = 256, n = 17, row_bytes = (k / 32) * 34;
  std::vector<uint8_t> packed(n * row_bytes, 0);
  const MLFloat16 scale(.25f);
  for (int col = 0; col < n; ++col) {
    auto* bytes = packed.data() + col * row_bytes;
    bytes[0] = static_cast<uint8_t>(scale.val);
    bytes[1] = static_cast<uint8_t>(scale.val >> 8);
    bytes[2] = static_cast<uint8_t>((col % 2 == 0) ? 4 : -4);
  }
  for (const char* act : {"none", "relu2", "relu_squared", "silu"}) {
    for (int rows : {1, 8, 9}) {
      SCOPED_TRACE(testing::Message() << "act=" << act << " rows=" << rows);
      std::vector<BFloat16> a(rows * k, BFloat16(0.f)), expected(rows * n);
      for (int row = 0; row < rows; ++row) {
        const float x = static_cast<float>(row % 3 + 1) * .5f;
        a[row * k] = BFloat16(x);
        for (int col = 0; col < n; ++col) {
          const float pre = BFloat16(x * ((col % 2 == 0) ? 1.f : -1.f)).ToFloat();
          if (std::string(act) == "none") {
            expected[row * n + col] = BFloat16(pre);
          } else if (std::string(act) == "silu") {
            expected[row * n + col] = BFloat16(pre / (1.f + std::exp(-pre)));
          } else {
            const float pos = pre < 0.f ? 0.f : pre;
            expected[row * n + col] = BFloat16(pos * pos);
          }
        }
      }
      OpTester test("MatMulPacked", 1, kMSDomain);
      test.AddAttribute("K", int64_t{k});
      test.AddAttribute("N", int64_t{n});
      test.AddAttribute("packed_type", int64_t{8});
      test.AddAttribute("activation_type", std::string(act));
      test.AddInput<BFloat16>("A", {rows, k}, a);
      test.AddInput<uint8_t>("B", {n, row_bytes}, packed);
      test.AddOutput<BFloat16>("Y", {rows, n}, expected);
      RunPackedCuda(test);
    }
  }
}

}  // namespace
}  // namespace onnxruntime::test
#endif
