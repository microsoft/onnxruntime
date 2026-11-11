// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
namespace onnxruntime::test {
namespace {

constexpr std::array<const char*, 3> kPrecisionPolicies{nullptr, "0", "1"};

void RunPackedCuda(OpTester& test, const char* policy = nullptr) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  SessionOptions options;
  if (policy != nullptr) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(
        kOrtSessionOptionsCudaMatMulPackedAllowUnroundedGemv, policy));
  }
  test.SetOutputTolerance(0, 0);
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
        RunPackedCuda(test, policy);
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

}  // namespace
}  // namespace onnxruntime::test
#endif
