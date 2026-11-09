// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
namespace onnxruntime::test {
namespace {

void RunPackedCuda(OpTester& test) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.SetOutputTolerance(0, 0);
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename T>
void TestNvfp4GlobalScale(int n = 17) {
  constexpr int k = 128;
  constexpr std::array<float, 16> values{
      0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
      0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  constexpr float global_scale = .333333f;
  for (int rows : {1, 7, 8, 9, 17, 129}) {
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
      OpTester test("MatMulPacked", 1, kMSDomain);
      test.AddAttribute("K", int64_t{k});
      test.AddAttribute("N", int64_t{n});
      test.AddAttribute("packed_type", int64_t{40});
      test.AddAttribute("dequantize_to_activation_dtype", int64_t{1});
      test.AddInput<T>("A", {1, rows, k}, a);
      test.AddInput<uint8_t>("B", {n, (k / 64) * 36}, packed, initializer);
      test.AddInput<float>("global_scale", {}, {global_scale}, initializer);
      test.AddOutput<T>("Y", {1, rows, n}, expected);
      RunPackedCuda(test);
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

TEST(MatMulPackedPrecisionTest, CompleteOperandRoundingIsOptIn) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
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
    const BFloat16 rounded(a0 * BFloat16(w0).ToFloat() + a1 * BFloat16(w1).ToFloat());
    ASSERT_NE(rounded.ToFloat(), 0.f);
    for (int rows : {1, 7, 8, 9, 17}) {
      for (int round_weights : {0, 1}) {
        SCOPED_TRACE(testing::Message() << "type=" << packed_type << " rows=" << rows
                                        << " round=" << round_weights);
        std::vector<BFloat16> a(rows * k, BFloat16(0.f));
        for (int row = 0; row < rows; ++row) {
          a[row * k] = BFloat16(a0);
          a[row * k + 1] = BFloat16(a1);
        }
        const BFloat16 result = round_weights || rows > 8 ? rounded : BFloat16(0.f);
        OpTester test("MatMulPacked", 1, kMSDomain);
        test.AddAttribute("K", int64_t{k});
        test.AddAttribute("N", int64_t{n});
        test.AddAttribute("packed_type", static_cast<int64_t>(packed_type));
        if (round_weights) test.AddAttribute("dequantize_to_activation_dtype", int64_t{1});
        test.AddInput<BFloat16>("A", {rows, k}, a);
        test.AddInput<uint8_t>("B", {n, row_bytes}, packed);
        test.AddOutput<BFloat16>("Y", {rows, n}, std::vector<BFloat16>(rows * n, result));
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
      // Keep unit terms out of large-magnitude MMA partials: their internal
      // accumulation order differs between Ampere and Hopper.
      a[row * k] = BFloat16(0x1p25f);
      a[row * k + 64] = BFloat16(-0x1p25f);
      a[row * k + 128] = BFloat16(1.f);
      a[row * k + 160] = BFloat16(1.f);
    }
    OpTester test("MatMulPacked", 1, kMSDomain);
    test.AddAttribute("K", int64_t{k});
    test.AddAttribute("N", int64_t{n});
    test.AddAttribute("packed_type", int64_t{8});
    test.AddAttribute("dequantize_to_activation_dtype", int64_t{1});
    test.AddInput<BFloat16>("A", {rows, k}, a);
    test.AddInput<uint8_t>("B", {n, row_bytes}, packed);
    test.AddOutput<BFloat16>("Y", {rows, n}, std::vector<BFloat16>(rows * n, BFloat16(2.f)));
    RunPackedCuda(test);
  }
}

}  // namespace
}  // namespace onnxruntime::test
#endif
