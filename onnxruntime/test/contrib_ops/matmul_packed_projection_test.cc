// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#if defined(USE_CUDA)
namespace onnxruntime::test {
namespace {

void RunPacked(OpTester& test) {
  test.SetOutputTolerance(0, 0);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename S>
void CheckQ8(int rows, int n) {
  constexpr int k = 256;
  std::vector<BFloat16> a(rows * k, BFloat16(0.0f));
  std::vector<uint8_t> b(n * k);
  std::vector<S> scales(n * k / 32);
  std::vector<BFloat16> expected(rows * n);
  for (int col = 0; col < n; ++col) {
    for (int block = 0; block < k / 32; ++block) {
      scales[col * (k / 32) + block] = S(0.001003f * (1 + col % 5 + block));
    }
    for (int kk = 0; kk < k; ++kk) b[col * k + kk] = static_cast<uint8_t>(col * 31 + kk);
  }
  for (int r = 0; r < rows; ++r) {
    const int kk = r * 32 + 3;
    a[r * k + kk] = BFloat16(1.0f);
    for (int col = 0; col < n; ++col) {
      expected[r * n + col] =
          BFloat16(static_cast<float>(scales[col * (k / 32) + kk / 32]) * (int(b[col * k + kk]) - 128));
    }
  }
  OpTester test("MatMulPackedQ8", 1, kMSDomain);
  // Non-matrix leading dimensions must survive flattening.
  test.AddInput<BFloat16>("A", {1, rows, k}, a);
  test.AddInput<uint8_t>("B", {n, k / 32, 32}, b);
  test.AddInput<S>("scales", {n, k / 32}, scales);
  test.AddOutput<BFloat16>("Y", {1, rows, n}, expected);
  RunPacked(test);
}

void CheckIQ2(int rows, int n) {
  constexpr int k = 256;
  std::vector<BFloat16> a(rows * k, BFloat16(0.0f));
  std::vector<uint8_t> b(n * k / 4, 0);
  std::vector<BFloat16> scales(n);
  std::vector<BFloat16> expected(rows * n);
  for (int col = 0; col < n; ++col) {
    const auto d = MLFloat16(0.001003f * (1 + col % 7));
    scales[col] = BFloat16::FromBits(d.val);  // Frozen ABI transports FP16 bits as BF16.
    for (int block = 0; block < 8; ++block) {
      auto* record = b.data() + (col * 8 + block) * 8;
      // Grid entry 1: {43,8,8,8,8,8,8,8}; sign index 1 negates element zero.
      record[0] = record[1] = record[2] = record[3] = 1;
      record[4] = 1;
      record[7] = static_cast<uint8_t>((block + col % 3) << 4);
    }
    for (int r = 0; r < rows; ++r) {
      const float block_scale = d.ToFloat() * (0.5f + float(r + col % 3)) * 0.25f;
      expected[r * n + col] = BFloat16(-43.0f * block_scale);
    }
  }
  for (int r = 0; r < rows; ++r) a[r * k + r * 32] = BFloat16(1.0f);
  OpTester test("MatMulPackedIQ2XXS", 1, kMSDomain);
  test.AddInput<BFloat16>("A", {rows, k}, a);
  test.AddInput<uint8_t>("B", {n, k / 32, 8}, b);
  test.AddInput<BFloat16>("scales", {n, k / 256}, scales);
  test.AddOutput<BFloat16>("Y", {rows, n}, expected);
  RunPacked(test);
}

void CheckNVFP4(int rows, int n) {
  constexpr int k = 4096;
  constexpr std::array<float, 8> values{0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  std::vector<BFloat16> a(rows * k, BFloat16(0.0f));
  std::vector<uint8_t> b(n * k / 2);
  std::vector<uint8_t> scales(n * k / 16);
  std::vector<BFloat16> expected(rows * n);
  for (int col = 0; col < n; ++col) {
    const uint8_t code = static_cast<uint8_t>(col % 16);
    std::fill_n(b.begin() + col * k / 2, k / 2, static_cast<uint8_t>(code * 17));
    std::fill_n(scales.begin() + col * k / 16, k / 16, uint8_t{0x3a});  // E4M3 1.25.
    const float weight = values[code & 7] * ((code & 8) ? -1.0f : 1.0f);
    for (int r = 0; r < rows; ++r) expected[r * n + col] = BFloat16(weight * 1.25f * 0.33333f);
  }
  for (int r = 0; r < rows; ++r) a[r * k + r * 16] = BFloat16(1.0f);
  OpTester test("MatMulPackedNVFP4", 1, kMSDomain);
  test.AddInput<BFloat16>("A", {rows, k}, a);
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("scales", {n, k / 16}, scales);
  test.AddInput<float>("global_scale", {1}, {0.33333f});
  test.AddOutput<BFloat16>("Y", {rows, n}, expected);
  RunPacked(test);
}

}  // namespace

TEST(MatMulPackedProjection, Q8SourceScalePrecisionAndRowTails) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int rows : {1, 3, 8}) {
    CheckQ8<MLFloat16>(rows, 19);
    CheckQ8<BFloat16>(rows, 19);
  }
}

TEST(MatMulPackedProjection, IQ2CompleteOperandRoundingAndRawScaleBits) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int rows : {1, 3, 8}) CheckIQ2(rows, 19);
}

TEST(MatMulPackedProjection, Q8BFloatScaleRowOneScalesAfterIntegerDot) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  std::vector<BFloat16> a(64, BFloat16(0.0f));
  a[0] = BFloat16(125.0f);
  a[1] = BFloat16(-127.0f);
  std::vector<uint8_t> b(64, uint8_t{128});
  b[0] = 255;
  b[1] = 253;
  OpTester test("MatMulPackedQ8", 1, kMSDomain);
  test.AddInput<BFloat16>("A", {1, 64}, a);
  test.AddInput<uint8_t>("B", {1, 2, 32}, b);
  test.AddInput<BFloat16>("scales", {1, 2}, {BFloat16(0.00314f), BFloat16(0.017f)});
  test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.0f)});
  RunPacked(test);
}

TEST(MatMulPackedProjection, NVFP4HalfPairsAndMmaWithGlobalScale) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int rows : {1, 3, 8}) CheckNVFP4(rows, 19);
}

#ifndef ORT_NO_EXCEPTIONS
TEST(MatMulPackedProjection, RejectsUnsupportedPrefillRows) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  OpTester test("MatMulPackedQ8", 1, kMSDomain);
  test.AddInput<BFloat16>("A", {9, 64}, std::vector<BFloat16>(9 * 64));
  test.AddInput<uint8_t>("B", {1, 2, 32}, std::vector<uint8_t>(64));
  test.AddInput<MLFloat16>("scales", {1, 2}, {MLFloat16(1.0f), MLFloat16(1.0f)});
  test.AddOutput<BFloat16>("Y", {9, 1}, std::vector<BFloat16>(9));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "Flattened activation rows must be in [1,8]", {}, nullptr,
           &providers);
}
#endif

}  // namespace onnxruntime::test
#endif
