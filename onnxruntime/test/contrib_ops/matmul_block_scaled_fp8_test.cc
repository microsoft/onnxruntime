// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/conversion.h"

namespace onnxruntime::test {

#if defined(USE_CUDA)

// MatMulBlockQuantizedFp8Weight: weight-only block-scaled FP8 (E4M3) matmul.
//   A       : [..., K] FP16/BF16 activation
//   B       : [N, K]   FP8 E4M3 weight (one byte per value, not packed)
//   b_scale : [N, ceil(K/block_size)] FP32 per-block weight scales
//   a_scale : optional scalar FP32 -> statically quantize A to FP8 (W8A8)
//   bias    : optional [N]
//   Y       : [..., N]
// Dequantized weight value is fp8_e4m3(B[n, k]) * b_scale[n, k / block_size].

namespace {
// Builds a [N, K] FP8 E4M3 weight where every element of row r equals row_value[r].
std::vector<Float8E4M3FN> MakeConstRowWeight(const std::vector<float>& row_value, int64_t k) {
  std::vector<Float8E4M3FN> b(static_cast<size_t>(row_value.size()) * static_cast<size_t>(k));
  for (size_t r = 0; r < row_value.size(); ++r) {
    for (int64_t c = 0; c < k; ++c) {
      b[r * static_cast<size_t>(k) + static_cast<size_t>(c)] = Float8E4M3FN(row_value[r]);
    }
  }
  return b;
}
}  // namespace

// GEMM path (K not a multiple of 16 forces the cuBLAS dequant path), FP16 activations.
// Weights are constant per row, so Y[m, n] = W_val[n] * sum_k A[m, k].
TEST(MatMulBlockQuantizedFp8WeightOpTest, WeightOnlyGemmFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 24;  // K % 16 != 0 -> cuBLAS GEMM path

  // Weight row 0 = +1.0, row 1 = +2.0 (both exact in E4M3).
  std::vector<Float8E4M3FN> b = MakeConstRowWeight({1.0f, 2.0f}, k);
  // One FP32 scale per row (single K block), both 1.0.
  std::vector<float> b_scale = {1.0f, 1.0f};

  std::vector<float> a(m * k);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      a[row * k + col] = static_cast<float>(row + 1);
    }
  }
  // sum_k A[0, :] = 24, sum_k A[1, :] = 48. Y = {{24, 48}, {48, 96}}.
  std::vector<float> expected = {24.0f, 48.0f, 48.0f, 96.0f};

  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 128);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<Float8E4M3FN>("B", {n, k}, b);
  test.AddInput<float>("b_scale", {n, 1}, b_scale);
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// GEMM path with non-unit per-block scales, negative weights and bias, BF16 activations/output.
TEST(MatMulBlockQuantizedFp8WeightOpTest, WeightOnlyGemmScalesBiasBf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  constexpr int64_t k = 24;  // K % 16 != 0 -> cuBLAS GEMM path

  // Weight row 0 = +1.0, row 1 = -1.0 (both exact in E4M3).
  std::vector<Float8E4M3FN> b = MakeConstRowWeight({1.0f, -1.0f}, k);
  // Row 0 scale = 2.0, row 1 scale = 1.0.
  std::vector<float> b_scale = {2.0f, 1.0f};

  std::vector<float> a(m * k, 1.0f);
  // W[0] = 1.0 * 2.0 = 2.0; W[1] = -1.0 * 1.0 = -1.0; sum_k A = 24.
  // Y = {2*24, -1*24} + bias{1, 2} = {49, -22}.
  std::vector<float> bias = {1.0f, 2.0f};
  std::vector<float> expected = {49.0f, -22.0f};

  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 128);
  test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
  test.AddInput<Float8E4M3FN>("B", {n, k}, b);
  test.AddInput<float>("b_scale", {n, 1}, b_scale);
  test.AddOptionalInputEdge<float>();  // a_scale (skipped)
  test.AddInput<BFloat16>("bias", {n}, FloatsToBFloat16s(bias));
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Fused decode GEMV fast path (small M, K % 16 == 0, block_size % 16 == 0) with a multi-block K
// and per-block scales, FP16. Weights are constant per row so Y[m, n] = W_val[n] * sum_k A[m, k].
TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvDecodeMultiBlockFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 32;  // K % 16 == 0 -> GEMV path
  constexpr int64_t block_size = 16;
  constexpr int64_t k_blocks = k / block_size;  // 2 blocks

  // Weight row 0 = +1.0, row 1 = +2.0.
  std::vector<Float8E4M3FN> b = MakeConstRowWeight({1.0f, 2.0f}, k);
  // Unit FP32 scale for every block of every row.
  std::vector<float> b_scale(n * k_blocks, 1.0f);

  std::vector<float> a(m * k);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      a[row * k + col] = static_cast<float>(row + 1);
    }
  }
  // sum_k A[0, :] = 32, sum_k A[1, :] = 64. Y = {{32, 64}, {64, 128}}.
  std::vector<float> expected = {32.0f, 64.0f, 64.0f, 128.0f};

  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<Float8E4M3FN>("B", {n, k}, b);
  test.AddInput<float>("b_scale", {n, k_blocks}, b_scale);
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Exercises the optional a_scale (W8A8) activation path: A is statically quantized to FP8 E4M3 and
// dequantized before the matmul. Activation values are exactly representable in E4M3 so the result
// is exact. Uses the GEMV path with per-block scales and bias, BF16.
TEST(MatMulBlockQuantizedFp8WeightOpTest, GemvW8A8ActivationScaleBf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp8Weight.";
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  constexpr int64_t k = 32;  // K % 16 == 0 -> GEMV path
  constexpr int64_t block_size = 16;
  constexpr int64_t k_blocks = k / block_size;  // 2 blocks

  // Weight row 0 = +1.0, row 1 = -1.0.
  std::vector<Float8E4M3FN> b = MakeConstRowWeight({1.0f, -1.0f}, k);
  // Row 0 scale = 2.0 for both blocks, row 1 scale = 1.0 for both blocks.
  std::vector<float> b_scale = {2.0f, 2.0f, 1.0f, 1.0f};

  // A = 1.0 everywhere; a_scale = 1.0 -> fp8(1.0) * 1.0 = 1.0 (exact).
  std::vector<float> a(m * k, 1.0f);
  std::vector<float> a_scale = {1.0f};
  // W[0] = 1.0 * 2.0 = 2.0; W[1] = -1.0 * 1.0 = -1.0; sum_k A = 32.
  // Y = {2*32, -1*32} + bias{1, 2} = {65, -30}.
  std::vector<float> bias = {1.0f, 2.0f};
  std::vector<float> expected = {65.0f, -30.0f};

  OpTester test("MatMulBlockQuantizedFp8Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
  test.AddInput<Float8E4M3FN>("B", {n, k}, b);
  test.AddInput<float>("b_scale", {n, k_blocks}, b_scale);
  test.AddInput<float>("a_scale", {}, a_scale);
  test.AddInput<BFloat16>("bias", {n}, FloatsToBFloat16s(bias));
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#endif  // USE_CUDA

}  // namespace onnxruntime::test
