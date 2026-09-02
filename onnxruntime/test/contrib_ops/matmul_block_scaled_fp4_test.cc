// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA)
// Needed for the CUDA_VERSION check below. MatMulBlockQuantizedFp4Weight relies on the NVFP4
// conversion intrinsics that are only available in CUDA 12.8 and newer.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "contrib_ops/cuda/math/matmul_block_scaled_fp4_tiling.h"
#endif

#include <algorithm>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/conversion.h"

namespace onnxruntime::test {

#if defined(USE_CUDA) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080

// NVFP4 (E2M1) 4-bit magnitude nibble encodings (sign bit is 0x8):
//   +0.0 -> 0x0, +0.5 -> 0x1, +1.0 -> 0x2, +1.5 -> 0x3,
//   +2.0 -> 0x4, +3.0 -> 0x5, +4.0 -> 0x6, +6.0 -> 0x7
// A packed byte holds two values: low nibble is element 2j, high nibble is element 2j+1.
//
// E4M3 (float8e4m3fn) scale byte encodings:
//   1.0 -> 0x38, 2.0 -> 0x40, 0.5 -> 0x30

// A -> [M, K] all ones per row scaled by (m + 1); weights are constant per row, so the
// operator must reproduce Y[m, n] = W_val[n] * sum_k A[m, k].
TEST(MatMulBlockQuantizedFp4WeightOpTest, WeightOnlyBasicFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 16;  // one block with block_size == 16

  // Weight row 0 = +1.0 (nibble 0x2 -> byte 0x22), row 1 = +2.0 (nibble 0x4 -> byte 0x44).
  std::vector<uint8_t> b(n * (k / 2));
  for (int64_t j = 0; j < k / 2; ++j) {
    b[0 * (k / 2) + j] = 0x22;
    b[1 * (k / 2) + j] = 0x44;
  }
  // One E4M3 scale per row (single K block), both 1.0.
  std::vector<uint8_t> weight_scale = {0x38, 0x38};
  std::vector<float> weight_scale_2 = {1.0f};

  std::vector<float> a(m * k);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      a[row * k + col] = static_cast<float>(row + 1);
    }
  }
  // W[0, :] = 1.0, W[1, :] = 2.0; sum_k A[0, :] = 16, sum_k A[1, :] = 32.
  std::vector<float> expected = {16.0f, 32.0f, 32.0f, 64.0f};

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, 1}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Exercises non-unit per-block E4M3 scales, a global weight_scale_2, negative weights, bias and
// a skipped optional input_scale, with BF16 activations/output.
TEST(MatMulBlockQuantizedFp4WeightOpTest, WeightOnlyScalesBiasBf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  constexpr int64_t k = 16;

  // Weight row 0 = +1.0 (0x22), row 1 = -1.0 (nibble 0xA -> byte 0xAA).
  std::vector<uint8_t> b(n * (k / 2));
  for (int64_t j = 0; j < k / 2; ++j) {
    b[0 * (k / 2) + j] = 0x22;
    b[1 * (k / 2) + j] = 0xAA;
  }
  // Row 0 scale = 2.0 (0x40), row 1 scale = 1.0 (0x38).
  std::vector<uint8_t> weight_scale = {0x40, 0x38};
  std::vector<float> weight_scale_2 = {3.0f};

  std::vector<float> a(m * k, 1.0f);
  // W[0, :] = 1.0 * 3.0 * 2.0 = 6.0; W[1, :] = -1.0 * 3.0 * 1.0 = -3.0; sum_k A = 16.
  // Y = {6*16, -3*16} + bias{1, 2} = {97, -46}.
  std::vector<float> bias = {1.0f, 2.0f};
  std::vector<float> expected = {97.0f, -46.0f};

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, 1}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
  test.AddOptionalInputEdge<float>();  // input_scale (skipped)
  test.AddInput<BFloat16>("bias", {n}, FloatsToBFloat16s(bias));
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Exercises the fused decode GEMV fast path (small M) with a multi-block K (K = 64 == 4 blocks,
// K % 32 == 0), FP16 activations. Weights are constant per row so Y[m, n] = W_val[n] * sum_k A[m, k].
TEST(MatMulBlockQuantizedFp4WeightOpTest, GemvDecodeMultiBlockFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 64;  // 4 blocks with block_size == 16, K % 32 == 0 -> GEMV path
  constexpr int64_t k_blocks = k / 16;

  // Weight row 0 = +1.0 (byte 0x22), row 1 = +2.0 (byte 0x44).
  std::vector<uint8_t> b(n * (k / 2));
  for (int64_t j = 0; j < k / 2; ++j) {
    b[0 * (k / 2) + j] = 0x22;
    b[1 * (k / 2) + j] = 0x44;
  }
  // Unit E4M3 scale (0x38 == 1.0) for every block of every row.
  std::vector<uint8_t> weight_scale(n * k_blocks, 0x38);
  std::vector<float> weight_scale_2 = {1.0f};

  std::vector<float> a(m * k);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      a[row * k + col] = static_cast<float>(row + 1);
    }
  }
  // W[0, :] = 1.0, W[1, :] = 2.0; sum_k A[0, :] = 64, sum_k A[1, :] = 128.
  std::vector<float> expected = {64.0f, 128.0f, 128.0f, 256.0f};

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, k_blocks}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Exercises the fused decode GEMV fast path with M == 1, per-block scales, a global weight_scale_2,
// negative weights and bias (BF16). K = 32 == 2 blocks, K % 32 == 0 -> GEMV path.
TEST(MatMulBlockQuantizedFp4WeightOpTest, GemvDecodeScalesBiasBf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  constexpr int64_t k = 32;  // 2 blocks with block_size == 16
  constexpr int64_t k_blocks = k / 16;

  // Weight row 0 = +1.0 (0x22), row 1 = -1.0 (nibble 0xA -> byte 0xAA).
  std::vector<uint8_t> b(n * (k / 2));
  for (int64_t j = 0; j < k / 2; ++j) {
    b[0 * (k / 2) + j] = 0x22;
    b[1 * (k / 2) + j] = 0xAA;
  }
  // Row 0 scale = 2.0 (0x40) for both blocks, row 1 scale = 1.0 (0x38) for both blocks.
  std::vector<uint8_t> weight_scale = {0x40, 0x40, 0x38, 0x38};
  std::vector<float> weight_scale_2 = {3.0f};

  std::vector<float> a(m * k, 1.0f);
  // W[0, :] = 1.0 * 3.0 * 2.0 = 6.0; W[1, :] = -1.0 * 3.0 * 1.0 = -3.0; sum_k A = 32.
  // Y = {6*32, -3*32} + bias{1, 2} = {193, -94}.
  std::vector<float> bias = {1.0f, 2.0f};
  std::vector<float> expected = {193.0f, -94.0f};

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, k_blocks}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
  test.AddOptionalInputEdge<float>();  // input_scale (skipped)
  test.AddInput<BFloat16>("bias", {n}, FloatsToBFloat16s(bias));
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Exercises K that is not a multiple of block_size: K = 24 with block_size == 16 gives
// ceil(K / 16) == 2 scale blocks where the second block only covers 8 of the 16 K elements.
// K % 32 != 0 also forces the reference dequantize + GEMM path instead of the decode GEMV.
TEST(MatMulBlockQuantizedFp4WeightOpTest, PartialTrailingScaleBlockFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  constexpr int64_t k = 24;  // ceil(24 / 16) == 2 blocks; the trailing block holds only 8 elements
  constexpr int64_t k_blocks = 2;

  // Weight row 0 = +1.0 (byte 0x22), row 1 = +2.0 (byte 0x44).
  std::vector<uint8_t> b(n * (k / 2));
  for (int64_t j = 0; j < k / 2; ++j) {
    b[0 * (k / 2) + j] = 0x22;
    b[1 * (k / 2) + j] = 0x44;
  }
  // Row 0: block 0 scale 1.0 (0x38), block 1 scale 2.0 (0x40). Row 1: both blocks 1.0.
  std::vector<uint8_t> weight_scale = {0x38, 0x40, 0x38, 0x38};
  std::vector<float> weight_scale_2 = {1.0f};

  std::vector<float> a(m * k, 1.0f);
  // Y[0, 0] = 16 * (1.0 * 1.0) + 8 * (1.0 * 2.0) = 32.
  // Y[0, 1] = 16 * (2.0 * 1.0) + 8 * (2.0 * 1.0) = 48.
  std::vector<float> expected = {32.0f, 48.0f};

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, k_blocks}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// K = 48 is a multiple of block_size (3 full scale blocks) but not of 32, so the decode GEMV
// (which requires K % 32 == 0) must be skipped even though M is small. Uses per-block scales that
// differ across all three blocks so a wrong ceil(K / block_size) index would change the result.
TEST(MatMulBlockQuantizedFp4WeightOpTest, KMultipleOf16NotOf32Bf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 48;  // 3 blocks with block_size == 16; 48 % 32 != 0
  constexpr int64_t k_blocks = k / 16;

  // Single weight row, all +1.0 (byte 0x22).
  std::vector<uint8_t> b(n * (k / 2), 0x22);
  // Block scales 1.0 (0x38), 2.0 (0x40), 0.5 (0x30).
  std::vector<uint8_t> weight_scale = {0x38, 0x40, 0x30};
  std::vector<float> weight_scale_2 = {1.0f};

  std::vector<float> a(m * k, 1.0f);
  // Y = 16 * 1.0 + 16 * 2.0 + 16 * 0.5 = 56.
  std::vector<float> expected = {56.0f};

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, k_blocks}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// K == 0 reduces to an empty sum: Y must be exactly zero (no uninitialized output) even though
// K % 32 == 0 would otherwise select a fast path. FP16, no bias.
TEST(MatMulBlockQuantizedFp4WeightOpTest, ZeroKNoBiasFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 3;
  constexpr int64_t n = 2;

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<MLFloat16>("A", {m, 0}, std::vector<MLFloat16>{});
  test.AddInput<uint8_t>("B", {n, 0}, std::vector<uint8_t>{});
  test.AddInput<uint8_t>("weight_scale", {n, 0}, std::vector<uint8_t>{});
  test.AddInput<float>("weight_scale_2", {1}, std::vector<float>{1.0f});
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(std::vector<float>(m * n, 0.0f)));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// K == 0 with a bias: Y must be the broadcast bias row. BF16.
TEST(MatMulBlockQuantizedFp4WeightOpTest, ZeroKWithBiasBf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 3;

  std::vector<float> bias = {1.0f, -2.0f, 0.5f};
  std::vector<float> expected = {1.0f, -2.0f, 0.5f, 1.0f, -2.0f, 0.5f};

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<BFloat16>("A", {m, 0}, std::vector<BFloat16>{});
  test.AddInput<uint8_t>("B", {n, 0}, std::vector<uint8_t>{});
  test.AddInput<uint8_t>("weight_scale", {n, 0}, std::vector<uint8_t>{});
  test.AddInput<float>("weight_scale_2", {1}, std::vector<float>{1.0f});
  test.AddOptionalInputEdge<float>();  // input_scale (skipped)
  test.AddInput<BFloat16>("bias", {n}, FloatsToBFloat16s(bias));
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Exercises the row-tiled GEMV (RowsPerBlock > 1). The tiling is only selected when the column
// grid (ceil(N / 8) blocks) already covers a full wave of SMs, so N must be large: N = 2048 gives
// 256 column blocks, which is above the SM count of every current device. Weights and per-block
// scales vary by column and A varies by row so that a mis-indexed row or column is caught.
//
// The M values sweep every tile the launcher can pick: 2 and 4 are exact tiles, 3 and 5 are ragged
// (gridDim.y == 2 with a partly out-of-range last block, whose extra rows are clamped and dropped).
// All expected values are multiples of K below 1024, hence exact in both FP16 and BF16.
namespace {

constexpr int64_t kRowTiledN = 2048;

// Weight row `col` is +1.0 (byte 0x22) or +2.0 (byte 0x44); its E4M3 block scale is 2.0 (0x40)
// on every third row and 1.0 (0x38) elsewhere.
float RowTiledWeightValue(int64_t col) { return (col % 2 == 0) ? 1.0f : 2.0f; }
float RowTiledScaleValue(int64_t col) { return (col % 3 == 0) ? 2.0f : 1.0f; }

void MakeRowTiledWeights(int64_t n, int64_t k, int64_t k_blocks,
                         std::vector<uint8_t>& b, std::vector<uint8_t>& weight_scale) {
  b.assign(n * (k / 2), 0);
  weight_scale.assign(n * k_blocks, 0);
  for (int64_t col = 0; col < n; ++col) {
    const uint8_t byte = (RowTiledWeightValue(col) == 1.0f) ? 0x22 : 0x44;
    for (int64_t j = 0; j < k / 2; ++j) {
      b[col * (k / 2) + j] = byte;
    }
    const uint8_t s = (RowTiledScaleValue(col) == 2.0f) ? 0x40 : 0x38;
    for (int64_t blk = 0; blk < k_blocks; ++blk) {
      weight_scale[col * k_blocks + blk] = s;
    }
  }
}

// A[row, :] = row + 1, so Y[row, col] = W[col] * S[col] * K * (row + 1).
void MakeRowTiledActivationsAndExpected(int64_t m, int64_t n, int64_t k,
                                        std::vector<float>& a, std::vector<float>& expected) {
  a.assign(m * k, 0.0f);
  expected.assign(m * n, 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      a[row * k + col] = static_cast<float>(row + 1);
    }
    for (int64_t col = 0; col < n; ++col) {
      expected[row * n + col] = RowTiledWeightValue(col) * RowTiledScaleValue(col) *
                                static_cast<float>(k) * static_cast<float>(row + 1);
    }
  }
}

}  // namespace

TEST(MatMulBlockQuantizedFp4WeightOpTest, GemvDecodeRowTiledFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t n = kRowTiledN;
  constexpr int64_t k = 64;
  constexpr int64_t k_blocks = k / 16;

  std::vector<uint8_t> b, weight_scale;
  MakeRowTiledWeights(n, k, k_blocks, b, weight_scale);
  const std::vector<float> weight_scale_2 = {1.0f};

  for (int m_val : {2, 3, 4, 5}) {
    const int64_t m = m_val;
    SCOPED_TRACE("M = " + std::to_string(m));
    std::vector<float> a, expected;
    MakeRowTiledActivationsAndExpected(m, n, k, a, expected);

    OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("block_size", 16);
    test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
    test.AddInput<uint8_t>("B", {n, k / 2}, b);
    test.AddInput<uint8_t>("weight_scale", {n, k_blocks}, weight_scale);
    test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
    test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
    test.SetOutputTolerance(0.5f);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MatMulBlockQuantizedFp4WeightOpTest, GemvDecodeRowTiledBf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t n = kRowTiledN;
  constexpr int64_t k = 32;
  constexpr int64_t k_blocks = k / 16;

  std::vector<uint8_t> b, weight_scale;
  MakeRowTiledWeights(n, k, k_blocks, b, weight_scale);
  const std::vector<float> weight_scale_2 = {1.0f};

  for (int m_val : {2, 3, 4, 5}) {
    const int64_t m = m_val;
    SCOPED_TRACE("M = " + std::to_string(m));
    std::vector<float> a, expected;
    MakeRowTiledActivationsAndExpected(m, n, k, a, expected);

    OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("block_size", 16);
    test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
    test.AddInput<uint8_t>("B", {n, k / 2}, b);
    test.AddInput<uint8_t>("weight_scale", {n, k_blocks}, weight_scale);
    test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
    test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
    test.SetOutputTolerance(0.5f);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

// Exercises the tensor-core GEMV sub-path (mma.m16n8k16), which the launcher selects on SM80+
// when K is a multiple of 128. None of the tests above reach it (they use K = 32/64).
//
// That path permutes the K axis so every lane's loads are contiguous, and folds the per-block
// E4M3 scale into the decoded weight instead of flushing the accumulator per block. A K
// permutation applied consistently to both operands is a no-op, so to catch an *inconsistent*
// one the weight, the block scales and the activation must all vary along K -- a uniform
// weight (as in the row-tiled tests above) would pass even with a broken k mapping.
//
// The shapes cover: a ragged N that is not a multiple of the 16-column warp tile, single and
// multi window K (128 -> 1 window, 256 -> 2, 512 -> 4), and an N large enough that the launcher
// picks the wide ColTiles = 4 / KSplit = 2 shape rather than the column-starved KSplit ladder.
namespace {

// FP4 codes cycling along K: +1.0, -2.0, +0.5, +1.5 (E2M1 codes 0x2, 0xC, 0x1, 0x3). Negative
// values exercise the sign path of the prmt-based decode.
constexpr uint8_t kMmaCodes[4] = {0x2, 0xC, 0x1, 0x3};
constexpr float kMmaValues[4] = {1.0f, -2.0f, 0.5f, 1.5f};

// The column rotates the cycle so neighbouring output columns are not interchangeable.
int MmaCodeIndex(int64_t col, int64_t kk) { return static_cast<int>((kk + col) & 3); }

// Block scale alternates 1.0 (E4M3 0x38) and 2.0 (0x40) along both K and N.
float MmaScaleValue(int64_t col, int64_t blk) { return ((col + blk) % 2 == 0) ? 1.0f : 2.0f; }
uint8_t MmaScaleByte(int64_t col, int64_t blk) { return ((col + blk) % 2 == 0) ? 0x38u : 0x40u; }

float MmaActValue(int64_t row, int64_t kk) { return static_cast<float>((kk % 4) + 1 + row); }

// Every product is a multiple of 0.5 and every partial sum stays far below 2^23, so the fp32
// reference below is exact and independent of summation order -- the kernel's permuted, split-K
// accumulation must match it bit for bit before the final cast to the output type.
void MakeMmaCase(int64_t m, int64_t n, int64_t k,
                 std::vector<uint8_t>& b, std::vector<uint8_t>& weight_scale,
                 std::vector<float>& a, std::vector<float>& expected) {
  const int64_t k_blocks = k / 16;
  b.assign(n * (k / 2), 0);
  weight_scale.assign(n * k_blocks, 0);
  for (int64_t col = 0; col < n; ++col) {
    for (int64_t kk = 0; kk < k; kk += 2) {
      const uint8_t lo = kMmaCodes[MmaCodeIndex(col, kk)];
      const uint8_t hi = kMmaCodes[MmaCodeIndex(col, kk + 1)];
      b[col * (k / 2) + kk / 2] = static_cast<uint8_t>(lo | (hi << 4));
    }
    for (int64_t blk = 0; blk < k_blocks; ++blk) {
      weight_scale[col * k_blocks + blk] = MmaScaleByte(col, blk);
    }
  }

  a.assign(m * k, 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t kk = 0; kk < k; ++kk) {
      a[row * k + kk] = MmaActValue(row, kk);
    }
  }

  expected.assign(m * n, 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int64_t kk = 0; kk < k; ++kk) {
        sum += kMmaValues[MmaCodeIndex(col, kk)] * MmaScaleValue(col, kk / 16) *
               MmaActValue(row, kk);
      }
      expected[row * n + col] = sum;
    }
  }
}

struct MmaShape {
  int64_t n;
  int64_t k;
};

// N = 8704 gives 544 column tiles and exercises a large, ragged grid. N = 36 leaves 4 columns in
// the last 16-column tile, which is the only way to make a lane's *low* column fall out of range
// (N = 40 only exercises the high column), so it covers the lo_ok == false predication.
constexpr MmaShape kMmaShapes[] = {{36, 128}, {40, 128}, {512, 256}, {2048, 512}, {8704, 256}};

}  // namespace

TEST(MatMulBlockQuantizedFp4WeightOpTest, GemvTensorCoreTilesFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  for (const auto& shape : kMmaShapes) {
    // 1-8 use one row tile, 9-16 two and 17-32 four. M=33 and 64 exercise a partial and full
    // second launch, respectively; odd values leave masked rows that must not be written.
    for (int m_val : {1, 3, 4, 8, 9, 16, 17, 32, 33, 64}) {
      const int64_t m = m_val;
      const int64_t n = shape.n;
      const int64_t k = shape.k;
      SCOPED_TRACE("N = " + std::to_string(n) + ", K = " + std::to_string(k) +
                   ", M = " + std::to_string(m));

      std::vector<uint8_t> b, weight_scale;
      std::vector<float> a, expected;
      MakeMmaCase(m, n, k, b, weight_scale, a, expected);

      OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
      test.AddAttribute<int64_t>("block_size", 16);
      test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
      test.AddInput<uint8_t>("B", {n, k / 2}, b);
      test.AddInput<uint8_t>("weight_scale", {n, k / 16}, weight_scale);
      test.AddInput<float>("weight_scale_2", {1}, {1.0f});
      test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
      test.SetOutputTolerance(0.5f);

      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

TEST(MatMulBlockQuantizedFp4WeightOpTest, GemvTensorCoreTilesBf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t n = 512;
  constexpr int64_t k = 256;

  for (int m_val : {1, 4, 8, 16, 32, 33, 64}) {
    const int64_t m = m_val;
    SCOPED_TRACE("M = " + std::to_string(m));

    std::vector<uint8_t> b, weight_scale;
    std::vector<float> a, expected;
    MakeMmaCase(m, n, k, b, weight_scale, a, expected);

    // Per-column bias, folded in by the kernel's store.
    std::vector<float> bias(n);
    for (int64_t col = 0; col < n; ++col) {
      bias[col] = static_cast<float>((col % 5) - 2);
      for (int64_t row = 0; row < m; ++row) {
        expected[row * n + col] += bias[col];
      }
    }

    OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("block_size", 16);
    test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
    test.AddInput<uint8_t>("B", {n, k / 2}, b);
    test.AddInput<uint8_t>("weight_scale", {n, k / 16}, weight_scale);
    test.AddInput<float>("weight_scale_2", {1}, {1.0f});
    test.AddOptionalInputEdge<float>();  // input_scale (skipped)
    test.AddInput<BFloat16>("bias", {n}, FloatsToBFloat16s(bias));
    test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
    test.SetOutputTolerance(0.5f);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MatMulBlockQuantizedFp4WeightOpTest, GemvTensorCoreQwenTilingBoundaries) {
  struct Case {
    int64_t n;
    int64_t k;
    int k_split;
    int col_tiles;
  };
  constexpr int sm_count = 132;
  const Case cases[] = {
      {64 * sm_count, 5120, 2, 1},        // 4 waves of single-column blocks, ordinary reduction
      {64 * sm_count, 8192, 8, 1},        // same grid, long reduction retains KSplit 8
      {512 * sm_count, 512, 2, 4},        // 8 waves of four-column blocks
      {64 * sm_count - 16, 2048, 16, 1},  // just below 4 single-column waves
  };

  for (const Case& shape : cases) {
    SCOPED_TRACE("N = " + std::to_string(shape.n) + ", K = " + std::to_string(shape.k));
    const auto config = onnxruntime::contrib::cuda::PickFp4MmaConfig(
        static_cast<int>(shape.n), static_cast<int>(shape.k), sm_count);
    EXPECT_EQ(config.k_split, shape.k_split);
    EXPECT_EQ(config.col_tiles, shape.col_tiles);
  }
}

// Lane-ownership probe for the tensor-core path.
//
// The tests above sum over the whole K axis, so a wrong lane -> (row, column) mapping could in
// principle still land on a plausible value. Here the activation is one-hot, which collapses
// Y[row, col] to a single decoded weight times its block scale, and the probe offset differs per
// row. Both the weight code and the block scale cycle modulo 3 along N, so the two columns a lane
// owns (g and g + 8, which differ by 8 == 2 mod 3) never carry the same value, and every one of
// the 8 rows produces a different value in a given column. Any swap of the mma ownership -- rows
// 2t / 2t + 1 against columns g / g + 8, or the activation row against the output column -- shows
// up as a mismatch instead of cancelling out.
namespace {

// E2M1 codes +1.0, -2.0, +0.5 and their values.
constexpr uint8_t kProbeCodes[3] = {0x2, 0xC, 0x1};
constexpr float kProbeValues[3] = {1.0f, -2.0f, 0.5f};
// E4M3 scale bytes 1.0, 4.0, 16.0 and their values. The 3 x 3 weight-times-scale products are all
// distinct, so no two (row, column) mappings can produce the same output value.
constexpr uint8_t kProbeScaleBytes[3] = {0x38, 0x48, 0x58};
constexpr float kProbeScaleValues[3] = {1.0f, 4.0f, 16.0f};

int ProbeIndex(int64_t col, int64_t x) { return static_cast<int>((col + x) % 3); }

}  // namespace

TEST(MatMulBlockQuantizedFp4WeightOpTest, GemvTensorCoreLaneOwnershipFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 8;    // full mma N extent: output rows 2t and 2t + 1 for every t
  constexpr int64_t n = 40;   // ragged: the last 16-column tile has only its low half in range
  constexpr int64_t k = 128;  // one K window
  constexpr int64_t k_blocks = k / 16;

  std::vector<uint8_t> b(n * (k / 2), 0);
  std::vector<uint8_t> weight_scale(n * k_blocks, 0);
  for (int64_t col = 0; col < n; ++col) {
    for (int64_t kk = 0; kk < k; kk += 2) {
      const uint8_t lo = kProbeCodes[ProbeIndex(col, kk)];
      const uint8_t hi = kProbeCodes[ProbeIndex(col, kk + 1)];
      b[col * (k / 2) + kk / 2] = static_cast<uint8_t>(lo | (hi << 4));
    }
    for (int64_t blk = 0; blk < k_blocks; ++blk) {
      weight_scale[col * k_blocks + blk] = kProbeScaleBytes[ProbeIndex(col, blk)];
    }
  }

  // Row `row` probes K offset 11 * row. The eight (code index, scale-block index) pairs that
  // produces are all different, and each row lands in a different 32-element lane slice of the
  // window, so no two rows can be confused for one another.
  std::vector<float> a(m * k, 0.0f);
  std::vector<float> expected(m * n, 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    const int64_t kk = 11 * row;
    a[row * k + kk] = 1.0f;
    for (int64_t col = 0; col < n; ++col) {
      expected[row * n + col] =
          kProbeValues[ProbeIndex(col, kk)] * kProbeScaleValues[ProbeIndex(col, kk / 16)];
    }
  }

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, k_blocks}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, {1.0f});
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
  test.SetOutputTolerance(0.01f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// ---------------------------------------------------------------------------
// Prefill dequantize + cuBLAS path.
//
// LaunchDequantizeNvFp4 picks DequantizeNvFp4Vec8Kernel when K % 8 == 0 and block_size is even,
// and the scalar DequantizeNvFp4Kernel otherwise. The cases below cover both sides of that guard;
// every one uses a block_size other than 16, which the decode GEMV does not handle, so the dequant
// actually runs regardless of M.
// ---------------------------------------------------------------------------

// Vectorized path: K % 8 == 0, even block_size, scale boundaries inside every 8-element chunk,
// and a negative weight row so the sign bit of the packed nibble is exercised.
TEST(MatMulBlockQuantizedFp4WeightOpTest, PrefillDequantVectorizedFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 16;  // > kGemvMaxM (8), so the dequant + cuBLAS path runs
  constexpr int64_t n = 3;
  constexpr int64_t k = 24;
  constexpr int64_t block_size = 6;
  constexpr int64_t k_blocks = k / block_size;

  // Row 0 = +1.0 (0x22), row 1 = +2.0 (0x44), row 2 = -1.5 (sign 0x8 | 0x3 -> 0xBB).
  std::vector<uint8_t> b(n * (k / 2));
  for (int64_t j = 0; j < k / 2; ++j) {
    b[0 * (k / 2) + j] = 0x22;
    b[1 * (k / 2) + j] = 0x44;
    b[2 * (k / 2) + j] = 0xBB;
  }
  // E4M3 scale bytes 0.5, 1.0, 2.0, and 4.0 arranged differently in each weight row.
  std::vector<uint8_t> weight_scale = {
      0x38, 0x40, 0x30, 0x48,
      0x40, 0x30, 0x48, 0x38,
      0x30, 0x48, 0x38, 0x40};
  std::vector<float> weight_scale_2 = {1.0f};

  std::vector<float> a(m * k);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      a[row * k + col] = static_cast<float>((row + 1) * (col + 1));
    }
  }
  constexpr float weight_values[] = {1.0f, 2.0f, -1.5f};
  constexpr float scale_values[][k_blocks] = {
      {1.0f, 2.0f, 0.5f, 4.0f},
      {2.0f, 0.5f, 4.0f, 1.0f},
      {0.5f, 4.0f, 1.0f, 2.0f}};
  std::vector<float> expected(m * n, 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      for (int64_t kk = 0; kk < k; ++kk) {
        expected[row * n + col] +=
            a[row * k + kk] * weight_values[col] * scale_values[col][kk / block_size];
      }
    }
  }

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, k_blocks}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Vectorized path with BF16, a bias, a non-unit weight_scale_2, and M = 9 to pin the boundary
// just above the decode GEMV cutoff.
TEST(MatMulBlockQuantizedFp4WeightOpTest, PrefillDequantVectorizedBiasBf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 9;  // kGemvMaxM + 1
  constexpr int64_t n = 2;
  constexpr int64_t k = 64;
  constexpr int64_t k_blocks = k / 16;

  // Row 0 = +1.0 (0x22), row 1 = -2.0 (sign 0x8 | 0x4 -> 0xCC).
  std::vector<uint8_t> b(n * (k / 2));
  for (int64_t j = 0; j < k / 2; ++j) {
    b[0 * (k / 2) + j] = 0x22;
    b[1 * (k / 2) + j] = 0xCC;
  }
  // Row 0 blocks {1.0, 2.0, 1.0, 2.0}, row 1 all 1.0.
  std::vector<uint8_t> weight_scale = {0x38, 0x40, 0x38, 0x40, 0x38, 0x38, 0x38, 0x38};
  std::vector<float> weight_scale_2 = {0.5f};
  std::vector<float> bias = {1.0f, 2.0f};

  std::vector<float> a(m * k, 1.0f);
  // Y[r, 0] = 16 * 0.5 * (1 + 2 + 1 + 2) + 1 = 48 + 1 = 49.
  // Y[r, 1] = 16 * 0.5 * 4 * (-2) + 2 = -64 + 2 = -62.
  std::vector<float> expected(m * n);
  for (int64_t row = 0; row < m; ++row) {
    expected[row * n + 0] = 49.0f;
    expected[row * n + 1] = -62.0f;
  }

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, k_blocks}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
  test.AddOptionalInputEdge<float>();  // input_scale
  test.AddInput<BFloat16>("bias", {n}, FloatsToBFloat16s(bias));
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Scalar fallback: K % 8 == 0 but block_size is odd, so the two nibbles of a packed byte can land
// in different scale blocks. The vectorized kernel assumes they never do, so it must be skipped.
// Blocks are [0,5) [5,10) [10,15) [15,16) with scales 1.0, 2.0, 0.5, 1.0.
TEST(MatMulBlockQuantizedFp4WeightOpTest, PrefillDequantOddBlockSizeFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 10;  // > kGemvMaxM
  constexpr int64_t n = 1;
  constexpr int64_t k = 16;
  constexpr int64_t block_size = 5;
  constexpr int64_t k_blocks = 4;  // ceil(16 / 5)

  std::vector<uint8_t> b(n * (k / 2), 0x22);  // all +1.0
  std::vector<uint8_t> weight_scale = {0x38, 0x40, 0x30, 0x38};
  std::vector<float> weight_scale_2 = {1.0f};

  std::vector<float> a(m * k, 1.0f);
  // Y = 5 * 1.0 + 5 * 2.0 + 5 * 0.5 + 1 * 1.0 = 18.5.
  std::vector<float> expected(m * n, 18.5f);

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, k_blocks}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Scalar fallback: K % 8 != 0, so a thread cannot own an aligned 8-element chunk.
TEST(MatMulBlockQuantizedFp4WeightOpTest, PrefillDequantKNotMultipleOf8Bf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockQuantizedFp4Weight.";
  }

  constexpr int64_t m = 12;  // > kGemvMaxM
  constexpr int64_t n = 2;
  constexpr int64_t k = 12;  // 12 % 8 == 4
  constexpr int64_t block_size = 4;
  constexpr int64_t k_blocks = k / block_size;

  // Row 0 = +1.0 (0x22), row 1 = +0.5 (0x11).
  std::vector<uint8_t> b(n * (k / 2));
  for (int64_t j = 0; j < k / 2; ++j) {
    b[0 * (k / 2) + j] = 0x22;
    b[1 * (k / 2) + j] = 0x11;
  }
  // Row 0 blocks {1.0, 2.0, 1.0}, row 1 all 1.0.
  std::vector<uint8_t> weight_scale = {0x38, 0x40, 0x38, 0x38, 0x38, 0x38};
  std::vector<float> weight_scale_2 = {1.0f};

  std::vector<float> a(m * k, 1.0f);
  // Y[r, 0] = 4 * (1*1 + 1*2 + 1*1) = 16. Y[r, 1] = 4 * 3 * 0.5 = 6.
  std::vector<float> expected(m * n);
  for (int64_t row = 0; row < m; ++row) {
    expected[row * n + 0] = 16.0f;
    expected[row * n + 1] = 6.0f;
  }

  OpTester test("MatMulBlockQuantizedFp4Weight", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, k_blocks}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#endif  // USE_CUDA && defined(CUDA_VERSION) && CUDA_VERSION >= 12080

}  // namespace onnxruntime::test
