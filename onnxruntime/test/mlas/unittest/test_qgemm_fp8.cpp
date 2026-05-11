// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "mlas.h"
#include "test_util.h"
#include "core/common/float8.h"
#include "core/mlas/inc/mlas.h"

#include <array>
#include <limits>
#include <vector>

#if !defined(DISABLE_FLOAT8_TYPES)

namespace {

uint8_t EncodeFp8(float value, mlas_fp8_mode mode) {
  using onnxruntime::Float8E4M3FN;
  using onnxruntime::Float8E4M3FNUZ;
  using onnxruntime::Float8E5M2;
  using onnxruntime::Float8E5M2FNUZ;

  switch (mode) {
    case MLAS_FP8_MODE_E4M3_INF:
      return Float8E4M3FN(value).val;
    case MLAS_FP8_MODE_E4M3_SAT:
      return Float8E4M3FNUZ(value).val;
    case MLAS_FP8_MODE_E5M2_INF:
      return Float8E5M2(value).val;
    case MLAS_FP8_MODE_E5M2_SAT:
      return Float8E5M2FNUZ(value).val;
    default:
      ORT_THROW("Unsupported FP8 GEMM test mode.");
  }
}

void RunFp8GemmBatchThreaded(mlas_fp8_mode mode) {
  constexpr size_t BatchN = 2;
  constexpr size_t M = 3;
  constexpr size_t N = 2;
  constexpr size_t K = 4;
  constexpr size_t BlockSizeM = 2;
  constexpr size_t BlockSizeK = 2;
  constexpr size_t BlockSizeN = 1;
  constexpr size_t BlocksM = 2;
  constexpr size_t BlocksK = 2;
  constexpr size_t BlocksN = 2;
  constexpr size_t ScaleElements = BlocksM * BlocksK;
  constexpr size_t BScaleElements = BlocksK * BlocksN;

  const std::array<float, BatchN * M * K> a_values{
      1.0f, 2.0f, -1.0f, 0.5f,
      -2.0f, 1.5f, 0.0f, 4.0f,
      0.25f, -0.5f, 3.0f, -4.0f,
      -1.0f, 0.5f, 2.0f, -2.0f,
      4.0f, -0.25f, -1.5f, 1.0f,
      0.0f, 3.0f, -4.0f, 0.5f};
  const std::array<float, BatchN * K * N> b_values{
      1.0f, -1.0f,
      0.5f, 2.0f,
      -2.0f, 0.25f,
      1.5f, -0.5f,
      -0.5f, 1.0f,
      2.0f, -2.0f,
      0.25f, 1.5f,
      -1.0f, 0.5f};
  const std::array<float, BatchN * ScaleElements> scale_a{
      1.0f, 0.5f,
      2.0f, 1.5f,
      0.25f, 1.0f,
      0.5f, 2.0f};
  const std::array<float, BatchN * BScaleElements> scale_b{
      1.0f, 2.0f,
      0.25f, 1.25f,
      0.5f, 1.0f,
      2.0f, 0.25f};
  const std::array<float, BatchN> y_scale{0.5f, 2.0f};

  std::vector<uint8_t> a_fp8(a_values.size());
  std::vector<uint8_t> b_fp8(b_values.size());
  for (size_t i = 0; i < a_values.size(); ++i) {
    a_fp8[i] = EncodeFp8(a_values[i], mode);
  }
  for (size_t i = 0; i < b_values.size(); ++i) {
    b_fp8[i] = EncodeFp8(b_values[i], mode);
  }

  std::array<float, BatchN * M * N> output{};
  std::array<float, BatchN * M * N> expected{};
  std::array<MLAS_FP8_GEMM_DATA_PARAMS, BatchN> params{};

  for (size_t batch = 0; batch < BatchN; ++batch) {
    params[batch].A = a_fp8.data() + batch * M * K;
    params[batch].lda = K;
    params[batch].B = b_fp8.data() + batch * K * N;
    params[batch].ldb = N;
    params[batch].C = output.data() + batch * M * N;
    params[batch].ldc = N;
    params[batch].ScaleA = scale_a.data() + batch * ScaleElements;
    params[batch].ScaleB = scale_b.data() + batch * BScaleElements;
    params[batch].ScaleY = y_scale.data() + batch;
    params[batch].Fp8Type = mode;
    params[batch].BlockSizeM = BlockSizeM;
    params[batch].BlockSizeK = BlockSizeK;
    params[batch].BlockSizeN = BlockSizeN;
    params[batch].BlocksM = BlocksM;
    params[batch].BlocksK = BlocksK;
    params[batch].BlocksN = BlocksN;
    params[batch].ScaleAStrideK = 1;
    params[batch].ScaleAStrideM = BlocksK;
    params[batch].ScaleBStrideN = 1;
    params[batch].ScaleBStrideK = BlocksN;

    for (size_t m = 0; m < M; ++m) {
      const size_t block_m = m / BlockSizeM;
      for (size_t n = 0; n < N; ++n) {
        const size_t block_n = n / BlockSizeN;
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          const size_t block_k = k / BlockSizeK;
          const size_t a_scale_idx = batch * ScaleElements + block_m * BlocksK + block_k;
          const size_t b_scale_idx = batch * BScaleElements + block_k * BlocksN + block_n;
          const float a_deq = a_values[batch * M * K + m * K + k] * scale_a[a_scale_idx];
          const float b_deq = b_values[batch * K * N + k * N + n] * scale_b[b_scale_idx];
          acc += a_deq * b_deq;
        }
        expected[batch * M * N + m * N + n] = acc * y_scale[batch];
      }
    }
  }

  MLAS_FP8_GEMM_SHAPE_PARAMS shape{M, N, K};
  MLAS_THREADPOOL* threadpool = GetMlasThreadPool();
  if (threadpool == nullptr) {
    GTEST_SKIP() << "MlasFp8GemmBatch threaded test requires an MLAS thread pool.";
  }

  MlasFp8GemmBatch(shape, params.data(), BatchN, threadpool);

  // Inputs are exactly representable test values, so the scalar fallback should match closely.
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-5f);
  }
}

}  // namespace

TEST(Fp8Gemm, BatchedModesSymmetricThreaded) {
  RunFp8GemmBatchThreaded(MLAS_FP8_MODE_E4M3_INF);
  RunFp8GemmBatchThreaded(MLAS_FP8_MODE_E4M3_SAT);
  RunFp8GemmBatchThreaded(MLAS_FP8_MODE_E5M2_INF);
  RunFp8GemmBatchThreaded(MLAS_FP8_MODE_E5M2_SAT);
}

TEST(Fp8Gemm, EmptyDimensionsSkipUnusedBufferValidation) {
  MLAS_FP8_GEMM_DATA_PARAMS params{};
  params.Fp8Type = MLAS_FP8_MODE_E4M3_INF;
  params.BlockSizeM = 2;
  params.BlockSizeK = 2;
  params.BlockSizeN = 2;

  MLAS_FP8_GEMM_SHAPE_PARAMS empty_output_shape{3, 0, 4};
  MlasFp8GemmBatch(empty_output_shape, &params, 1, nullptr);

  std::array<float, 6> output;
  output.fill(-1.0f);

  MLAS_FP8_GEMM_SHAPE_PARAMS empty_reduction_shape{3, 2, 0};
  params.C = output.data();
  params.ldc = 2;
  MlasFp8GemmBatch(empty_reduction_shape, &params, 1, nullptr);

  for (float value : output) {
    EXPECT_EQ(value, 0.0f);
  }
}

TEST(Fp8Gemm, ZeroColumnReturnsBeforeWorkItemOverflow) {
  MLAS_FP8_GEMM_SHAPE_PARAMS shape{std::numeric_limits<size_t>::max(), 0, 4};
  EXPECT_NO_THROW(MlasFp8GemmBatch(shape, nullptr, 2, nullptr));
}

#endif  // !defined(DISABLE_FLOAT8_TYPES)
