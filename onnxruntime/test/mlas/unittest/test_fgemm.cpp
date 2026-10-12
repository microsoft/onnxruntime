// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "test_fgemm.h"
#include "test_fgemm_fixture.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#if defined(MLAS_KLEIDIAI_OPS_AVAILABLE)
#include "core/mlas/lib/kleidiai/mlasi_kleidiai.h"
#endif

TEST(SGemmPackedModelShapesTest, M1BetaZeroAndBetaOne) {
  struct GemmShape {
    size_t n;
    size_t k;
  };

  constexpr GemmShape model_shapes[] = {
      {128, 32},
      {128, 128},
      {32, 128},
      {256, 64},
      {256, 256},
      {128, 256},
  };
  constexpr float betas[] = {0.0f, 1.0f};

  MlasFgemmTest<float, true, false> tester;
  for (const auto& shape : model_shapes) {
    for (float beta : betas) {
      SCOPED_TRACE(testing::Message()
                   << "M=1 N=" << shape.n << " K=" << shape.k
                   << " beta=" << beta);
      // The fixture initializes C to -0.5, so beta=1 exercises the same
      // pre-populated-output accumulation used after Gemm bias broadcasting.
      tester.Test(false, false, 1, shape.n, shape.k, 1, 1.0f, beta);
    }
  }
}

#if defined(MLAS_KLEIDIAI_OPS_AVAILABLE)
namespace {

bool KleidiAIOpsSgemmIsAvailable(size_t n, size_t k) {
  return ArmKleidiAI::MlasGemmPackBSizeKleidiAIOps(
             CblasNoTrans, CblasNoTrans, n, k) != 0;
}

}  // namespace

TEST(SGemmKleidiAIOpsTest, ReusePackedWeightsAcrossRowsThreadsAndScales) {
  // In particular, interleaved kernels must retain the same B packing when
  // switching between row-only and 2D scheduling, including K-blocked GEMMs.
  constexpr size_t N = 129;
  constexpr size_t K = 513;
  constexpr size_t Ldb = N + 3;
  constexpr size_t Lda = K + 5;
  constexpr size_t Ldc = N + 7;
  if (!KleidiAIOpsSgemmIsAvailable(N, K)) {
    GTEST_SKIP() << "KleidiAI Ops SGEMM is unavailable for N=" << N << " K=" << K << ".";
  }
  std::vector<float> b(K * Ldb, -123.0f);
  for (size_t depth = 0; depth < K; ++depth) {
    for (size_t column = 0; column < N; ++column) {
      b[depth * Ldb + column] = static_cast<float>(static_cast<int>((depth + column) % 5) - 2);
    }
  }

  const size_t packed_size = ArmKleidiAI::MlasGemmPackBSizeKleidiAIOps(CblasNoTrans, CblasNoTrans, N, K);
  ASSERT_NE(packed_size, size_t{0});
  std::vector<uint8_t> packed_b(packed_size);
  ASSERT_TRUE(ArmKleidiAI::MlasGemmPackBKleidiAIOps(
      CblasNoTrans, CblasNoTrans, N, K, b.data(), Ldb, packed_b.data()));
  ASSERT_TRUE(ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(packed_b.data()));

  for (size_t m : {1, 3, 24, 96}) {
    std::vector<float> a(m * Lda, -123.0f);
    std::vector<float> product(m * N, 0.0f);
    for (size_t row = 0; row < m; ++row) {
      for (size_t depth = 0; depth < K; ++depth) {
        a[row * Lda + depth] = static_cast<float>(static_cast<int>((row + depth) % 7) - 3);
      }
      for (size_t column = 0; column < N; ++column) {
        for (size_t depth = 0; depth < K; ++depth) {
          product[row * N + column] += a[row * Lda + depth] * b[depth * Ldb + column];
        }
      }
    }
    for (auto* thread_pool : {static_cast<MLAS_THREADPOOL*>(nullptr), GetMlasThreadPool()}) {
      for (float alpha : {1.0f, -0.5f, 0.0f}) {
        for (float beta : {0.0f, 1.0f, 0.25f}) {
          SCOPED_TRACE(testing::Message() << "M=" << m << " threaded=" << (thread_pool != nullptr)
                                          << " alpha=" << alpha << " beta=" << beta);
          std::vector<float> c(m * Ldc, 2.0f);
          MLAS_SGEMM_DATA_PARAMS data{};
          data.A = a.data();
          data.lda = Lda;
          data.B = reinterpret_cast<const float*>(packed_b.data());
          data.BIsPacked = true;
          data.C = c.data();
          data.ldc = Ldc;
          data.alpha = alpha;
          data.beta = beta;
          // Call the adapter directly so a generic MLAS fallback cannot mask
          // a broken packed-buffer contract.
          ASSERT_TRUE(ArmKleidiAI::MlasGemmBatchKleidiAIOps(
              CblasNoTrans, CblasTrans, m, N, K, &data, 1, thread_pool));
          for (size_t row = 0; row < m; ++row) {
            for (size_t column = 0; column < Ldc; ++column) {
              const float expected = column < N ? alpha * product[row * N + column] + beta * 2.0f : 2.0f;
              ASSERT_EQ(c[row * Ldc + column], expected) << "row=" << row << " column=" << column;
            }
          }
        }
      }
    }
  }
}

TEST(SGemmKleidiAIOpsTest, ThreadedPackedGemm) {
  constexpr size_t N = 128;
  constexpr size_t K = 128;
  if (!KleidiAIOpsSgemmIsAvailable(N, K)) {
    GTEST_SKIP() << "KleidiAI Ops SGEMM is unavailable for N=" << N << " K=" << K << ".";
  }
  if (MlasGetMaximumThreadCount(GetMlasThreadPool()) <= 1) {
    GTEST_SKIP() << "A multi-threaded MLAS pool is required.";
  }
  MlasFgemmTest<float, true, true> tester;
  tester.Test(false, false, 24, N, K, 1, 1.0f, 0.0f);
}

TEST(SGemmKleidiAIOpsTest, RejectsForeignPackedBufferAfterReadingOnlyPrefix) {
  // BIsPacked is not an Ops ownership tag: it is also used for generic MLAS
  // buffers. Put the smallest prefix needed for ownership detection directly
  // before a guard page so reading a complete Ops header would fault.
  constexpr size_t ForeignPrefixSize = 16;
  MatrixGuardBuffer<uint8_t> guarded_buffer;
  const auto* foreign_packed_b = guarded_buffer.GetBuffer(ForeignPrefixSize, true);
  const float a = 1.0f;
  float c = 0.0f;
  MLAS_SGEMM_DATA_PARAMS data{};
  data.A = &a;
  data.lda = 1;
  data.B = reinterpret_cast<const float*>(foreign_packed_b);
  data.BIsPacked = true;
  data.C = &c;
  data.ldc = 1;
  data.alpha = 1.0f;

  EXPECT_FALSE(ArmKleidiAI::MlasGemmBatchKleidiAIOps(
      CblasNoTrans, CblasTrans, 1, 1, 1, &data, 1, nullptr));
}

TEST(SGemmKleidiAIOpsTest, MixedPackedAndUnpackedBatch) {
  constexpr size_t M = 3;
  constexpr size_t N = 17;
  constexpr size_t K = 13;
  constexpr size_t BatchSize = 3;
  if (!KleidiAIOpsSgemmIsAvailable(N, K)) {
    GTEST_SKIP() << "KleidiAI Ops SGEMM is unavailable for N=" << N << " K=" << K << ".";
  }

  std::vector<float> a(M * K * BatchSize);
  std::vector<float> b(K * N * BatchSize);
  std::vector<float> c(M * N * BatchSize, 0.0f);
  std::vector<float> expected(M * N * BatchSize, 0.0f);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
  }

  const size_t packed_b_size = MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, N, K, nullptr);
  ASSERT_NE(packed_b_size, size_t{0});
  std::vector<uint8_t> packed_b(packed_b_size);
  MlasGemmPackB(CblasNoTrans, CblasNoTrans, N, K, b.data(), N, packed_b.data(), nullptr);
  ASSERT_TRUE(ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(packed_b.data()));

  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG selector_config{};
  selector_config.use_kleidiai = false;
  const size_t generic_packed_size = MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, N, K, &selector_config);
  ASSERT_NE(generic_packed_size, size_t{0});
  std::vector<uint8_t> generic_packed_b(generic_packed_size);
  MlasGemmPackB(CblasNoTrans, CblasNoTrans, N, K, b.data() + K * N, N,
                generic_packed_b.data(), &selector_config);
  ASSERT_FALSE(ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(generic_packed_b.data()));

  MLAS_SGEMM_DATA_PARAMS data[BatchSize]{};
  for (size_t batch = 0; batch < BatchSize; ++batch) {
    data[batch].A = a.data() + batch * M * K;
    data[batch].lda = K;
    data[batch].B = b.data() + batch * K * N;
    data[batch].ldb = N;
    data[batch].C = c.data() + batch * M * N;
    data[batch].ldc = N;
    data[batch].alpha = 1.0f;
  }
  data[0].B = reinterpret_cast<const float*>(packed_b.data());
  data[0].BIsPacked = true;
  data[1].B = reinterpret_cast<const float*>(generic_packed_b.data());
  data[1].BIsPacked = true;

  for (size_t batch = 0; batch < BatchSize; ++batch) {
    for (size_t row = 0; row < M; ++row) {
      for (size_t column = 0; column < N; ++column) {
        for (size_t depth = 0; depth < K; ++depth) {
          expected[batch * M * N + row * N + column] +=
              a[batch * M * K + row * K + depth] *
              b[batch * K * N + depth * N + column];
        }
      }
    }
  }
  for (auto* thread_pool : {static_cast<MLAS_THREADPOOL*>(nullptr), GetMlasThreadPool()}) {
    for (float beta : {0.0f, 1.0f}) {
      std::array<size_t, BatchSize> order{0, 1, 2};
      do {
        SCOPED_TRACE(testing::Message() << "order=" << order[0] << order[1] << order[2]
                                        << " beta=" << beta << " threaded=" << (thread_pool != nullptr));
        std::fill(c.begin(), c.end(), 2.0f);
        MLAS_SGEMM_DATA_PARAMS ordered_data[BatchSize]{data[order[0]], data[order[1]], data[order[2]]};
        for (auto& params : ordered_data) {
          params.beta = beta;
        }
        MlasGemmBatch(CblasNoTrans, CblasNoTrans, M, N, K, ordered_data, BatchSize, thread_pool, nullptr);
        for (size_t i = 0; i < c.size(); ++i) {
          ASSERT_EQ(c[i], expected[i] + beta * 2.0f) << "index=" << i;
        }
      } while (std::next_permutation(order.begin(), order.end()));
    }
  }
}
#endif

static size_t FGemmRegistLongExecute() {
  size_t count = 0;

  count += MlasLongExecuteTests<MlasFgemmTest<float, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasFgemmTest<float, true, false>>::RegisterLongExecute();

  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasFgemmTest<float, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasFgemmTest<float, true, true>>::RegisterLongExecute();
  }

#ifdef MLAS_SUPPORTS_GEMM_DOUBLE

  count += MlasLongExecuteTests<MlasFgemmTest<double, false, false>>::RegisterLongExecute();
  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasFgemmTest<double, false, true>>::RegisterLongExecute();
  }

#endif

  return count;
}

static size_t FGemmRegistShortExecute() {
  size_t count = 0;

  count += FgemmShortExecuteTest<float, false, false>::RegisterShortExecuteTests();
  count += FgemmShortExecuteTest<float, true, false>::RegisterShortExecuteTests();

  if (GetMlasThreadPool() != nullptr) {
    count += FgemmShortExecuteTest<float, false, true>::RegisterShortExecuteTests();
    count += FgemmShortExecuteTest<float, true, true>::RegisterShortExecuteTests();
  }

#ifdef MLAS_SUPPORTS_GEMM_DOUBLE

  count += FgemmShortExecuteTest<double, false, false>::RegisterShortExecuteTests();
  if (GetMlasThreadPool() != nullptr) {
    count += FgemmShortExecuteTest<double, false, true>::RegisterShortExecuteTests();
  }

#endif

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? FGemmRegistShortExecute() : FGemmRegistLongExecute();
});
