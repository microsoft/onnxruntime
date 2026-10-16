// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "test_fgemm.h"
#include "test_fgemm_fixture.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#if defined(MLAS_TARGET_ARM64) && (!defined(_MSC_VER) || defined(__clang__))
#include "core/mlas/lib/kleidiai/sgemm_common.h"
#endif

#if defined(MLAS_KLEIDIAI_OPS_AVAILABLE)
#include "core/mlas/lib/kleidiai/mlasi_kleidiai.h"
#endif

#if defined(__linux__) && defined(__aarch64__) && defined(MLAS_KLEIDIAI_OPS_AVAILABLE)
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <thread>
#endif

#if defined(MLAS_TARGET_ARM64) && (!defined(_MSC_VER) || defined(__clang__))
TEST(SGemmPostProcessTest, StridedAlphaBeta) {
  ArmKleidiAI::MlasGemmApplyAlphaBetaStrided(nullptr, 0, 1.0f, 0.0f, nullptr, 1, true);
  struct Case {
    size_t count;
    size_t stride;
    float alpha;
    float beta;
    bool allow_memcpy;
  };
  // Cover the SIMD threshold/tail, copy path and GEMV's strided destination
  // without repeating the full GEMM parameter matrix here.
  constexpr Case cases[] = {
      {31, 1, 1.0f, 0.25f, true},
      {32, 1, -0.5f, 0.0f, true},
      {32, 1, 1.0f, 0.25f, true},
      {33, 1, -0.5f, 0.25f, true},
      {33, 1, 1.0f, 0.0f, true},
      {33, 1, 1.0f, 0.0f, false},
      {33, 3, -0.5f, 0.25f, true},
  };
  constexpr float sentinel = -123.0f;
  for (const auto& test_case : cases) {
    SCOPED_TRACE(testing::Message() << "count=" << test_case.count << " stride=" << test_case.stride
                                    << " memcpy=" << test_case.allow_memcpy << " alpha=" << test_case.alpha
                                    << " beta=" << test_case.beta);
    std::vector<float> product(test_case.count);
    std::vector<float> output(test_case.count * test_case.stride + 2, sentinel);
    float* destination = output.data() + 1;
    for (size_t i = 0; i < test_case.count; ++i) {
      product[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
      destination[i * test_case.stride] =
          test_case.beta == 0.0f ? std::numeric_limits<float>::quiet_NaN() : 2.0f;
    }
    ArmKleidiAI::MlasGemmApplyAlphaBetaStrided(
        product.data(), test_case.count, test_case.alpha, test_case.beta,
        destination, test_case.stride, test_case.allow_memcpy);
    for (size_t i = 0; i < test_case.count; ++i) {
      EXPECT_EQ(destination[i * test_case.stride], test_case.alpha * product[i] + test_case.beta * 2.0f);
      for (size_t padding = 1; padding < test_case.stride; ++padding) {
        EXPECT_EQ(destination[i * test_case.stride + padding], sentinel);
      }
    }
    EXPECT_EQ(output.front(), sentinel);
    EXPECT_EQ(output.back(), sentinel);
  }
}

TEST(SGemmPostProcessTest, PaddedOutputAndBetaZeroNaNs) {
  ArmKleidiAI::MlasGemmApplyAlphaBeta(nullptr, 0, 33, 1.0f, 0.0f, nullptr, 33);
  ArmKleidiAI::MlasGemmApplyBeta(nullptr, 3, 0, 3, 0.0f);
  constexpr size_t rows = 3;
  constexpr size_t columns = 33;
  constexpr float sentinel = -123.0f;
  std::vector<float> product(rows * columns, 2.0f);
  for (size_t padding : {0, 3}) {
    const size_t ldc = columns + padding;
    for (bool beta_only : {false, true}) {
      SCOPED_TRACE(testing::Message() << "padding=" << padding << " beta_only=" << beta_only);
      std::vector<float> output(rows * ldc + 2, sentinel);
      float* destination = output.data() + 1;
      for (size_t row = 0; row < rows; ++row) {
        std::fill_n(destination + row * ldc, columns, std::numeric_limits<float>::quiet_NaN());
      }
      if (beta_only) {
        ArmKleidiAI::MlasGemmApplyBeta(destination, rows, columns, ldc, 0.0f);
      } else {
        ArmKleidiAI::MlasGemmApplyAlphaBeta(product.data(), rows, columns, -0.5f, 0.0f, destination, ldc);
      }
      for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < ldc; ++column) {
          EXPECT_EQ(destination[row * ldc + column], column < columns ? (beta_only ? 0.0f : -1.0f) : sentinel);
        }
      }
      EXPECT_EQ(output.front(), sentinel);
      EXPECT_EQ(output.back(), sentinel);
    }
  }
}
#endif

TEST(SGemmPackedModelShapesTest, M1BetaZeroAndBetaOne) {
  struct GemmShape {
    size_t n;
    size_t k;
  };

  constexpr GemmShape model_shapes[] = {
      {128, 32},
      {32, 128},
      {256, 256},
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
  return ArmKleidiAI::MlasGemmPackBExtensionSizeKleidiAIOps(
             CblasNoTrans, CblasNoTrans, n, k, true) >
         ArmKleidiAI::MlasGemmPackBExtensionSizeKleidiAIOps(
             CblasNoTrans, CblasNoTrans, n, k, false);
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

  const size_t packed_size = MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, N, K, nullptr);
  ASSERT_NE(packed_size, size_t{0});
  std::vector<uint8_t> packed_b(packed_size);
  MlasGemmPackB(CblasNoTrans, CblasNoTrans, N, K, b.data(), Ldb, packed_b.data(), nullptr);
  ASSERT_TRUE(ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(packed_b.data(), N, K));

  constexpr std::array<std::array<float, 2>, 5> scales = {
      {{1.0f, 0.0f}, {1.0f, 1.0f}, {-0.5f, 0.0f}, {-0.5f, 0.25f}, {0.0f, 0.25f}}};
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
      for (const auto& scale : scales) {
        const float alpha = scale[0];
        const float beta = scale[1];
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

TEST(SGemmKleidiAIOpsTest, NativeOnlyPackAtGuardPage) {
  // Even disabled-selector packs reserve an initialized extension header.
  // Use the exact public size before a guard page to catch speculative reads.
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG selector{};
  selector.use_kleidiai = false;
  const size_t packed_size = MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 1, 1, &selector);
  MatrixGuardBuffer<uint8_t> guarded_buffer;
  auto* native_packed_b = guarded_buffer.GetBuffer(packed_size, true);
  const float a = 1.0f;
  const float b = 3.0f;
  MlasGemmPackB(CblasNoTrans, CblasNoTrans, 1, 1, &b, 1, native_packed_b, &selector);
  ASSERT_FALSE(ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(native_packed_b, 1, 1));
  float c = 2.0f;
  MLAS_SGEMM_DATA_PARAMS data{};
  data.A = &a;
  data.lda = 1;
  data.B = reinterpret_cast<const float*>(native_packed_b);
  data.BIsPacked = true;
  data.C = &c;
  data.ldc = 1;
  data.alpha = 1.0f;
  data.beta = 1.0f;

  EXPECT_FALSE(ArmKleidiAI::MlasGemmBatchKleidiAIOps(
      CblasNoTrans, CblasTrans, 1, 1, 1, &data, 1, nullptr));
  EXPECT_EQ(c, 2.0f);
  MlasGemmBatch(CblasNoTrans, CblasTrans, 1, 1, 1, &data, 1, nullptr, nullptr);
  EXPECT_EQ(c, 5.0f);
}

TEST(SGemmKleidiAIOpsTest, PublicPackedWeightsFallBackAcrossLayoutsAndSelectors) {
  constexpr size_t M = 3;
  constexpr size_t N = 17;
  constexpr size_t K = 13;
  constexpr size_t Ldc = N + 3;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG disabled_selector{};
  disabled_selector.use_kleidiai = false;

  for (auto trans_b : {CblasNoTrans, CblasTrans}) {
    const size_t ldb = (trans_b == CblasNoTrans ? N : K) + 2;
    std::vector<float> b((trans_b == CblasNoTrans ? K : N) * ldb, -123.0f);
    for (size_t depth = 0; depth < K; ++depth) {
      for (size_t column = 0; column < N; ++column) {
        b[trans_b == CblasNoTrans ? depth * ldb + column : column * ldb + depth] =
            static_cast<float>(static_cast<int>((depth + column) % 5) - 2);
      }
    }
    const size_t packed_size = MlasGemmPackBSize(CblasNoTrans, trans_b, N, K, nullptr);
    std::vector<uint8_t> packed_b(packed_size);
    MlasGemmPackB(CblasNoTrans, trans_b, N, K, b.data(), ldb, packed_b.data(), nullptr);
    const bool has_ops = trans_b == CblasNoTrans && KleidiAIOpsSgemmIsAvailable(N, K);
    ASSERT_EQ(ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(packed_b.data(), N, K), has_ops);

    for (auto trans_a : {CblasNoTrans, CblasTrans}) {
      const size_t lda = (trans_a == CblasNoTrans ? K : M) + 2;
      std::vector<float> a((trans_a == CblasNoTrans ? M : K) * lda, -123.0f);
      std::vector<float> product(M * N, 0.0f);
      for (size_t row = 0; row < M; ++row) {
        for (size_t depth = 0; depth < K; ++depth) {
          const float value = static_cast<float>(static_cast<int>((row + depth) % 7) - 3);
          a[trans_a == CblasNoTrans ? row * lda + depth : depth * lda + row] = value;
          for (size_t column = 0; column < N; ++column) {
            product[row * N + column] += value *
                                         b[trans_b == CblasNoTrans ? depth * ldb + column : column * ldb + depth];
          }
        }
      }
      std::vector<float> c(M * Ldc, 2.0f);
      MLAS_SGEMM_DATA_PARAMS data{};
      data.A = a.data();
      data.lda = lda;
      data.B = reinterpret_cast<const float*>(packed_b.data());
      data.BIsPacked = true;
      data.C = c.data();
      data.ldc = Ldc;
      data.alpha = 1.0f;
      data.beta = 1.0f;
      // Establish whether this exact pack can execute through Ops before
      // exercising disabled-selector and unsupported-layout fallback.
      ASSERT_EQ(ArmKleidiAI::MlasGemmBatchKleidiAIOps(
                    trans_a, CblasTrans, M, N, K, &data, 1, nullptr),
                has_ops && trans_a == CblasNoTrans);
      // Scale/thread combinations are covered by pack reuse above; this
      // test isolates dispatch across source layouts and selector changes.
      for (bool use_ops : {false, true}) {
        SCOPED_TRACE(testing::Message() << "TransA=" << trans_a << " TransB=" << trans_b << " use_ops=" << use_ops);
        std::fill(c.begin(), c.end(), 2.0f);
        data.alpha = -0.5f;
        data.beta = 0.25f;
        MlasGemmBatch(trans_a, CblasTrans, M, N, K, &data, 1, nullptr,
                      use_ops ? nullptr : &disabled_selector);
        for (size_t row = 0; row < M; ++row) {
          for (size_t column = 0; column < Ldc; ++column) {
            const float expected = column < N ? -0.5f * product[row * N + column] + 0.25f * 2.0f : 2.0f;
            ASSERT_EQ(c[row * Ldc + column], expected) << "row=" << row << " column=" << column;
          }
        }
      }
    }
  }
}

TEST(SGemmKleidiAIOpsTest, RelocatedConcatenatedPacksAndCorruptedExtensionUseNativeFallback) {
  constexpr size_t M = 3;
  constexpr size_t N = 17;
  constexpr size_t K = 13;
  constexpr size_t BatchSize = 2;
  std::vector<float> a(M * K);
  std::vector<float> b(BatchSize * K * N);
  std::vector<float> product(BatchSize * M * N, 0.0f);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
  }
  const size_t packed_size = MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, N, K, nullptr);
  std::vector<uint8_t> original(BatchSize * packed_size);
  for (size_t batch = 0; batch < BatchSize; ++batch) {
    MlasGemmPackB(CblasNoTrans, CblasNoTrans, N, K, b.data() + batch * K * N, N,
                  original.data() + batch * packed_size, nullptr);
    for (size_t row = 0; row < M; ++row) {
      for (size_t column = 0; column < N; ++column) {
        for (size_t depth = 0; depth < K; ++depth) {
          product[batch * M * N + row * N + column] +=
              a[row * K + depth] * b[batch * K * N + depth * N + column];
        }
      }
    }
  }
  // Relocate to a different address and deliberately change its alignment.
  // Attention also stores multiple packs at successive public-size offsets.
  std::vector<uint8_t> relocated(original.size() + sizeof(float));
  auto* packed_b = relocated.data() + sizeof(float);
  std::copy(original.begin(), original.end(), packed_b);
  const bool has_ops = KleidiAIOpsSgemmIsAvailable(N, K);
  std::vector<float> c(BatchSize * M * N, 2.0f);
  MLAS_SGEMM_DATA_PARAMS data[BatchSize]{};
  for (size_t batch = 0; batch < BatchSize; ++batch) {
    data[batch].A = a.data();
    data[batch].lda = K;
    data[batch].B = reinterpret_cast<const float*>(packed_b + batch * packed_size);
    data[batch].BIsPacked = true;
    data[batch].C = c.data() + batch * M * N;
    data[batch].ldc = N;
    data[batch].alpha = -0.5f;
    data[batch].beta = 0.25f;
    ASSERT_EQ(ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(data[batch].B, N, K), has_ops);
    ASSERT_EQ(ArmKleidiAI::MlasGemmBatchKleidiAIOps(
                  CblasNoTrans, CblasTrans, M, N, K, &data[batch], 1, nullptr),
              has_ops);
  }
  for (bool corrupt_extension : {false, true}) {
    if (corrupt_extension) {
      // Corrupt only one optional extension, retaining its native prefix.
      packed_b[MlasSgemmNativePackBSize(N, K)] ^= 1;
      ASSERT_FALSE(ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(packed_b, N, K));
      std::fill(c.begin(), c.end(), 2.0f);
      EXPECT_FALSE(ArmKleidiAI::MlasGemmBatchKleidiAIOps(
          CblasNoTrans, CblasTrans, M, N, K, data, 1, nullptr));
      EXPECT_TRUE(std::all_of(c.begin(), c.end(), [](float value) { return value == 2.0f; }));
    }
    for (auto* thread_pool : {static_cast<MLAS_THREADPOOL*>(nullptr), GetMlasThreadPool()}) {
      SCOPED_TRACE(testing::Message() << "corrupt_extension=" << corrupt_extension
                                      << " threaded=" << (thread_pool != nullptr));
      std::fill(c.begin(), c.end(), 2.0f);
      MlasGemmBatch(CblasNoTrans, CblasTrans, M, N, K, data, BatchSize, thread_pool, nullptr);
      for (size_t i = 0; i < c.size(); ++i) {
        ASSERT_EQ(c[i], -0.5f * product[i] + 0.25f * 2.0f) << "index=" << i;
      }
    }
  }
}

TEST(SGemmKleidiAIOpsTest, RejectsOversizedDimensionsAndStridesBeforeReadingInputs) {
  const size_t oversized = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  const size_t large_stride = static_cast<size_t>(std::numeric_limits<int>::max()) - 1;
  MatrixGuardBuffer<uint8_t> guard;
  const auto* inaccessible_input = reinterpret_cast<const float*>(guard.GetBuffer(1, true) + 1);
  struct PackingShape {
    size_t n;
    size_t k;
    size_t ldb;
  };
  const PackingShape packing_shapes[] = {
      {oversized, 1, oversized}, {1, oversized, 1}, {65536, 65536, 65536}, {1, 2, oversized}, {2, 3, 1}, {2, 2, large_stride}};
  const size_t header_only_size = ArmKleidiAI::MlasGemmPackBExtensionSizeKleidiAIOps(
      CblasNoTrans, CblasNoTrans, 1, 1, false);
  for (const auto& shape : packing_shapes) {
    SCOPED_TRACE(testing::Message() << "N=" << shape.n << " K=" << shape.k << " ldb=" << shape.ldb);
    const size_t extension_size = ArmKleidiAI::MlasGemmPackBExtensionSizeKleidiAIOps(
        CblasNoTrans, CblasNoTrans, shape.n, shape.k, true);
    if (shape.n >= 65536 || shape.k >= 65536) {
      ASSERT_EQ(extension_size, header_only_size);
    }
    std::vector<uint8_t> extension(extension_size);
    EXPECT_FALSE(ArmKleidiAI::MlasGemmPackBExtensionKleidiAIOps(
        CblasNoTrans, CblasNoTrans, shape.n, shape.k, inaccessible_input, shape.ldb,
        extension.data(), true));
  }
  struct ExecutionShape {
    size_t m;
    size_t n;
    size_t k;
    size_t lda;
    size_t ldc;
  };
  const ExecutionShape execution_shapes[] = {
      {oversized, 1, 1, 1, 1}, {1, oversized, 1, 1, oversized}, {1, 1, oversized, oversized, 1}, {1, 1, 1, oversized, 1}, {1, 1, 1, 1, oversized}, {2, 1, 2, 1, 1}, {2, 1, 2, large_stride, 1}, {2, 2, 1, 1, large_stride}};
  for (const auto& shape : execution_shapes) {
    SCOPED_TRACE(testing::Message() << "M=" << shape.m << " N=" << shape.n << " K=" << shape.k
                                    << " lda=" << shape.lda << " ldc=" << shape.ldc);
    float c = 2.0f;
    MLAS_SGEMM_DATA_PARAMS data{};
    data.A = inaccessible_input;
    data.lda = shape.lda;
    data.B = inaccessible_input;
    data.BIsPacked = true;
    data.C = &c;
    data.ldc = shape.ldc;
    data.alpha = 1.0f;
    data.beta = 0.25f;
    EXPECT_FALSE(ArmKleidiAI::MlasGemmBatchKleidiAIOps(
        CblasNoTrans, CblasTrans, shape.m, shape.n, shape.k, &data, 1, nullptr));
    EXPECT_EQ(c, 2.0f);
  }
}

TEST(SGemmKleidiAIOpsDeathTest, PackingVectorLengthChangesCannotOutgrowSizedBuffers) {
#if defined(__linux__) && defined(__aarch64__) && GTEST_HAS_DEATH_TEST
  struct VectorLengthControl {
    const char* name;
    int get_operation;
    int set_operation;
    int length_mask;
  };
  std::vector<VectorLengthControl> controls;
#if defined(PR_SVE_GET_VL) && defined(PR_SVE_SET_VL) && defined(PR_SVE_VL_LEN_MASK)
  controls.push_back({"SVE", PR_SVE_GET_VL, PR_SVE_SET_VL, PR_SVE_VL_LEN_MASK});
#endif
#if defined(PR_SME_GET_VL) && defined(PR_SME_SET_VL) && defined(PR_SME_VL_LEN_MASK)
  controls.push_back({"SME", PR_SME_GET_VL, PR_SME_SET_VL, PR_SME_VL_LEN_MASK});
#endif
  constexpr size_t M = 3;
  constexpr size_t N = 129;
  constexpr size_t K = 513;
  if (!KleidiAIOpsSgemmIsAvailable(N, K)) {
    GTEST_SKIP() << "KleidiAI Ops SGEMM is unavailable for N=" << N << " K=" << K << ".";
  }
  auto check_packing_change = [](const VectorLengthControl& control, int original_config,
                                 int alternate_config) {
    std::vector<float> a(M * K, 1.0f);
    std::vector<float> b(K * N, 1.0f);
    std::vector<float> c(M * N, 2.0f);
    const size_t full_size = MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, N, K, nullptr);
    std::vector<uint8_t> existing_pack(full_size);
    MlasGemmPackB(CblasNoTrans, CblasNoTrans, N, K, b.data(), N, existing_pack.data(), nullptr);
    if (!ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(existing_pack.data(), N, K)) {
      return 1;
    }
    MLAS_BACKEND_KERNEL_SELECTOR_CONFIG disabled{};
    disabled.use_kleidiai = false;
    const size_t native_only_size = MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, N, K, &disabled);
#if !defined(BUILD_MLAS_NO_ONNXRUNTIME)
    // Workers inherit the creator's VL, while the execution caller retains
    // the VL used for packing. Incompatible output ranges must be replayed
    // on that caller exactly once, including beta=1 accumulation.
    std::unique_ptr<onnxruntime::concurrency::ThreadPool> alternate_pool;
    std::thread pool_creator([&]() {
      if (prctl(control.set_operation, static_cast<unsigned long>(alternate_config), 0UL, 0UL, 0UL) >= 0) {
        alternate_pool = std::make_unique<onnxruntime::concurrency::ThreadPool>(
            &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr, 2, false);
      }
    });
    pool_creator.join();
    if (alternate_pool == nullptr) {
      return 9;
    }
    constexpr size_t WorkerM = 96;
    std::vector<float> worker_a(WorkerM * K, 1.0f);
    std::vector<float> worker_c(WorkerM * N, 2.0f);
    MLAS_SGEMM_DATA_PARAMS worker_data{};
    worker_data.A = worker_a.data();
    worker_data.lda = K;
    worker_data.B = reinterpret_cast<const float*>(existing_pack.data());
    worker_data.BIsPacked = true;
    worker_data.C = worker_c.data();
    worker_data.ldc = N;
    worker_data.alpha = 1.0f;
    worker_data.beta = 1.0f;
    if (!ArmKleidiAI::MlasGemmBatchKleidiAIOps(
            CblasNoTrans, CblasTrans, WorkerM, N, K, &worker_data, 1, alternate_pool.get()) ||
        !std::all_of(worker_c.begin(), worker_c.end(), [](float value) { return value == K + 2.0f; })) {
      return 10;
    }
#endif
    int thread_result = 0;
    std::thread packing_thread([&]() {
      const int changed_config = prctl(control.set_operation, static_cast<unsigned long>(alternate_config), 0UL, 0UL, 0UL);
      if (changed_config < 0 ||
          (changed_config & control.length_mask) == (original_config & control.length_mask)) {
        thread_result = 2;
        return;
      }
      MatrixGuardBuffer<uint8_t> full_guard;
      auto* full_pack = full_guard.GetBuffer(full_size, true);
      MlasGemmPackB(CblasNoTrans, CblasNoTrans, N, K, b.data(), N, full_pack, nullptr);
      if (ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(full_pack, N, K)) {
        thread_result = 3;
        return;
      }
      const size_t mismatched_size = MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, N, K, nullptr);
      if (mismatched_size != native_only_size) {
        thread_result = 4;
        return;
      }
      const int restored_config = prctl(control.set_operation, static_cast<unsigned long>(original_config), 0UL, 0UL, 0UL);
      if (restored_config < 0 ||
          (restored_config & control.length_mask) != (original_config & control.length_mask)) {
        thread_result = 5;
        return;
      }
      // A size queried at the other VL must stay safe after restoring the VL.
      MatrixGuardBuffer<uint8_t> native_guard;
      auto* native_pack = native_guard.GetBuffer(mismatched_size, true);
      MlasGemmPackB(CblasNoTrans, CblasNoTrans, N, K, b.data(), N, native_pack, nullptr);
      if (ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(native_pack, N, K)) {
        thread_result = 6;
      }
    });
    packing_thread.join();
    if (thread_result != 0) {
      return thread_result;
    }
    if (MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, N, K, nullptr) != native_only_size) {
      return 7;
    }
    // Disabling future Ops packing must not invalidate an existing pack.
    MLAS_SGEMM_DATA_PARAMS data{};
    data.A = a.data();
    data.lda = K;
    data.B = reinterpret_cast<const float*>(existing_pack.data());
    data.BIsPacked = true;
    data.C = c.data();
    data.ldc = N;
    data.alpha = 1.0f;
    data.beta = 1.0f;
    return ArmKleidiAI::MlasGemmBatchKleidiAIOps(
               CblasNoTrans, CblasTrans, M, N, K, &data, 1, nullptr) &&
                   std::all_of(c.begin(), c.end(), [](float value) { return value == K + 2.0f; })
               ? 0
               : 8;
  };
  // The sticky packing guard is process-wide. Re-exec isolates its state and
  // avoids using allocator/static locks inherited from MLAS worker threads.
  const auto previous_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  bool exercised = false;
  for (const auto& control : controls) {
    const int original_config = prctl(control.get_operation, 0UL, 0UL, 0UL, 0UL);
    if (original_config < 0) {
      continue;
    }
    int alternate_config = -1;
    std::thread probe([&]() {
      const unsigned long requested_length = (original_config & control.length_mask) == 16 ? 256 : 16;
      alternate_config = prctl(control.set_operation, requested_length, 0UL, 0UL, 0UL);
    });
    probe.join();
    if (alternate_config < 0 ||
        (alternate_config & control.length_mask) == (original_config & control.length_mask)) {
      continue;
    }
    SCOPED_TRACE(control.name);
    exercised = true;
    EXPECT_EXIT(std::_Exit(check_packing_change(control, original_config, alternate_config)),
                testing::ExitedWithCode(0), "");
  }
  GTEST_FLAG_SET(death_test_style, previous_style);
  if (!exercised) {
    GTEST_SKIP() << "The system does not expose two usable SVE or SME vector lengths.";
  }
#else
  GTEST_SKIP() << "Packing VL changes require Linux/Android AArch64 and GoogleTest death-test support.";
#endif
}

TEST(SGemmKleidiAIOpsTest, MixedPackedAndUnpackedBatch) {
  constexpr size_t M = 3;
  constexpr size_t N = 17;
  constexpr size_t K = 13;
  constexpr size_t BatchSize = 3;

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
  ASSERT_EQ(ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(packed_b.data(), N, K),
            KleidiAIOpsSgemmIsAvailable(N, K));

  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG selector_config{};
  selector_config.use_kleidiai = false;
  const size_t generic_packed_size = MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, N, K, &selector_config);
  ASSERT_NE(generic_packed_size, size_t{0});
  std::vector<uint8_t> generic_packed_b(generic_packed_size);
  MlasGemmPackB(CblasNoTrans, CblasNoTrans, N, K, b.data() + K * N, N,
                generic_packed_b.data(), &selector_config);
  ASSERT_FALSE(ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(generic_packed_b.data(), N, K));

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
  constexpr std::array<std::array<float, 2>, 3> scales = {{{1.0f, 1.0f}, {-0.5f, 0.25f}, {0.0f, 0.25f}}};
  for (auto* thread_pool : {static_cast<MLAS_THREADPOOL*>(nullptr), GetMlasThreadPool()}) {
    for (const auto& scale : scales) {
      // Rotations put each representation first and in every batch position.
      for (size_t first = 0; first < BatchSize; ++first) {
        const float alpha = scale[0];
        const float beta = scale[1];
        SCOPED_TRACE(testing::Message() << "first=" << first << " alpha=" << alpha << " beta=" << beta
                                        << " threaded=" << (thread_pool != nullptr));
        std::fill(c.begin(), c.end(), 2.0f);
        MLAS_SGEMM_DATA_PARAMS ordered_data[BatchSize]{
            data[first], data[(first + 1) % BatchSize], data[(first + 2) % BatchSize]};
        for (auto& params : ordered_data) {
          params.alpha = alpha;
          params.beta = beta;
        }
        MlasGemmBatch(CblasNoTrans, CblasNoTrans, M, N, K, ordered_data, BatchSize, thread_pool, nullptr);
        for (size_t i = 0; i < c.size(); ++i) {
          ASSERT_EQ(c[i], alpha * expected[i] + beta * 2.0f) << "index=" << i;
        }
      }
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
