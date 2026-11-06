// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_fgemm.h"
#include "test_fgemm_fixture.h"

#include <memory>
#include <limits>
#include <sstream>

TEST(FGemmPackB, SizeBounds) {
  const size_t max = (std::numeric_limits<size_t>::max)();
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config{};
  for (bool use_kleidiai : {false, true}) {
    config.use_kleidiai = use_kleidiai;
    EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, max, 1, &config), size_t{0});
    EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 16, max, &config), size_t{0});
    EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 16, max / 16, &config), size_t{0});
  }
  config.use_kleidiai = false;
  const size_t alignment = MlasGetPreferredBufferAlignment();
  const size_t expected = (16 * 3 * sizeof(float) + alignment - 1) & ~(alignment - 1);
  EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 5, 3, &config), expected);
  const size_t largest_k = (max - (alignment - 1)) / (16 * sizeof(float));
  const size_t largest_size = (16 * largest_k * sizeof(float) + alignment - 1) & ~(alignment - 1);
  EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 16, largest_k, &config), largest_size);
  EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 16, largest_k + 1, &config), size_t{0});
  EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 0, 3, &config), size_t{0});
  EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 5, 0, &config), size_t{0});
}

TEST(SGemm, Bias) {
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config{};
  config.use_kleidiai = false;
  MlasFgemmTest<float, true, false> test;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<uint8_t> BufferPackedB;
  constexpr float alpha = 0.25f, beta = 0.5f;
  for (const bool Packed : {false, true}) {
    for (size_t M : {1, 4}) {
      for (const auto TransA : {CblasNoTrans, CblasTrans}) {
        for (size_t N : {32, 33}) {
          // Packed K blocks contain 256 elements; bias only initializes the first.
          for (size_t K : {256, 513}) {
            const size_t lda = TransA == CblasTrans ? M + 1 : K;
            const size_t ldc = N + 1;
            const float* A = test.BufferA.GetBuffer(TransA == CblasTrans ? K * lda : M * lda);
            const float* B = test.BufferB.GetBuffer(K * N);
            const float* Bias = BufferBias.GetBuffer(N);
            float* C = test.BufferC.GetBuffer(M * ldc);
            float* CReference = test.BufferCReference.GetBuffer(M * ldc);
            std::fill_n(C, M * ldc, 2.0f);
            std::fill_n(CReference, M * ldc, 2.0f);

            MLAS_SGEMM_DATA_PARAMS data;
            data.A = A;
            data.lda = lda;
            data.B = B;
            data.ldb = N;
            data.C = C;
            data.ldc = ldc;
            data.alpha = alpha;
            data.beta = beta;
            data.BIsPacked = Packed;
            data.Bias = Bias;
            if (Packed) {
              const size_t PackedBSize = MlasGemmPackBSize(TransA, CblasNoTrans, N, K, &config);
              ASSERT_NE(PackedBSize, size_t{0});
              void* PackedB = BufferPackedB.GetBuffer(PackedBSize, true);
              MlasGemmPackB(TransA, CblasNoTrans, N, K, B, N, PackedB, &config);
              data.B = static_cast<const float*>(PackedB);
            }
            MlasGemm(TransA, CblasNoTrans, M, N, K, data, nullptr, &config);
            test.ReferenceGemm(TransA, CblasNoTrans, M, N, K, 1, alpha, A, lda, B, N, beta, CReference, ldc, Bias);
            for (size_t m = 0; m < M; ++m) {
              // Include padding, which the reference leaves at the sentinel value.
              for (size_t n = 0; n < ldc; ++n) {
                ASSERT_EQ(C[m * ldc + n], CReference[m * ldc + n]) << "m=" << m << " n=" << n;
              }
            }
          }
        }
      }
    }
  }
}

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
