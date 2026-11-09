// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_fgemm.h"
#include "test_fgemm_fixture.h"

#include <cmath>
#include <memory>
#include <limits>
#include <sstream>
#include <vector>

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

static size_t FGemmRegistLongExecute() {
  size_t count = 0;

  count += MlasLongExecuteTests<MlasFgemmTest<float, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasFgemmTest<float, true, false>>::RegisterLongExecute();

  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasFgemmTest<float, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasFgemmTest<float, true, true>>::RegisterLongExecute();
  }

#if defined(MLAS_USE_SVE)
  count += MlasLongExecuteTests<MlasFgemmTest<float, false, false, true>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasFgemmTest<float, true, false, true>>::RegisterLongExecute();
#endif

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

#if defined(MLAS_USE_SVE)
  count += FgemmShortExecuteTest<float, false, false, true>::RegisterShortExecuteTests();
  count += FgemmShortExecuteTest<float, true, false, true>::RegisterShortExecuteTests();
  if (GetMlasThreadPool() != nullptr) {
    count += FgemmShortExecuteTest<float, false, true, true>::RegisterShortExecuteTests();
    count += FgemmShortExecuteTest<float, true, true, true>::RegisterShortExecuteTests();
  }
#endif

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

#if defined(MLAS_USE_SVE)

//
// The SVE small-K kernel must apply alpha to the finished sums: folding it
// into A first overflows here although every result is finite. M == 5 runs
// both the four-row block and the remaining-row loop, and N == 128 is at
// least two vectors at every SVE vector length.
//
TEST(FgemmSve, SmallKAlphaDoesNotOverflow) {
  constexpr size_t M = 5;
  constexpr size_t N = 128;
  const float big = std::ldexp(1.0f, 100);
  const float small = std::ldexp(1.0f, -100);

  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  config.enable_sve_sgemm = true;
  config.use_kleidiai = false;

  for (size_t K : {1u, 3u, 8u}) {
    for (float beta : {0.0f, 1.0f}) {
      std::vector<float> A(M * K, big);
      std::vector<float> B(K * N, small);
      std::vector<float> C(M * N, 0.0f);

      MLAS_SGEMM_DATA_PARAMS data;
      data.A = A.data();
      data.lda = K;
      data.B = B.data();
      data.ldb = N;
      data.C = C.data();
      data.ldc = N;
      data.alpha = big;
      data.beta = beta;
      MlasGemmBatch(CblasNoTrans, CblasNoTrans, M, N, K, &data, 1, nullptr, &config);

      const float expected = static_cast<float>(K) * big;
      for (size_t i = 0; i < M * N; i++) {
        ASSERT_EQ(C[i], expected) << "K=" << K << " beta=" << beta << " m=" << i / N << " n=" << i % N;
      }
    }
  }
}

#endif
