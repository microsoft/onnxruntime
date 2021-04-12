// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_fgemm.h"
#include "test_fgemm_fixture.h"

#include <memory>
#include <sstream>


template <>
void FgemmPackedContext<float, false>::TestGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    size_t BatchSize,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc,
    MLAS_THREADPOOL* threadpool) {
  std::vector<MLAS_SGEMM_DATA_PARAMS> data(BatchSize);
  for (size_t i = 0; i < BatchSize; i++) {
    data[i].A = A + M * K * i;
    data[i].lda = lda;
    data[i].B = B + K * N * i;
    data[i].ldb = ldb;
    data[i].C = C + M * N * i;
    data[i].ldc = ldc;
    data[i].alpha = alpha;
    data[i].beta = beta;
  }
  MlasGemmBatch(TransA, TransB, M, N, K, data.data(), BatchSize, threadpool);
}

#ifdef MLAS_TARGET_AMD64

template <>
void FgemmPackedContext<double, false>::TestGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    size_t BatchSize,
    double alpha,
    const double* A,
    size_t lda,
    const double* B,
    size_t ldb,
    double beta,
    double* C,
    size_t ldc,
    MLAS_THREADPOOL* threadpool) {
  std::vector<MLAS_DGEMM_DATA_PARAMS> data(BatchSize);
  for (size_t i = 0; i < BatchSize; i++) {
    data[i].A = A + M * K * i;
    data[i].lda = lda;
    data[i].B = B + K * N * i;
    data[i].ldb = ldb;
    data[i].C = C + M * N * i;
    data[i].ldc = ldc;
    data[i].alpha = alpha;
    data[i].beta = beta;
  }
  MlasGemmBatch(TransA, TransB, M, N, K, data.data(), BatchSize, threadpool);
}
#endif


template <> MlasFgemmTest<float, false, false>* MlasTestFixture<MlasFgemmTest<float, false, false>>::mlas_tester(nullptr);
template <> MlasFgemmTest<float, false, true>* MlasTestFixture<MlasFgemmTest<float, false, true>>::mlas_tester(nullptr);
template <> MlasFgemmTest<float, true, false>* MlasTestFixture<MlasFgemmTest<float, true, false>>::mlas_tester(nullptr);
template <> MlasFgemmTest<float, true, true>* MlasTestFixture<MlasFgemmTest<float, true, true>>::mlas_tester(nullptr);

#ifdef MLAS_SUPPORTS_GEMM_DOUBLE

template <> MlasFgemmTest<double, false, false>* MlasTestFixture<MlasFgemmTest<double, false, false>>::mlas_tester(nullptr);
template <> MlasFgemmTest<double, false, true>* MlasTestFixture<MlasFgemmTest<double, false, true>>::mlas_tester(nullptr);

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
