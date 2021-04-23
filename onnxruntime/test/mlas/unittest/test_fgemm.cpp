// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_fgemm.h"
#include "test_fgemm_fixture.h"

#include <memory>
#include <sstream>


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
