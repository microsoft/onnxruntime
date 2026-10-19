// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_fgemm.h"
#include "test_fgemm_fixture.h"

#include <memory>
#include <sstream>

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
