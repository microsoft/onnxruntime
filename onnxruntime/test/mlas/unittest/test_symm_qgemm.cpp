#include "test_symm_qgemm_fixture.h"

template <> MlasSymmQgemmTest<int8_t, int32_t, false>* MlasTestFixture<MlasSymmQgemmTest<int8_t, int32_t, false>>::mlas_tester(nullptr);
template <> MlasSymmQgemmTest<int8_t, int32_t, true>* MlasTestFixture<MlasSymmQgemmTest<int8_t, int32_t, true>>::mlas_tester(nullptr);

static size_t SymmQgemmRegistLongExecute() {
  size_t count = 0;

  if (MlasSymmQgemmSupported(true)) {
    count += MlasLongExecuteTests<MlasSymmQgemmTest<int8_t, int32_t, false>>::RegisterLongExecute();

    if (GetMlasThreadPool() != nullptr) {
      count += MlasLongExecuteTests<MlasSymmQgemmTest<int8_t, int32_t, true>>::RegisterLongExecute();
    }
  }

  return count;
}

static size_t SymmQgemmRegistShortExecute() {
  size_t count = 0;

  if (MlasSymmQgemmSupported(true)) {
    count += SymmQgemmShortExecuteTest<int8_t, int32_t, false>::RegisterShortExecuteTests();

    if (GetMlasThreadPool() != nullptr) {
      count += SymmQgemmShortExecuteTest<int8_t, int32_t, true>::RegisterShortExecuteTests();
    }
  }

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? SymmQgemmRegistShortExecute() : SymmQgemmRegistLongExecute();
});