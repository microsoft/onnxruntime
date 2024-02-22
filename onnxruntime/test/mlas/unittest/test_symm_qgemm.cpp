#include "test_symm_qgemm_fixture.h"

static size_t SymmQgemmRegistLongExecute() {
  if (MlasSymmQgemmPackBSize(16, 16, true) == 0) {
    return 0;
  }

  size_t count = MlasLongExecuteTests<MlasSymmQgemmTest<int8_t, int32_t, false>>::RegisterLongExecute();

  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasSymmQgemmTest<int8_t, int32_t, true>>::RegisterLongExecute();
  }

  return count;
}

static size_t SymmQgemmRegistShortExecute() {
  if (MlasSymmQgemmPackBSize(16, 16, true) == 0) {
    return 0;
  }

  size_t count = SymmQgemmShortExecuteTest<int8_t, int32_t, false>::RegisterShortExecuteTests();

  if (GetMlasThreadPool() != nullptr) {
    count += SymmQgemmShortExecuteTest<int8_t, int32_t, true>::RegisterShortExecuteTests();
  }

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? SymmQgemmRegistShortExecute() : SymmQgemmRegistLongExecute();
});