#include "test_qgemm.h"
#include "test_qgemm_fixture.h"

template <> MlasQgemmU8X8Test<int8_t, int32_t, false, false>* MlasTestFixture<MlasQgemmU8X8Test<int8_t, int32_t, false, false>>::mlas_tester(nullptr);
template <> MlasQgemmU8X8Test<int8_t, int32_t, false, true>* MlasTestFixture<MlasQgemmU8X8Test<int8_t, int32_t, false, true>>::mlas_tester(nullptr);
template <> MlasQgemmU8X8Test<int8_t, int32_t, true, false>* MlasTestFixture<MlasQgemmU8X8Test<int8_t, int32_t, true, false>>::mlas_tester(nullptr);
template <> MlasQgemmU8X8Test<int8_t, int32_t, true, true>* MlasTestFixture<MlasQgemmU8X8Test<int8_t, int32_t, true, true>>::mlas_tester(nullptr);

template <> MlasQgemmU8X8Test<uint8_t, int32_t, false, false>* MlasTestFixture<MlasQgemmU8X8Test<uint8_t, int32_t, false, false>>::mlas_tester(nullptr);
template <> MlasQgemmU8X8Test<uint8_t, int32_t, false, true>* MlasTestFixture<MlasQgemmU8X8Test<uint8_t, int32_t, false, true>>::mlas_tester(nullptr);
template <> MlasQgemmU8X8Test<uint8_t, int32_t, true, false>* MlasTestFixture<MlasQgemmU8X8Test<uint8_t, int32_t, true, false>>::mlas_tester(nullptr);
template <> MlasQgemmU8X8Test<uint8_t, int32_t, true, true>* MlasTestFixture<MlasQgemmU8X8Test<uint8_t, int32_t, true, true>>::mlas_tester(nullptr);

template <> MlasQgemmU8X8Test<int8_t, float, false, false>* MlasTestFixture<MlasQgemmU8X8Test<int8_t, float, false, false>>::mlas_tester(nullptr);
template <> MlasQgemmU8X8Test<int8_t, float, false, true>* MlasTestFixture<MlasQgemmU8X8Test<int8_t, float, false, true>>::mlas_tester(nullptr);
template <> MlasQgemmU8X8Test<int8_t, float, true, false>* MlasTestFixture<MlasQgemmU8X8Test<int8_t, float, true, false>>::mlas_tester(nullptr);
template <> MlasQgemmU8X8Test<int8_t, float, true, true>* MlasTestFixture<MlasQgemmU8X8Test<int8_t, float, true, true>>::mlas_tester(nullptr);

template <> MlasQgemmU8X8Test<uint8_t, float, false, false>* MlasTestFixture<MlasQgemmU8X8Test<uint8_t, float, false, false>>::mlas_tester(nullptr);
template <> MlasQgemmU8X8Test<uint8_t, float, false, true>* MlasTestFixture<MlasQgemmU8X8Test<uint8_t, float, false, true>>::mlas_tester(nullptr);
template <> MlasQgemmU8X8Test<uint8_t, float, true, false>* MlasTestFixture<MlasQgemmU8X8Test<uint8_t, float, true, false>>::mlas_tester(nullptr);
template <> MlasQgemmU8X8Test<uint8_t, float, true, true>* MlasTestFixture<MlasQgemmU8X8Test<uint8_t, float, true, true>>::mlas_tester(nullptr);

static size_t QGemmRegistLongExecute() {
  size_t count = 0;

  count += MlasLongExecuteTests<MlasQgemmU8X8Test<int8_t, int32_t, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmU8X8Test<int8_t, int32_t, true, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmU8X8Test<uint8_t, int32_t, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmU8X8Test<uint8_t, int32_t, true, false>>::RegisterLongExecute();
  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasQgemmU8X8Test<int8_t, int32_t, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmU8X8Test<int8_t, int32_t, true, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmU8X8Test<uint8_t, int32_t, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmU8X8Test<uint8_t, int32_t, true, true>>::RegisterLongExecute();
  }

  return count;
}

static size_t QGemmRegistShortExecute() {
  size_t count = 0;

  count += QgemmShortExecuteTest<int8_t, float, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<uint8_t, float, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<int8_t, int32_t, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<uint8_t, int32_t, false, false>::RegisterShortExecuteTests();
  if (MlasGemmPackBSize(128, 128, false) > 0) {
    // QGEMM U8U8=float packed tests
    count += QgemmShortExecuteTest<uint8_t, float, true, false>::RegisterShortExecuteTests();
    // QGEMM U8u8=int32_t packed tests
    count += QgemmShortExecuteTest<uint8_t, int32_t, true, false>::RegisterShortExecuteTests();
  }
  if (MlasGemmPackBSize(128, 128, true) > 0) {
    // QGEMM U8S8=float packed tests
    count += QgemmShortExecuteTest<int8_t, float, true, false>::RegisterShortExecuteTests();
    // QGEMM U8S8=int32_t packed tests
    count += QgemmShortExecuteTest<int8_t, int32_t, true, false>::RegisterShortExecuteTests();
  }

  if (GetMlasThreadPool() != nullptr) {
    count += QgemmShortExecuteTest<int8_t, float, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<uint8_t, float, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<int8_t, int32_t, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<uint8_t, int32_t, false, true>::RegisterShortExecuteTests();
    if (MlasGemmPackBSize(128, 128, false) > 0) {
      count += QgemmShortExecuteTest<uint8_t, float, true, true>::RegisterShortExecuteTests();
      count += QgemmShortExecuteTest<uint8_t, int32_t, true, true>::RegisterShortExecuteTests();
    }
    if (MlasGemmPackBSize(128, 128, true) > 0) {
      count += QgemmShortExecuteTest<int8_t, float, true, true>::RegisterShortExecuteTests();
      count += QgemmShortExecuteTest<int8_t, int32_t, true, true>::RegisterShortExecuteTests();
    }
  }

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? QGemmRegistShortExecute() : QGemmRegistLongExecute();
});