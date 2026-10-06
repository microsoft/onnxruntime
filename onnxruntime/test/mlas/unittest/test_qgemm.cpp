#include "test_qgemm.h"
#include "test_qgemm_fixture.h"

TEST(QGemmPackB, SizeDoesNotIncludeDynamicQGemmPacking) {
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG selector_config{};
  selector_config.use_kleidiai = false;
  const size_t standard_packed_size = MlasGemmPackBSize(
      1024, 1024, true /* AIsSigned */, true /* BIsSigned */, &selector_config);

  selector_config.use_kleidiai = true;
  const size_t packed_size_with_kleidiai_enabled = MlasGemmPackBSize(
      1024, 1024, true /* AIsSigned */, true /* BIsSigned */, &selector_config);

  ASSERT_EQ(packed_size_with_kleidiai_enabled, standard_packed_size);
}

static size_t QGemmRegistLongExecute() {
  size_t count = 0;

  count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, int8_t, int32_t, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, int8_t, int32_t, true, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, uint8_t, int32_t, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, uint8_t, int32_t, true, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<int8_t, int8_t, int32_t, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<int8_t, int8_t, int32_t, true, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<int8_t, uint8_t, int32_t, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasQgemmTest<int8_t, uint8_t, int32_t, true, false>>::RegisterLongExecute();

  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, int8_t, int32_t, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, int8_t, int32_t, true, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, uint8_t, int32_t, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<uint8_t, uint8_t, int32_t, true, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<int8_t, int8_t, int32_t, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<int8_t, int8_t, int32_t, true, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<int8_t, uint8_t, int32_t, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasQgemmTest<int8_t, uint8_t, int32_t, true, true>>::RegisterLongExecute();
  }

  return count;
}

static size_t QGemmRegistShortExecute() {
  size_t count = 0;

  count += QgemmShortExecuteTest<uint8_t, int8_t, float, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<uint8_t, uint8_t, float, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<uint8_t, int8_t, int32_t, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<uint8_t, uint8_t, int32_t, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<int8_t, int8_t, float, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<int8_t, int8_t, int32_t, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<int8_t, uint8_t, float, false, false>::RegisterShortExecuteTests();
  count += QgemmShortExecuteTest<int8_t, uint8_t, int32_t, false, false>::RegisterShortExecuteTests();
  count += QgemmS8U8SignedInputTest<false>::RegisterShortExecuteTests();
  if (MlasGemmPackBSize(128, 128, false /*AIsSigned*/, false /*BIsSigned*/, nullptr) > 0) {
    // QGEMM U8U8=float packed tests
    count += QgemmShortExecuteTest<uint8_t, uint8_t, float, true, false>::RegisterShortExecuteTests();
    // QGEMM U8U8=int32_t packed tests
    count += QgemmShortExecuteTest<uint8_t, uint8_t, int32_t, true, false>::RegisterShortExecuteTests();
  }
  if (MlasGemmPackBSize(128, 128, false /*AIsSigned*/, true /*BIsSigned*/, nullptr) > 0) {
    // QGEMM U8S8=float packed tests
    count += QgemmShortExecuteTest<uint8_t, int8_t, float, true, false>::RegisterShortExecuteTests();
    // QGEMM U8S8=int32_t packed tests
    count += QgemmShortExecuteTest<uint8_t, int8_t, int32_t, true, false>::RegisterShortExecuteTests();
  }
  if (MlasGemmPackBSize(128, 128, true /*AIsSigned*/, true /*BIsSigned*/, nullptr) > 0) {
    // QGEMM S8S8=float packed tests
    count += QgemmShortExecuteTest<int8_t, int8_t, float, true, false>::RegisterShortExecuteTests();
    // QGEMM S8S8=int32_t packed tests
    count += QgemmShortExecuteTest<int8_t, int8_t, int32_t, true, false>::RegisterShortExecuteTests();
  }
  if (MlasGemmPackBSize(128, 128, true /*AIsSigned*/, false /*BIsSigned*/, nullptr) > 0) {
    // QGEMM S8U8=float packed tests
    count += QgemmShortExecuteTest<int8_t, uint8_t, float, true, false>::RegisterShortExecuteTests();
    // QGEMM S8U8=int32_t packed tests
    count += QgemmShortExecuteTest<int8_t, uint8_t, int32_t, true, false>::RegisterShortExecuteTests();
  }

  if (GetMlasThreadPool() != nullptr) {
    count += QgemmShortExecuteTest<uint8_t, int8_t, float, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<uint8_t, uint8_t, float, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<uint8_t, int8_t, int32_t, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<uint8_t, uint8_t, int32_t, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<int8_t, int8_t, float, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<int8_t, int8_t, int32_t, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<int8_t, uint8_t, float, false, true>::RegisterShortExecuteTests();
    count += QgemmShortExecuteTest<int8_t, uint8_t, int32_t, false, true>::RegisterShortExecuteTests();
    count += QgemmS8U8SignedInputTest<true>::RegisterShortExecuteTests();
    if (MlasGemmPackBSize(128, 128, false /*AIsSigned*/, false /*BIsSigned*/, nullptr) > 0) {
      count += QgemmShortExecuteTest<uint8_t, uint8_t, float, true, true>::RegisterShortExecuteTests();
      count += QgemmShortExecuteTest<uint8_t, uint8_t, int32_t, true, true>::RegisterShortExecuteTests();
    }
    if (MlasGemmPackBSize(128, 128, false /*AIsSigned*/, true /*BIsSigned*/, nullptr) > 0) {
      count += QgemmShortExecuteTest<uint8_t, int8_t, float, true, true>::RegisterShortExecuteTests();
      count += QgemmShortExecuteTest<uint8_t, int8_t, int32_t, true, true>::RegisterShortExecuteTests();
    }
    if (MlasGemmPackBSize(128, 128, true /*AIsSigned*/, true /*BIsSigned*/, nullptr) > 0) {
      count += QgemmShortExecuteTest<int8_t, int8_t, float, true, true>::RegisterShortExecuteTests();
      count += QgemmShortExecuteTest<int8_t, int8_t, int32_t, true, true>::RegisterShortExecuteTests();
    }
    if (MlasGemmPackBSize(128, 128, true /*AIsSigned*/, false /*BIsSigned*/, nullptr) > 0) {
      count += QgemmShortExecuteTest<int8_t, uint8_t, float, true, true>::RegisterShortExecuteTests();
      count += QgemmShortExecuteTest<int8_t, uint8_t, int32_t, true, true>::RegisterShortExecuteTests();
    }
  }

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? QGemmRegistShortExecute() : QGemmRegistLongExecute();
});