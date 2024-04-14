/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_halfgemm.cpp

Abstract:

    Tests for MLAS half precision GEMM.

--*/

#include "test_halfgemm.h"

//
// Short Execute() test helper to register each test seperately by all parameters.
//
template <typename AType, typename BType, bool Packed, bool Threaded>
class HalfGemmShortExecuteTest : public MlasTestFixture<MlasHalfGemmTest<AType, BType, Packed, Threaded>> {
 public:
  explicit HalfGemmShortExecuteTest(size_t M, size_t N, size_t K, size_t Batch, bool hasBias)
      : M_(M), N_(N), K_(K), Batch_(Batch), hasBias_(hasBias) {}

  void TestBody() override {
    MlasTestFixture<MlasHalfGemmTest<AType, BType, Packed, Threaded>>::mlas_tester->Test(M_, N_, K_, Batch_, hasBias_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, size_t Batch, bool hasBias) {
    std::stringstream ss;
    ss << "Batch" << Batch << "/M" << M << "xN" << N << "xK" << K << "/"
       << "hasBias" << hasBias;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasHalfGemmTest<AType, BType, Packed, Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasHalfGemmTest<AType, BType, Packed, Threaded>>* {
          return new HalfGemmShortExecuteTest<AType, BType, Packed, Threaded>(
              M, N, K, Batch, hasBias);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    for (size_t b = 1; b < 16; b++) {
      test_registered += RegisterSingleTest(b, b, b, 1, false);
      test_registered += RegisterSingleTest(b, b, b, 1, true);
    }
    for (size_t b = 16; b <= 256; b <<= 1) {
      test_registered += RegisterSingleTest(b, b, b, 1, false);
      test_registered += RegisterSingleTest(b, b, b, 1, true);
    }
    for (size_t b = 256; b < 320; b += 32) {
      test_registered += RegisterSingleTest(b, b, b, 1, true);
    }
    for (size_t b = 1; b < 96; b++) {
      test_registered += RegisterSingleTest(1, b, 32, 1, false);
      test_registered += RegisterSingleTest(1, 32, b, 1, true);
      test_registered += RegisterSingleTest(1, b, b, 1, false);
      if (!Packed) {
        test_registered += RegisterSingleTest(1, b, 32, 3, true);
        test_registered += RegisterSingleTest(1, 32, b, 5, false);
      }
    }
    test_registered += RegisterSingleTest(43, 500, 401, 1, true);
    //    test_registered += RegisterSingleTest(1001, 1027, 1031, 1, false);
    if (!Packed) {
      test_registered += RegisterSingleTest(43, 500, 401, 5, true);
      //      test_registered += RegisterSingleTest(1000, 1029, 1030, 3, false);
    }

    return test_registered;
  }

 private:
  size_t M_, N_, K_, Batch_;
  bool hasBias_;
};

static size_t HalfGemmRegistLongExecute() {
  size_t count = 0;

  count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, float, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasHalfGemmTest<float, MLFp16, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, MLFp16, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasHalfGemmTest<float, float, false, false>>::RegisterLongExecute();
  if (MlasHalfGemmPackBSize(128, 128, false) > 0) {
    count += MlasLongExecuteTests<MlasHalfGemmTest<float, MLFp16, true, false>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, MLFp16, true, false>>::RegisterLongExecute();
  }
  if (MlasHalfGemmPackBSize(128, 128, true) > 0) {
    count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, float, true, false>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasHalfGemmTest<float, float, true, false>>::RegisterLongExecute();
  }

  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, float, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasHalfGemmTest<float, MLFp16, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, MLFp16, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasHalfGemmTest<float, float, false, true>>::RegisterLongExecute();
    if (MlasHalfGemmPackBSize(128, 128, false) > 0) {
      count += MlasLongExecuteTests<MlasHalfGemmTest<float, MLFp16, true, true>>::RegisterLongExecute();
      count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, MLFp16, true, true>>::RegisterLongExecute();
    }
    if (MlasHalfGemmPackBSize(128, 128, true) > 0) {
      count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, float, true, true>>::RegisterLongExecute();
      count += MlasLongExecuteTests<MlasHalfGemmTest<float, float, true, true>>::RegisterLongExecute();
    }
  }

  return count;
}

static size_t HalfGemmRegistShortExecute() {
  size_t count = 0;

  count += HalfGemmShortExecuteTest<float, float, false, false>::RegisterShortExecuteTests();
  count += HalfGemmShortExecuteTest<MLFp16, float, false, false>::RegisterShortExecuteTests();
  count += HalfGemmShortExecuteTest<float, MLFp16, false, false>::RegisterShortExecuteTests();
  count += HalfGemmShortExecuteTest<MLFp16, MLFp16, false, false>::RegisterShortExecuteTests();
  if (MlasHalfGemmPackBSize(128, 128, false) > 0) {
    count += HalfGemmShortExecuteTest<MLFp16, MLFp16, true, false>::RegisterShortExecuteTests();
    count += HalfGemmShortExecuteTest<float, MLFp16, true, false>::RegisterShortExecuteTests();
  }
  if (MlasHalfGemmPackBSize(128, 128, true) > 0) {
    count += HalfGemmShortExecuteTest<MLFp16, float, true, false>::RegisterShortExecuteTests();
    count += HalfGemmShortExecuteTest<float, float, true, false>::RegisterShortExecuteTests();
  }

  if (GetMlasThreadPool() != nullptr) {
    count += HalfGemmShortExecuteTest<float, float, false, true>::RegisterShortExecuteTests();
    count += HalfGemmShortExecuteTest<MLFp16, float, false, true>::RegisterShortExecuteTests();
    count += HalfGemmShortExecuteTest<float, MLFp16, false, true>::RegisterShortExecuteTests();
    count += HalfGemmShortExecuteTest<MLFp16, MLFp16, false, true>::RegisterShortExecuteTests();
    if (MlasHalfGemmPackBSize(128, 128, false) > 0) {
      count += HalfGemmShortExecuteTest<MLFp16, MLFp16, true, true>::RegisterShortExecuteTests();
      count += HalfGemmShortExecuteTest<float, MLFp16, true, true>::RegisterShortExecuteTests();
    }
    if (MlasHalfGemmPackBSize(128, 128, true) > 0) {
      count += HalfGemmShortExecuteTest<MLFp16, float, true, true>::RegisterShortExecuteTests();
      count += HalfGemmShortExecuteTest<float, float, true, true>::RegisterShortExecuteTests();
    }
  }

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  if (!MlasFp16AccelerationSupported()) {
    return false;
  }
  if (is_short_execute) {
    return HalfGemmRegistShortExecute() > 0;
  }
  return HalfGemmRegistLongExecute() > 0;
});
