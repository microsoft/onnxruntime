/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_q4gemm.cpp

Abstract:

    Tests for MLAS GEMM for blockwise int4 quantization.

--*/

#ifndef ORT_MINIMAL_BUILD

#include "test_q4gemm.h"

//
// Short Execute() test helper to register each test separately by all parameters.
//
template <MLAS_BLK_QUANT_TYPE QType, bool Threaded>
class Q4GemmShortExecuteTest : public MlasTestFixture<MlasQ4GemmTest<QType, Threaded>> {
 public:
  explicit Q4GemmShortExecuteTest(size_t M, size_t N, size_t K, bool hasBias)
      : M_(M), N_(N), K_(K), hasBias_(hasBias) {}

  void TestBody() override {
    MlasTestFixture<MlasQ4GemmTest<QType, Threaded>>::mlas_tester->Test(M_, N_, K_, hasBias_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, bool hasBias) {
    std::stringstream ss;
    ss << "/M" << M << "xN" << N << "xK" << K << "/"
       << "hasBias" << hasBias;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasQ4GemmTest<QType, Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasQ4GemmTest<QType, Threaded>>* {
          return new Q4GemmShortExecuteTest<QType, Threaded>(
              M, N, K, hasBias);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    for (size_t b = 1; b < 16; b++) {
      test_registered += RegisterSingleTest(b, b, b, false);
      test_registered += RegisterSingleTest(b, b, b, true);
    }
    for (size_t b = 16; b <= 256; b <<= 1) {
      test_registered += RegisterSingleTest(b, b, b, false);
      test_registered += RegisterSingleTest(b, b, b, true);
    }
    for (size_t b = 256; b < 320; b += 32) {
      test_registered += RegisterSingleTest(b, b, b, true);
    }
    for (size_t b = 1; b < 96; b++) {
      test_registered += RegisterSingleTest(1, b, 32, false);
      test_registered += RegisterSingleTest(1, 32, b, true);
      test_registered += RegisterSingleTest(1, b, b, false);
    }
    test_registered += RegisterSingleTest(43, 500, 401, true);
    // test_registered += RegisterSingleTest(1001, 1027, 1031, 1, false);

    return test_registered;
  }

 private:
  size_t M_, N_, K_;
  bool hasBias_;
};

static size_t Q4GemmRegistShortExecute() {
  size_t count = 0;

  count += Q4GemmShortExecuteTest<BlkQ4Sym, false>::RegisterShortExecuteTests();
  count += Q4GemmShortExecuteTest<BlkQ4Sym, true>::RegisterShortExecuteTests();
  count += Q4GemmShortExecuteTest<BlkQ4Zp8, false>::RegisterShortExecuteTests();
  count += Q4GemmShortExecuteTest<BlkQ4Zp8, true>::RegisterShortExecuteTests();
  count += Q4GemmShortExecuteTest<BlkQ4Sym128, false>::RegisterShortExecuteTests();
  count += Q4GemmShortExecuteTest<BlkQ4Sym128, true>::RegisterShortExecuteTests();

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  if (MlasQ4GemmPackBSize(BlkQ4Sym, 32, 32) == 0) {
    return false;
  }
  if (is_short_execute) {
    return Q4GemmRegistShortExecute() > 0;
  }
  return false;
});

#endif  // ORT_MINIMAL_BUILD
