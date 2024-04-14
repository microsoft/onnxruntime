/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the MIT License.

Module Name:

    test_sbgemm.cpp

Abstract:

    Tests for MLAS bf16 precision GEMM.

--*/

#if defined(__aarch64__) && defined(__linux__)

#include "test_sbgemm.h"

//
// Short Execute() test helper to register each test seperately by all parameters.
//
template <typename AType, typename BType, bool Packed, bool Threaded>
class SBGemmShortExecuteTest : public MlasTestFixture<MlasSBGemmTest<AType, BType, Packed, Threaded>> {
 public:
  explicit SBGemmShortExecuteTest(size_t M, size_t N, size_t K, size_t Batch, bool hasBias)
      : M_(M), N_(N), K_(K), Batch_(Batch), hasBias_(hasBias) {}

  void TestBody() override {
    MlasTestFixture<MlasSBGemmTest<AType, BType, Packed, Threaded>>::mlas_tester->Test(M_, N_, K_, Batch_, hasBias_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, size_t Batch, bool hasBias) {
    std::stringstream ss;
    ss << "Batch" << Batch << "/M" << M << "xN" << N << "xK" << K << "/"
       << "hasBias" << hasBias;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasSBGemmTest<AType, BType, Packed, Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasSBGemmTest<AType, BType, Packed, Threaded>>* {
          return new SBGemmShortExecuteTest<AType, BType, Packed, Threaded>(
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
    // TODO: check why the cosine similary is < 0.99 for this shape alone
    // test_registered += RegisterSingleTest(43, 500, 401, 1, true);
    test_registered += RegisterSingleTest(1001, 1027, 1031, 1, false);
    if (!Packed) {
      test_registered += RegisterSingleTest(43, 500, 401, 5, true);
      test_registered += RegisterSingleTest(1000, 1029, 1030, 3, false);
    }

    return test_registered;
  }

 private:
  size_t M_, N_, K_, Batch_;
  bool hasBias_;
};

static size_t SBGemmRegistLongExecute() {
  size_t count = 0;

  count += MlasLongExecuteTests<MlasSBGemmTest<float, float, false, false>>::RegisterLongExecute();
  if (MlasSBGemmPackBSize(128, 128) > 0) {
    count += MlasLongExecuteTests<MlasSBGemmTest<float, float, true, false>>::RegisterLongExecute();
  }

  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasSBGemmTest<float, float, false, true>>::RegisterLongExecute();
    if (MlasSBGemmPackBSize(128, 128) > 0) {
      count += MlasLongExecuteTests<MlasSBGemmTest<float, float, true, true>>::RegisterLongExecute();
    }
  }

  return count;
}

static size_t SBGemmRegistShortExecute() {
  size_t count = 0;

  count += SBGemmShortExecuteTest<float, float, false, false>::RegisterShortExecuteTests();
  if (MlasSBGemmPackBSize(128, 128) > 0) {
    count += SBGemmShortExecuteTest<float, float, true, false>::RegisterShortExecuteTests();
  }

  if (GetMlasThreadPool() != nullptr) {
    count += SBGemmShortExecuteTest<float, float, false, true>::RegisterShortExecuteTests();
    if (MlasSBGemmPackBSize(128, 128) > 0) {
      count += SBGemmShortExecuteTest<float, float, true, true>::RegisterShortExecuteTests();
    }
  }

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  if (!MlasBf16AccelerationSupported()) {
    return false;
  }

  if (is_short_execute) {
    return SBGemmRegistShortExecute() > 0;
  }
  return SBGemmRegistLongExecute() > 0;
});
#endif  // defined(__aarch64__) && defined(__linux__)
