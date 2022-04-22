
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_symm_qgemm.h"

//
// Short Execute() test helper to register each test seperately by all parameters.
//
template <typename AType, typename OutputType, bool Threaded>
class SymmQgemmShortExecuteTest;

template <typename AType, bool Threaded>
class SymmQgemmShortExecuteTest<AType, int32_t, Threaded> : public MlasTestFixture<MlasSymmQgemmTest<AType, int32_t, Threaded>> {
 public:
  explicit SymmQgemmShortExecuteTest(size_t M, size_t N, size_t K, size_t Batch, int32_t offa)
      : M_(M), N_(N), K_(K), Batch_(Batch), offa_(offa) {
  }

  void TestBody() override {
    MlasTestFixture<MlasSymmQgemmTest<AType, int32_t, Threaded>>::mlas_tester->Test(M_, N_, K_, Batch_, offa_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, size_t Batch, int32_t offa) {
    std::stringstream ss;
    ss << "Batch" << Batch << "/M" << M << "xN" << N << "xK" << K << "/"
       << "offa" << offa;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasSymmQgemmTest<AType, int32_t, Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasSymmQgemmTest<AType, int32_t, Threaded>>* {
          return new SymmQgemmShortExecuteTest<AType, int32_t, Threaded>(
              M, N, K, Batch, offa);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    for (size_t b = 1; b < 16; b++) {
      test_registered += RegisterSingleTest(b, b, b, 1, 21);
      test_registered += RegisterSingleTest(b, b, b, 2 + b / 4, -21);
    }
    for (size_t b = 1; b < 16; b++) {
      test_registered += RegisterSingleTest(b, b, b, 1, 17);
    }
    for (size_t b = 16; b <= 256; b <<= 1) {
      test_registered += RegisterSingleTest(b, b, b, 1, -1);
    }
    for (size_t b = 256; b < 320; b += 32) {
      test_registered += RegisterSingleTest(b, b, b, 1, 85);
    }
    for (size_t b = 1; b < 96; b++) {
      test_registered += RegisterSingleTest(1, b, 32, 1, 0);
      test_registered += RegisterSingleTest(1, 32, b, 1, 0);
      test_registered += RegisterSingleTest(1, b, b, 1, 0);
      test_registered += RegisterSingleTest(1, b, 32, 3, 0);
      test_registered += RegisterSingleTest(1, 32, b, 5, 0);      
    }
    test_registered += RegisterSingleTest(43, 500, 401, 7, 113);
    test_registered += RegisterSingleTest(2003, 212, 1020, 3, -5);
    test_registered += RegisterSingleTest(202, 2003, 1023, 3, 15);

    return test_registered;
  }

 private:
  size_t M_, N_, K_, Batch_;
  int32_t offa_;
};

