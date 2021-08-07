
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_fgemm.h"
#include <memory>
#include <sstream>

//
// Short Execute() test helper to register each test seperately by all parameters.
//
template <typename T, bool Packed, bool Threaded>
class FgemmShortExecuteTest : public MlasTestFixture<MlasFgemmTest<T, Packed, Threaded>> {
 public:
  explicit FgemmShortExecuteTest(bool trans_a, bool trans_b, size_t M, size_t N, size_t K, size_t BatchSize, float alpha, float beta)
      : trans_a_(trans_a), trans_b_(trans_b), M_(M), N_(N), K_(K), Batch_(BatchSize), alpha_(alpha), beta_(beta) {
  }

  void TestBody() override {
    MlasTestFixture<MlasFgemmTest<T, Packed, Threaded>>::mlas_tester->Test(
        trans_a_, trans_b_, M_, N_, K_, Batch_, alpha_, beta_);
  }

  static size_t RegisterSingleTest(bool trans_a, bool trans_b, size_t M, size_t N, size_t K, size_t BatchSize, float alpha, float beta) {
    std::stringstream ss;
    ss << (trans_a ? "TransA" : "A") << "/"
       << (trans_b ? "TransB" : "B") << "/"
       << "BatchSize" << BatchSize << "/M" << M << "xN" << N << "xK" << K << "/"
       << "Alpha" << alpha << "/"
       << "Beta" << beta;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasFgemmTest<T, Packed, Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasFgemmTest<T, Packed, Threaded>>* {
          return new FgemmShortExecuteTest<T, Packed, Threaded>(
              trans_a, trans_b, M, N, K, BatchSize, alpha, beta);
        });
    return 1;
  }

  static size_t RegisterTestTransposeABProduct(size_t M, size_t N, size_t K, size_t BatchSize, float alpha, float beta) {
    return RegisterSingleTest(false, false, M, N, K, BatchSize, alpha, beta) +
           RegisterSingleTest(false, true, M, N, K, BatchSize, alpha, beta) +
           RegisterSingleTest(true, false, M, N, K, BatchSize, alpha, beta) +
           RegisterSingleTest(true, true, M, N, K, BatchSize, alpha, beta);
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;
    for (size_t b = 0; b < 16; b++) {
      test_registered += RegisterTestTransposeABProduct(b, b, b, 1, 1.0f, 0.0f);
      test_registered += RegisterTestTransposeABProduct(b, b, b, 3, 1.0f, 0.0f);      
    }
    for (size_t b = 16; b <= 256; b <<= 1) {
      test_registered += RegisterTestTransposeABProduct(b, b, b, 1, 1.0f, 0.0f);
    }
    for (size_t b = 256; b < 320; b += 32) {
      test_registered += RegisterTestTransposeABProduct(b, b, b, 1, 1.0f, 0.0f);
    }

    test_registered += RegisterTestTransposeABProduct(128, 3072, 768, 1, 1.0f, 0.0f);
    test_registered += RegisterTestTransposeABProduct(128, 768, 3072, 1, 1.0f, 0.0f);
    test_registered += RegisterTestTransposeABProduct(25, 81, 79, 7, 1.0f, 0.0f);
    return test_registered;
  }

 private:
  bool trans_a_, trans_b_;
  const size_t M_, N_, K_, Batch_;
  const T alpha_, beta_;
};
