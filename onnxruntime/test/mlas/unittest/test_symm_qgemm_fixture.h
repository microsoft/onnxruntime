
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_symm_qgemm.h"

//
// Short Execute() test helper to register each test separately by all parameters.
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

//
// Explicit S8S8 coverage for int8 extreme values on both A and B; the
// generic buffer fill never produces bytes outside [21,63].
//
template <bool Threaded>
class SymmQgemmS8SignedInputTest : public MlasTestFixture<MlasSymmQgemmTest<int8_t, int32_t, Threaded>> {
 public:
  SymmQgemmS8SignedInputTest(size_t M, size_t N, size_t K, int32_t offa)
      : M_(M), N_(N), K_(K), offa_(offa) {
  }

  void TestBody() override {
    static const int8_t a_values[] = {-128, -1, 0, 1, 127, -64, 64, -33};
    static const int8_t b_values[] = {-128, -1, 0, 1, 127, -100, 100, 42};

    // Symmetric kernels may read up to 15 bytes past the logical end of A.
    constexpr size_t OVERRUN = 15;
    uint8_t* A = BufferA.GetFilledBuffer(M_ * K_ + OVERRUN, [](uint8_t* p, size_t n) {
      for (size_t i = 0; i < n; i++) {
        p[i] = uint8_t(a_values[i % _countof(a_values)]);
      }
    });
    int8_t* B = BufferB.GetFilledBuffer(K_ * N_, [](int8_t* p, size_t n) {
      for (size_t i = 0; i < n; i++) {
        p[i] = b_values[i % _countof(b_values)];
      }
    });
    int32_t* C = BufferC.GetBuffer(M_ * N_);
    int32_t* CReference = BufferCReference.GetBuffer(M_ * N_);

    MlasTestFixture<MlasSymmQgemmTest<int8_t, int32_t, Threaded>>::mlas_tester
        ->Test(M_, N_, K_, 1, A, K_, offa_, B, N_, C, CReference, N_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, int32_t offa) {
    std::stringstream ss;
    ss << "M" << M << "xN" << N << "xK" << K << "/"
       << "offa" << offa;
    auto test_name = ss.str();

    testing::RegisterTest(
        Threaded ? "SymmQGemmS8_Int32_SignedInput_Threaded" : "SymmQGemmS8_Int32_SignedInput_SingleThread",
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<MlasSymmQgemmTest<int8_t, int32_t, Threaded>>* {
          return new SymmQgemmS8SignedInputTest<Threaded>(M, N, K, offa);
        });
    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    static const size_t Ks[] = {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33};
    static const size_t Ms[] = {1, 2, 3, 4, 5, 7, 8, 9};
    static const size_t Ns[] = {16, 32};
    static const int32_t offas[] = {0, -128, 127};

    for (size_t k = 0; k < _countof(Ks); k++) {
      for (size_t m = 0; m < _countof(Ms); m++) {
        for (size_t n = 0; n < _countof(Ns); n++) {
          for (size_t a = 0; a < _countof(offas); a++) {
            test_registered += RegisterSingleTest(Ms[m], Ns[n], Ks[k], offas[a]);
          }
        }
      }
    }

    return test_registered;
  }

 private:
  MatrixGuardBuffer<uint8_t> BufferA;
  MatrixGuardBuffer<int8_t> BufferB;
  MatrixGuardBuffer<int32_t> BufferC;
  MatrixGuardBuffer<int32_t> BufferCReference;
  size_t M_, N_, K_;
  int32_t offa_;
};
