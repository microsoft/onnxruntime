
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_qgemm.h"

//
// Short Execute() test helper to register each test separately by all parameters.
//
template <typename AType, typename BType, typename OutputType, bool Packed, bool Threaded>
class QgemmShortExecuteTest;

template <typename AType, typename BType, bool Packed, bool Threaded>
class QgemmShortExecuteTest<AType, BType, int32_t, Packed, Threaded> : public MlasTestFixture<MlasQgemmTest<AType, BType, int32_t, Packed, Threaded>> {
 public:
  explicit QgemmShortExecuteTest(bool use_offb, size_t M, size_t N, size_t K, size_t Batch, uint8_t offa, uint8_t offb)
      : use_offb_(use_offb), M_(M), N_(N), K_(K), Batch_(Batch), offa_(offa), offb_(offb) {
  }

  void TestBody() override {
    if (use_offb_) {
      MlasTestFixture<MlasQgemmTest<AType, BType, int32_t, Packed, Threaded>>::mlas_tester->Test(M_, N_, K_, Batch_, offa_, offb_);
    } else {
      MlasTestFixture<MlasQgemmTest<AType, BType, int32_t, Packed, Threaded>>::mlas_tester->Test(M_, N_, K_, Batch_, offa_);
    }
  }

  static size_t RegisterSingleTest(bool use_offb, size_t M, size_t N, size_t K, size_t Batch, uint8_t offa, uint8_t offb) {
    std::stringstream ss;
    ss << "Batch" << Batch << "/M" << M << "xN" << N << "xK" << K << "/"
       << "offa" << (unsigned)offa << "/"
       << "offb";
    if (use_offb) {
      ss << (unsigned)offb;
    } else {
      ss << "--";
    }
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasQgemmTest<AType, BType, int32_t, Packed, Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasQgemmTest<AType, BType, int32_t, Packed, Threaded>>* {
          return new QgemmShortExecuteTest<AType, BType, int32_t, Packed, Threaded>(
              use_offb, M, N, K, Batch, offa, offb);
        });

    return 1;
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, size_t Batch, uint8_t offa, uint8_t offb) {
    return RegisterSingleTest(true, M, N, K, Batch, offa, offb);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, size_t Batch, uint8_t offa) {
    return RegisterSingleTest(false, M, N, K, Batch, offa, 0);
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    for (size_t b = 1; b < 16; b++) {
      test_registered += RegisterSingleTest(b, b, b, 1, 14, 211);
      test_registered += RegisterSingleTest(b, b, b, 1, 21);
      if (!Packed) {
        test_registered += RegisterSingleTest(b, b, b, 3 + b / 4, 14, 211);
        test_registered += RegisterSingleTest(b, b, b, 2 + b / 4, 21);
      }
    }
    for (size_t b = 1; b < 16; b++) {
      test_registered += RegisterSingleTest(b, b, b, 1, 14, 211);
      test_registered += RegisterSingleTest(b, b, b, 1, 17);
    }
    for (size_t b = 16; b <= 256; b <<= 1) {
      test_registered += RegisterSingleTest(b, b, b, 1, 34, 1);
      test_registered += RegisterSingleTest(b, b, b, 1, 1);
    }
    for (size_t b = 256; b < 320; b += 32) {
      test_registered += RegisterSingleTest(b, b, b, 1, 85, 173);
    }
    for (size_t b = 1; b < 96; b++) {
      test_registered += RegisterSingleTest(1, b, 32, 1, 0, 0);
      test_registered += RegisterSingleTest(1, 32, b, 1, 0, 0);
      test_registered += RegisterSingleTest(1, b, b, 1, 0, 0);
      if (!Packed) {
        test_registered += RegisterSingleTest(1, b, 32, 3, 0, 0);
        test_registered += RegisterSingleTest(1, 32, b, 5, 0, 0);
      }
    }
    // Conv-like qgemm shapes that exposed AMD64 dispatch bugs in the VNNI/AMX paths.
    test_registered += RegisterSingleTest(6, 30, 207, 1, 183, 223);
    test_registered += RegisterSingleTest(6, 30, 207, 1, 17);
    test_registered += RegisterSingleTest(169, 30, 207, 1, 183, 223);
    test_registered += RegisterSingleTest(169, 30, 207, 1, 17);
    test_registered += RegisterSingleTest(43, 500, 401, 1, 183, 223);
    test_registered += RegisterSingleTest(1023, 1023, 1023, 1, 5, 8);
    test_registered += RegisterSingleTest(1023, 1023, 1023, 1, 7);
    if (!Packed) {
      test_registered += RegisterSingleTest(43, 500, 401, 7, 183, 223);
      test_registered += RegisterSingleTest(1023, 1023, 1023, 3, 5, 8);
    }
    size_t dims[] = {400, 500, 1024};
    size_t kdims[] = {1003, 2048 + 50, 4096 - 100};
    for (size_t m : dims) {
      for (size_t n : dims) {
        for (size_t k : kdims) {
          test_registered += RegisterSingleTest(m - 3, n + 5, k, 1, 14, 211);
          test_registered += RegisterSingleTest(m + 5, n - 3, k, 1, 17);
        }
      }
    }

    return test_registered;
  }

 private:
  bool use_offb_;
  size_t M_, N_, K_, Batch_;
  uint8_t offa_, offb_;
};

template <typename AType, typename BType, bool Packed, bool Threaded>
class QgemmShortExecuteTest<AType, BType, float, Packed, Threaded> : public MlasTestFixture<MlasQgemmTest<AType, BType, float, Packed, Threaded>> {
 public:
  explicit QgemmShortExecuteTest(size_t M, size_t N, size_t K, uint8_t offa, uint8_t offb)
      : M_(M), N_(N), K_(K), offa_(offa), offb_(offb) {
  }

  void TestBody() override {
    // Batching code is agnostic to result type. Only cover batches above, not here.
    MlasTestFixture<MlasQgemmTest<AType, BType, float, Packed, Threaded>>::mlas_tester->Test(M_, N_, K_, 1, offa_, offb_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, uint8_t offa, uint8_t offb) {
    std::stringstream ss;
    ss << "M" << M << "xN" << N << "xK" << K << "/"
       << "offa" << (unsigned)offa << "/"
       << "offb" << (unsigned)offb;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasQgemmTest<AType, BType, float, Packed, Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasQgemmTest<AType, BType, float, Packed, Threaded>>* {
          return new QgemmShortExecuteTest<AType, BType, float, Packed, Threaded>(M, N, K, offa, offb);
        });
    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    for (size_t b = 1; b < 16; b++) {
      test_registered += RegisterSingleTest(b, b, b, 34, 46);
    }
    for (size_t b = 16; b <= 256; b <<= 1) {
      test_registered += RegisterSingleTest(b, b, b, 15, 191);
    }
    for (size_t b = 256; b < 320; b += 32) {
      test_registered += RegisterSingleTest(b, b, b, 223, 73);
    }
    for (size_t b = 1; b < 96; b++) {
      test_registered += RegisterSingleTest(1, b, 32, 0, 0);
    }
    test_registered += RegisterSingleTest(169, 30, 207, 183, 223);
    test_registered += RegisterSingleTest(43, 503, 401, 183, 223);
    test_registered += RegisterSingleTest(1024, 1024, 256, 13, 15);

    return test_registered;
  }

 private:
  bool use_offb_;
  size_t M_, N_, K_;
  uint8_t offa_, offb_;
};

//
// Explicit S8U8 coverage for signed A boundary values (-128/-1/0/127); the
// generic buffer fill never produces bytes outside [21,63].
//
template <bool Threaded>
class QgemmS8U8SignedInputTest : public MlasTestFixture<MlasQgemmTest<int8_t, uint8_t, int32_t, false, Threaded>> {
 public:
  QgemmS8U8SignedInputTest(size_t M, size_t N, size_t K, uint8_t offa, uint8_t offb)
      : M_(M), N_(N), K_(K), offa_(offa), offb_(offb) {
  }

  void TestBody() override {
    static const int8_t a_values[] = {-128, -1, 0, 1, 127, -64, 64, -33};
    static const uint8_t b_values[] = {0, 1, 127, 128, 255, 63, 200, 17};

    uint8_t* A = BufferA.GetFilledBuffer(M_ * K_, [](uint8_t* p, size_t n) {
      for (size_t i = 0; i < n; i++) {
        p[i] = uint8_t(a_values[i % _countof(a_values)]);
      }
    });
    uint8_t* B = BufferB.GetFilledBuffer(K_ * N_, [](uint8_t* p, size_t n) {
      for (size_t i = 0; i < n; i++) {
        p[i] = b_values[i % _countof(b_values)];
      }
    });
    int32_t* C = BufferC.GetBuffer(M_ * N_);
    int32_t* CReference = BufferCReference.GetBuffer(M_ * N_);

    MlasTestFixture<MlasQgemmTest<int8_t, uint8_t, int32_t, false, Threaded>>::mlas_tester
        ->Test(M_, N_, K_, 1, A, K_, offa_, B, N_, offb_, C, CReference, N_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, uint8_t offa, uint8_t offb) {
    std::stringstream ss;
    ss << "M" << M << "xN" << N << "xK" << K << "/"
       << "offa" << (unsigned)offa << "/"
       << "offb" << (unsigned)offb;
    auto test_name = ss.str();

    testing::RegisterTest(
        Threaded ? "QGemmS8U8_Int32_SignedInput_Threaded" : "QGemmS8U8_Int32_SignedInput_SingleThread",
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<MlasQgemmTest<int8_t, uint8_t, int32_t, false, Threaded>>* {
          return new QgemmS8U8SignedInputTest<Threaded>(M, N, K, offa, offb);
        });
    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    static const size_t Ks[] = {1, 3, 4, 5, 7, 8, 9, 15, 16, 17};
    static const size_t Ms[] = {1, 2, 3, 4, 7, 8, 9};
    static const uint8_t offas[] = {0, 128, 255};
    static const uint8_t offbs[] = {0, 255};

    for (size_t k = 0; k < _countof(Ks); k++) {
      for (size_t m = 0; m < _countof(Ms); m++) {
        for (size_t a = 0; a < _countof(offas); a++) {
          for (size_t b = 0; b < _countof(offbs); b++) {
            test_registered += RegisterSingleTest(Ms[m], 16, Ks[k], offas[a], offbs[b]);
          }
        }
      }
    }

    return test_registered;
  }

 private:
  MatrixGuardBuffer<uint8_t> BufferA;
  MatrixGuardBuffer<uint8_t> BufferB;
  MatrixGuardBuffer<int32_t> BufferC;
  MatrixGuardBuffer<int32_t> BufferCReference;
  size_t M_, N_, K_;
  uint8_t offa_, offb_;
};
