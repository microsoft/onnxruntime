
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_qgemm.h"

//
// Short Execute() test helper to register each test seperately by all parameters.
//
template <typename xint8_t, typename OutputType, bool Packed, bool Threaded>
class QgemmShortExecuteTest;

template <typename xint8_t, bool Packed, bool Threaded>
class QgemmShortExecuteTest<xint8_t, int32_t, Packed, Threaded> : public MlasTestFixture<MlasQgemmU8X8Test<xint8_t, int32_t, Packed, Threaded>> {
 public:
  explicit QgemmShortExecuteTest(bool use_offb, size_t M, size_t N, size_t K, uint8_t offa, uint8_t offb)
      : use_offb_(use_offb), M_(M), N_(N), K_(K), offa_(offa), offb_(offb) {
  }

  void TestBody() override {
    if (use_offb_) {
      MlasTestFixture<MlasQgemmU8X8Test<xint8_t, int32_t, Packed, Threaded>>::mlas_tester->Test(M_, N_, K_, offa_, offb_);
    } else {
      MlasTestFixture<MlasQgemmU8X8Test<xint8_t, int32_t, Packed, Threaded>>::mlas_tester->Test(M_, N_, K_, offa_);
    }
  }

  static size_t RegisterSingleTest(bool use_offb, size_t M, size_t N, size_t K, uint8_t offa, uint8_t offb) {
    std::stringstream ss;
    ss << "M" << M << "xN" << N << "xK" << K << "/"
       << "offa" << (unsigned)offa << "/"
       << "offb";
    if (use_offb) {
      ss << (unsigned)offb;
    } else {
      ss << "--";
    }
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasQgemmU8X8Test<xint8_t, int32_t, Packed, Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasQgemmU8X8Test<xint8_t, int32_t, Packed, Threaded>>* {
          return new QgemmShortExecuteTest<xint8_t, int32_t, Packed, Threaded>(
              use_offb, M, N, K, offa, offb);
        });

    return 1;
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, uint8_t offa, uint8_t offb) {
    return RegisterSingleTest(true, M, N, K, offa, offb);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, uint8_t offa) {
    return RegisterSingleTest(false, M, N, K, offa, 0);
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;
    
    for (size_t b = 1; b < 16; b++) {
      test_registered += RegisterSingleTest(b, b, b, 14, 211);
      test_registered += RegisterSingleTest(b, b, b, 21);
    }
    for (size_t b = 1; b < 16; b++) {
      test_registered += RegisterSingleTest(b, b, b, 14, 211);
      test_registered += RegisterSingleTest(b, b, b, 17);
    }
    for (size_t b = 16; b <= 256; b <<= 1) {
      test_registered += RegisterSingleTest(b, b, b, 34, 1);
      test_registered += RegisterSingleTest(b, b, b, 1);
    }
    for (size_t b = 256; b < 320; b += 32) {
      test_registered += RegisterSingleTest(b, b, b, 85, 173);
    }
    for (size_t b = 1; b < 96; b++) {
      test_registered += RegisterSingleTest(1, b, 32, 0, 0);
      test_registered += RegisterSingleTest(1, 32, b, 0, 0);
      test_registered += RegisterSingleTest(1, b, b, 0, 0);
    }
    test_registered += RegisterSingleTest(43, 500, 401, 183, 223);
    test_registered += RegisterSingleTest(1023, 1023, 1023, 5, 8);
    test_registered += RegisterSingleTest(1023, 1023, 1023, 7);

    return test_registered;
  }

 private:
  bool use_offb_;
  size_t M_, N_, K_;
  uint8_t offa_, offb_;
};

template <typename xint8_t, bool Packed, bool Threaded>
class QgemmShortExecuteTest<xint8_t, float, Packed, Threaded> : public MlasTestFixture<MlasQgemmU8X8Test<xint8_t, float, Packed, Threaded>> {
 public:
  explicit QgemmShortExecuteTest(size_t M, size_t N, size_t K, uint8_t offa, uint8_t offb)
      : M_(M), N_(N), K_(K), offa_(offa), offb_(offb) {
  }

  void TestBody() override {
    MlasTestFixture<MlasQgemmU8X8Test<xint8_t, float, Packed, Threaded>>::mlas_tester->Test(M_, N_, K_, offa_, offb_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, uint8_t offa, uint8_t offb) {
    std::stringstream ss;
    ss << "M" << M << "xN" << N << "xK" << K << "/"
       << "offa" << (unsigned)offa << "/"
       << "offb" << (unsigned)offb;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasQgemmU8X8Test<xint8_t, float, Packed, Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasQgemmU8X8Test<xint8_t, float, Packed, Threaded>>* {
          return new QgemmShortExecuteTest<xint8_t, float, Packed, Threaded>(M, N, K, offa, offb);
        });
    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    for (size_t b = 1; b < 16; b++) {
      RegisterSingleTest(b, b, b, 34, 46);
    }
    for (size_t b = 16; b <= 256; b <<= 1) {
      RegisterSingleTest(b, b, b, 15, 191);
    }
    for (size_t b = 256; b < 320; b += 32) {
      RegisterSingleTest(b, b, b, 223, 73);
    }
    for (size_t b = 1; b < 96; b++) {
      RegisterSingleTest(1, b, 32, 0, 0);
    }
    RegisterSingleTest(43, 503, 401, 183, 223);
    RegisterSingleTest(1024, 1024, 256, 13, 15);

    return test_registered;
  }

 private:
  bool use_offb_;
  size_t M_, N_, K_;
  uint8_t offa_, offb_;
};
