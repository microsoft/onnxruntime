// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_MINIMAL_BUILD

#include "test_util.h"
#include "mlas_q4.h"

#define QK8_0 64
typedef struct {
  float d;           // delta
  int8_t qs[QK8_0];  // quants
} block_q8_0;

static void quantize_reference(const float* src, void* dst, size_t M, size_t k) {
  const size_t nb = k / QK8_0;
  block_q8_0* blob = reinterpret_cast<block_q8_0*>(dst);

  for (size_t m = 0; m < M; m++) {
    for (size_t i = 0; i < nb; i++, blob++, src += QK8_0) {
      float amax = 0.0f;  // absolute max

      for (size_t j = 0; j < QK8_0; j++) {
        const float v = src[j];
        amax = std::max(amax, fabsf(v));
      }

      const float d = amax / ((1 << 7) - 1);
      const float id = d ? 1.0f / d : 0.0f;

      blob->d = d;

      for (int j = 0; j < QK8_0; ++j) {
        const float x0 = src[j] * id;

        blob->qs[j] = (int8_t)roundf(x0);
      }
    }

    const size_t remain = k % QK8_0;
    if (remain > 0) {
      float amax = 0.0f;  // absolute max

      for (size_t j = 0; j < remain; j++) {
        const float v = src[j];
        amax = std::max(amax, fabsf(v));
      }

      const float d = amax / 127.f;
      const float id = (amax != 0.0f) ? 127.f / amax : 0.0f;

      blob->d = d;

      for (size_t j = 0; j < remain; ++j) {
        const float x0 = src[j] * id;

        blob->qs[j] = (int8_t)roundf(x0);
      }
      for (size_t j = remain; j < QK8_0; ++j) {
        blob->qs[j] = 0;
      }
      blob++;
      src += remain;
    }
  }
}

template <bool Threaded>
class MlasBlkQ8Test : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> FpInputBuf;
  MatrixGuardBuffer<int8_t> PackedBuf;
  MatrixGuardBuffer<int8_t> ReferenceBuf;
  MLAS_THREADPOOL* threadpool_;

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name = std::string("Q8DQ") +
                                          (Threaded ? "_Threaded" : "_SingleThread");
    return suite_name.c_str();
  }

  void Test(size_t M, size_t K) {
    float* Input = FpInputBuf.GetBuffer(M * K);

    const size_t qsize = MlasQ80BlkQuantSize(BlkQ4Sym64, M, K);
    int8_t* Packed = PackedBuf.GetBuffer(qsize, true);
    int8_t* Ref = ReferenceBuf.GetBuffer(qsize, true);

    MlasQ80BlkQuant(BlkQ4Sym64, Packed, Input, M, K, K, threadpool_);
    quantize_reference(Input, Ref, M, K);

    for (size_t i = 0; i < qsize; i++) {
      ASSERT_EQ(Packed[i], Ref[i]) << ", index=" << i << ", [" << M << "x"
                                   << K << "]";
    }
  }

  MlasBlkQ8Test() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}
};

template <bool Threaded>
class MlasBlkQ8ShortExeTest : public MlasTestFixture<MlasBlkQ8Test<Threaded>> {
 public:
  explicit MlasBlkQ8ShortExeTest(size_t M, size_t K) : M_(M), K_(K) {}

  void TestBody() override {
    MlasTestFixture<MlasBlkQ8Test<Threaded>>::mlas_tester->Test(M_, K_);
  }

  static size_t RegisterSingleTest(size_t M, size_t K) {
    std::stringstream ss;
    ss << "/M" << M << "xK" << K;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasBlkQ8Test<Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasBlkQ8Test<Threaded>>* {
          return new MlasBlkQ8ShortExeTest<Threaded>(
              M, K);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    test_registered += RegisterSingleTest(1, 13);
    test_registered += RegisterSingleTest(1, 20);
    test_registered += RegisterSingleTest(1, 52);
    test_registered += RegisterSingleTest(1, 70);
    test_registered += RegisterSingleTest(3, 13);
    test_registered += RegisterSingleTest(3, 20);
    test_registered += RegisterSingleTest(3, 52);
    test_registered += RegisterSingleTest(3, 70);
    test_registered += RegisterSingleTest(41, 305);
    test_registered += RegisterSingleTest(83, 497);

    return test_registered;
  }

 private:
  size_t M_, K_;
};

template <>
MlasBlkQ8Test<true>* MlasTestFixture<MlasBlkQ8Test<true>>::mlas_tester(nullptr);

template <>
MlasBlkQ8Test<false>* MlasTestFixture<MlasBlkQ8Test<false>>::mlas_tester(nullptr);

static size_t BlkQ8ReisterShortTests() {
  size_t cnt = 0;
  cnt += MlasBlkQ8ShortExeTest<true>::RegisterShortExecuteTests();
  cnt += MlasBlkQ8ShortExeTest<false>::RegisterShortExecuteTests();
  return cnt;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  if (MlasQ80BlkQuantSize(BlkQ4Sym, 32, 32) == 0) {
    return false;  // operation not yet supported on current hardware
  }
  if (is_short_execute) {
    return BlkQ8ReisterShortTests() > 0;
  }
  return false;
});

#endif  // ORT_MINIMAL_BUILD
