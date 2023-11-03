/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_q8q4gemm.cpp

Abstract:

    Tests for MLAS int8 x int4 block quantized GEMM.

--*/

#ifndef ORT_MINIMAL_BUILD

#include "test_util.h"
#include "mlas_q4.h"

inline bool
CloseEnough(float actual, float expected) {
  if (std::isnan(actual)) {
    return std::isnan(expected);
  }
  float diff = std::abs(actual - expected);
  float top = std::max(std::abs(actual), std::abs(expected));
  float ratio = 0;
  if (top > 0.0001) {
    ratio = diff / top;
  }
  return ratio < 0.005;
}

template <size_t QBlkLen>
static void blkq8_dequant_reference(const int8_t* src, float* dst, size_t M, size_t K) {
  const size_t num_blks = K / QBlkLen;
  const size_t remain = K % QBlkLen;
  const auto* blob = reinterpret_cast<const int8_t*>(src);

  for (size_t m = 0; m < M; m++) {
    for (size_t i = 0; i < num_blks; i++, dst += QBlkLen) {
      const float scale = *reinterpret_cast<const float*>(blob);
      blob += sizeof(float);
      for (size_t j = 0; j < QBlkLen; ++j) {
        dst[j] = *(blob++) * scale;
      }
    }

    if (remain > 0) {
      const float scale = *reinterpret_cast<const float*>(blob);
      blob += sizeof(float);
      for (size_t j = 0; j < remain; ++j) {
        dst[j] = blob[j] * scale;
      }
      blob += QBlkLen;
      dst += remain;
    }
  }
}

/**
 * @brief Test class for int8 x int4 block quantized GEMM
 *        Note: only 2-D matmul supported for now
 */
template <MLAS_BLK_QUANT_TYPE QType, bool Threaded>
class MlasQ8Q4GemmTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferA;
  MatrixGuardBuffer<int8_t> BufferAQuant;
  MatrixGuardBuffer<float> BufferDequantA;
  MatrixGuardBuffer<float> BufferB;
  MatrixGuardBuffer<uint8_t> BufferBPacked;
  MatrixGuardBuffer<float> BufferUnpack;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<float> BufferC;
  MatrixGuardBuffer<float> BufferCReference;
  MLAS_THREADPOOL* threadpool_;

  void* PackB(size_t N, size_t K, const float* B, size_t ldb) {
    size_t PackedBSize = MlasQ4GemmPackBSize(QType, N, K);
    if (PackedBSize == 0) {
      return nullptr;
    }
    void* PackedB = BufferBPacked.GetBuffer(PackedBSize);
    MlasQ4GemmPackB(QType, PackedB, B, N, K, ldb);
    return PackedB;
  }

  int8_t* QuantizeA(size_t M, size_t K, const float* A, size_t lda) {
    size_t bufsize = MlasQ80BlkQuantSize(QType, M, K);
    if (bufsize == 0) {
      return nullptr;
    }
    auto* QuantA = BufferAQuant.GetBuffer(bufsize);
    MlasQ80BlkQuant(QType, QuantA, A, M, K, lda, threadpool_);
    return QuantA;
  }

  void CallGemm(size_t M,
                size_t N,
                size_t K,
                const int8_t* QuantA,
                const uint8_t* PackedB,
                const float* Bias,
                float* C,
                size_t ldc) {
    MLAS_Q8Q4_GEMM_DATA_PARAMS params;
    params.A = QuantA;
    params.B = PackedB;
    params.Bias = Bias;
    params.C = C;
    params.ldc = ldc;
    params.OutputProcessor = nullptr;

    MlasQ8Q4GemmBatch(QType, M, N, K, 1, &params, threadpool_);
  }

  void ReferenceQgemm(size_t M,
                      size_t N,
                      size_t K,
                      const int8_t* QuantA,
                      const uint8_t* PackedB,
                      const float* Bias,
                      float* C) {
    //    std::vector<float> B(K * N);
    //    MlasQ4GemmUnPackB(QType, B.data(), PackedB, N, K, N);
    float* bdata = BufferUnpack.GetBuffer(K * N);
    MlasQ4GemmUnPackB(QType, bdata, PackedB, N, K, N);

    float* adata = BufferDequantA.GetBuffer(M * K);
    switch (QType) {
      case BlkQ4Sym64:
        blkq8_dequant_reference<64>(QuantA, adata, M, K);
        break;
      case BlkQ4Sym128:
        blkq8_dequant_reference<128>(QuantA, adata, M, K);
        break;
      default:
        blkq8_dequant_reference<32>(QuantA, adata, M, K);
        break;
    }

    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        const float* a = adata + m * K;
        const float* b = bdata + n;
        float* c = C + (m * N) + n;

        float sum = Bias == nullptr ? 0.0f : Bias[n];
        for (size_t k = 0; k < K; k++) {
          sum += (*a) * (*b);
          b += N;
          a += 1;
        }
        *c = sum;
      }
    }
  }

 public:
  MlasQ8Q4GemmTest() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

  void Test(size_t M, size_t N, size_t K, bool withBias) {
    const float* A = BufferA.GetBuffer(K * M);

    const float* B = BufferB.GetBuffer(N * K);

    const float* Bias = nullptr;
    if (withBias) {
      Bias = BufferBias.GetBuffer(N);
    }

    float* C = BufferC.GetBuffer(N * M, true);
    float* CReference = BufferCReference.GetFilledBuffer(
        N * M,
        [](float* start, size_t size) {
          std::fill_n(start, size, -1.0f);
        });
    const uint8_t* PackedB = (uint8_t*)PackB(N, K, B, N);
    const int8_t* QuantA = QuantizeA(M, K, A, K);
    this->CallGemm(M, N, K, QuantA, PackedB, Bias, C, N);
    ReferenceQgemm(M, N, K, QuantA, PackedB, Bias, CReference);
    size_t f = 0;
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++, f++) {
        ASSERT_TRUE(CloseEnough(C[f], CReference[f]))
            << "Expected: " << CReference[f] << " Actual: " << C[f] << "@[" << m << "x" << n << "], "
            << "M=" << M << ", N=" << N << ", K=" << K;
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    /*
          BlkQ4Sym = 0,
          BlkQ4Zp8 = 1,
          BlkQ4Sym64 = 2,
          BlkQ4Sym128 = 4
    */
    static const std::vector<std::string> qtype_names = {"BlkQ4Sym", "BlkQ4Zp8", "BlkQ4Sym64", "", "BlkQ4Sym128"};
    static std::string suite_name = std::string("Q8Q4GemmFP") +
                                    qtype_names[QType] +
                                    (Threaded ? "_Threaded" : "_SingleThread");
    return suite_name.c_str();
  }
};

//
// Short Execute() test helper to register each test separately by all parameters.
//
template <MLAS_BLK_QUANT_TYPE QType, bool Threaded>
class Q8Q4GemmShortExecuteTest : public MlasTestFixture<MlasQ8Q4GemmTest<QType, Threaded>> {
 public:
  explicit Q8Q4GemmShortExecuteTest(size_t M, size_t N, size_t K, bool hasBias)
      : M_(M), N_(N), K_(K), hasBias_(hasBias) {}

  void TestBody() override {
    MlasTestFixture<MlasQ8Q4GemmTest<QType, Threaded>>::mlas_tester->Test(M_, N_, K_, hasBias_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, bool hasBias) {
    std::stringstream ss;
    ss << "/M" << M << "xN" << N << "xK" << K << "/"
       << "hasBias" << hasBias;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasQ8Q4GemmTest<QType, Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasQ8Q4GemmTest<QType, Threaded>>* {
          return new Q8Q4GemmShortExecuteTest<QType, Threaded>(
              M, N, K, hasBias);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    for (size_t b = 1; b < 16; b++) {
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

    return test_registered;
  }

 private:
  size_t M_, N_, K_;
  bool hasBias_;
};

static size_t Q8Q4GemmRegistShortExecute() {
  size_t count = 0;

  count += Q8Q4GemmShortExecuteTest<BlkQ4Sym, false>::RegisterShortExecuteTests();
  count += Q8Q4GemmShortExecuteTest<BlkQ4Sym, true>::RegisterShortExecuteTests();
  count += Q8Q4GemmShortExecuteTest<BlkQ4Zp8, false>::RegisterShortExecuteTests();
  count += Q8Q4GemmShortExecuteTest<BlkQ4Zp8, true>::RegisterShortExecuteTests();
  count += Q8Q4GemmShortExecuteTest<BlkQ4Sym128, false>::RegisterShortExecuteTests();
  count += Q8Q4GemmShortExecuteTest<BlkQ4Sym128, true>::RegisterShortExecuteTests();

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  if (MlasQ80BlkQuantSize(BlkQ4Sym, 32, 32) == 0) {
    return false;  // operation not yet supported on current hardware
  }
  if (is_short_execute) {
    return Q8Q4GemmRegistShortExecute() > 0;
  }
  return false;
});

#endif  // ORT_MINIMAL_BUILD
