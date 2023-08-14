/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_q4gemm.h

Abstract:

    Tests for MLAS int4 block quantized GEMM.

--*/

#pragma once

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

/**
 * @brief Test class for int4 block quantized GEMM
 *        Note: only 2-D matmul supported for now
 */
template <MLAS_BLK_QUANT_TYPE QType, bool Threaded>
class MlasQ4GemmTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<uint8_t> BufferBPacked;
  MatrixGuardBuffer<float> BufferA;
  MatrixGuardBuffer<float> BufferB;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<float> BufferC;
  MatrixGuardBuffer<float> BufferCReference;
  MatrixGuardBuffer<float> BufferUnpack;
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

  void CallGemm(size_t M,
                size_t N,
                size_t K,
                const float* A,
                size_t lda,
                const uint8_t* PackedB,
                const float* Bias,
                float* C,
                size_t ldc) {
    MLAS_Q4_GEMM_DATA_PARAMS params;
    params.A = A;
    params.lda = lda;
    params.Bias = Bias;
    params.C = C;
    params.ldc = ldc;
    params.B = PackedB;
    params.OutputProcessor = nullptr;

    MlasQ4GemmBatch(QType, M, N, K, 1, &params, threadpool_);
  }

  void ReferenceQgemm(size_t M,
                      size_t N,
                      size_t K,
                      const float* A,
                      const uint8_t* PackedB,
                      const float* Bias,
                      float* C) {
    //    std::vector<float> B(K * N);
    //    MlasQ4GemmUnPackB(QType, B.data(), PackedB, N, K, N);
    float* bdata = BufferUnpack.GetBuffer(K * N);
    MlasQ4GemmUnPackB(QType, bdata, PackedB, N, K, N);

    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        const float* a = A + m * K;
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
  MlasQ4GemmTest() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

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
    this->CallGemm(M, N, K, A, K, PackedB, Bias, C, N);
    ReferenceQgemm(M, N, K, A, PackedB, Bias, CReference);
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
    static std::string suite_name = std::string("Q4GemmFP") +
                                    qtype_names[QType] +
                                    (Threaded ? "_Threaded" : "_SingleThread");
    return suite_name.c_str();
  }
};
