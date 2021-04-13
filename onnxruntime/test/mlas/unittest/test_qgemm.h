// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_util.h"

template <bool Packed, bool Threaded>
class MlasQgemmU8X8U8X8TestBase : public MlasTestBase {
 private:
  void* PackB(size_t N, size_t K, const uint8_t* B, size_t ldb, bool BIsSigned) {
    size_t PackedBSize = MlasGemmPackBSize(N, K, BIsSigned);
    void* PackedB = BufferBPacked.GetBuffer(PackedBSize);
    MlasGemmPackB(N, K, B, ldb, BIsSigned, PackedB);
    return PackedB;
  }

 protected:
  MLAS_THREADPOOL* threadpool_;

  MlasQgemmU8X8U8X8TestBase() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

  void TestGemm(size_t M,
                size_t N,
                size_t K,
                const uint8_t* A,
                size_t lda,
                uint8_t offa,
                const uint8_t* B,
                size_t ldb,
                uint8_t offb,
                bool BIsSigned,
                int32_t* C,
                size_t ldc) {
    MLAS_GEMM_U8X8_PARAMETERS GemmParameters;

    GemmParameters.M = M;
    GemmParameters.N = N;
    GemmParameters.K = K;
    GemmParameters.A = A;
    GemmParameters.lda = lda;
    GemmParameters.ZeroPointA = offa;
    GemmParameters.ZeroPointB = &offb;
    GemmParameters.BIsSigned = BIsSigned;
    GemmParameters.C = C;
    GemmParameters.ldc = ldc;

    if (Packed) {
      GemmParameters.B = PackB(N, K, B, ldb, BIsSigned);
      GemmParameters.BIsPacked = true;
    } else {
      GemmParameters.B = B;
      GemmParameters.ldb = ldb;
    }

    MlasGemm(&GemmParameters, threadpool_);
  }

  void TestGemm(size_t M,
                size_t N,
                size_t K,
                const uint8_t* A,
                size_t lda,
                uint8_t offa,
                const uint8_t* B,
                size_t ldb,
                const uint8_t* offb,
                bool BIsSigned,
                int32_t* C,
                size_t ldc) {
    MLAS_GEMM_U8X8_PARAMETERS GemmParameters;

    GemmParameters.M = M;
    GemmParameters.N = N;
    GemmParameters.K = K;
    GemmParameters.A = A;
    GemmParameters.lda = lda;
    GemmParameters.ZeroPointA = offa;
    GemmParameters.ZeroPointB = offb;
    GemmParameters.BIsSigned = BIsSigned;
    GemmParameters.PerColumnZeroPoints = true;
    GemmParameters.C = C;
    GemmParameters.ldc = ldc;

    if (Packed) {
      GemmParameters.B = PackB(N, K, B, ldb, BIsSigned);
      GemmParameters.BIsPacked = true;
    } else {
      GemmParameters.B = B;
      GemmParameters.ldb = ldb;
    }

    MlasGemm(&GemmParameters, threadpool_);
  }

  void TestGemm(size_t M,
                size_t N,
                size_t K,
                const uint8_t* A,
                size_t lda,
                uint8_t offa,
                const uint8_t* B,
                size_t ldb,
                uint8_t offb,
                bool BIsSigned,
                float* C,
                size_t ldc,
                float CScale,
                const float* Bias) {
    MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR ScaleBiasProcessor(C, ldc, &CScale, Bias);

    MLAS_GEMM_U8X8_PARAMETERS GemmParameters;

    GemmParameters.M = M;
    GemmParameters.N = N;
    GemmParameters.K = K;
    GemmParameters.A = A;
    GemmParameters.lda = lda;
    GemmParameters.ZeroPointA = offa;
    GemmParameters.ZeroPointB = &offb;
    GemmParameters.BIsSigned = BIsSigned;
    GemmParameters.C = reinterpret_cast<int32_t*>(C);
    GemmParameters.ldc = ldc;
    GemmParameters.OutputProcessor = &ScaleBiasProcessor;

    if (Packed) {
      GemmParameters.B = PackB(N, K, B, ldb, BIsSigned);
      GemmParameters.BIsPacked = true;
    } else {
      GemmParameters.B = B;
      GemmParameters.ldb = ldb;
    }

    MlasGemm(&GemmParameters, threadpool_);
  }

 private:
  MatrixGuardBuffer<uint8_t> BufferBPacked;
};

template <typename xint8_t, typename OutputType, bool Packed, bool Threaded>
class MlasQgemmU8X8Test;

template <typename xint8_t, bool Packed, bool Threaded>
class MlasQgemmU8X8Test<xint8_t, int32_t, Packed, Threaded> : public MlasQgemmU8X8U8X8TestBase<Packed, Threaded> {
 public:
  void Test(size_t M, size_t N, size_t K, uint8_t offa, uint8_t offb) {
    const uint8_t* A = BufferA.GetBuffer(K * M);
    const uint8_t* B = BufferB.GetBuffer(N * K);
    int32_t* C = BufferC.GetBuffer(N * M);
    int32_t* CReference = BufferCReference.GetBuffer(N * M);

    Test(M, N, K, A, K, offa, B, N, offb, C, CReference, N);
  }

  void Test(size_t M, size_t N, size_t K, uint8_t offa) {
    const uint8_t* A = BufferA.GetBuffer(K * M);
    const uint8_t* B = BufferB.GetBuffer(N * K);
    const uint8_t* ZeroPointB = BufferZeroPointB.GetBuffer(N);
    int32_t* C = BufferC.GetBuffer(N * M);
    int32_t* CReference = BufferCReference.GetBuffer(N * M);

    Test(M, N, K, A, K, offa, B, N, ZeroPointB, C, CReference, N);
  }

  void Test(size_t M,
            size_t N,
            size_t K,
            const uint8_t* A,
            size_t lda,
            uint8_t offa,
            const uint8_t* B,
            size_t ldb,
            uint8_t offb,
            int32_t* C,
            int32_t* CReference,
            size_t ldc) {
    std::fill_n(C, M * N, -1);
    std::fill_n(CReference, M * N, -1);

    this->TestGemm(M, N, K, A, lda, offa, B, ldb, offb, BIsSigned, C, ldc);
    ReferenceQgemm(M, N, K, A, lda, offa, (const xint8_t*)B, ldb, (xint8_t)offb, CReference, ldc);

    for (size_t m = 0, f = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++, f++) {
        ASSERT_EQ(C[f], CReference[f]) << "@[" << m << "x" << n << "], "
                                       << "M=" << M << ", N=" << N << ", K=" << K
                                       << ", offa=" << int(offa) << ", offb=" << offb;
      }
    }
  }

  void Test(size_t M,
            size_t N,
            size_t K,
            const uint8_t* A,
            size_t lda,
            uint8_t offa,
            const uint8_t* B,
            size_t ldb,
            const uint8_t* offb,
            int32_t* C,
            int32_t* CReference,
            size_t ldc) {
    std::fill_n(C, M * N, -1);
    std::fill_n(CReference, M * N, -1);

    this->TestGemm(M, N, K, A, lda, offa, B, ldb, offb, BIsSigned, C, ldc);
    ReferenceQgemm(M, N, K, A, lda, offa, (const xint8_t*)B, ldb, (const xint8_t*)offb, CReference, ldc);

    for (size_t m = 0, f = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++, f++) {
        ASSERT_EQ(C[f], CReference[f]) << "@[" << m << "x" << n << "], "
                                       << "M=" << M << ", N=" << N << ", K=" << K << ", offa=" << int(offa);
      }
    }
  }

 private:
  void ReferenceQgemm(size_t M,
                      size_t N,
                      size_t K,
                      const uint8_t* A,
                      size_t lda,
                      uint8_t offa,
                      const xint8_t* B,
                      size_t ldb,
                      xint8_t offb,
                      int32_t* C,
                      size_t ldc) {
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        const uint8_t* a = A + (m * lda);
        const xint8_t* b = B + n;
        int32_t* c = C + (m * ldc) + n;
        int32_t sum = 0;

        for (size_t k = 0; k < K; k++) {
          sum += ((int32_t(*b) - offb) * (int32_t(*a) - offa));
          b += ldb;
          a += 1;
        }

        *c = sum;
      }
    }
  }

  void ReferenceQgemm(size_t M,
                      size_t N,
                      size_t K,
                      const uint8_t* A,
                      size_t lda,
                      uint8_t offa,
                      const xint8_t* B,
                      size_t ldb,
                      const xint8_t* offb,
                      int32_t* C,
                      size_t ldc) {
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        const uint8_t* a = A + (m * lda);
        const xint8_t* b = B + n;
        int32_t* c = C + (m * ldc) + n;
        int32_t sum = 0;

        for (size_t k = 0; k < K; k++) {
          sum += ((int32_t(*b) - offb[n]) * (int32_t(*a) - offa));
          b += ldb;
          a += 1;
        }

        *c = sum;
      }
    }
  }

  MatrixGuardBuffer<uint8_t> BufferA;
  MatrixGuardBuffer<uint8_t> BufferB;
  MatrixGuardBuffer<uint8_t> BufferZeroPointB;
  MatrixGuardBuffer<int32_t> BufferC;
  MatrixGuardBuffer<int32_t> BufferCReference;
  const bool BIsSigned = std::is_signed<xint8_t>::value;

 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("QGemmU8") +
                                    (std::is_signed<xint8_t>::value ? "S8" : "U8") +
                                    (Packed ? "_Int32_Packed" : "_Int32_NoPack") +
                                    (Threaded ? "_Threaded" : "_SingleThread");
    return suite_name.c_str();
  }

  void ExecuteLong(void) override {
    static const uint8_t zero_points[] = {0, 18, 75, 128, 157, 231, 255};

    for (size_t a = 0; a < _countof(zero_points); a++) {
      uint8_t offa = zero_points[a];

      for (size_t b = 0; b < _countof(zero_points); b++) {
        uint8_t offb = zero_points[b];

        for (size_t M = 16; M < 160; M += 32) {
          for (size_t N = 16; N < 160; N += 32) {
            static const size_t ks[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 20, 32, 48, 64, 118, 119, 120, 121, 122, 160, 240, 320};
            for (size_t k = 0; k < _countof(ks); k++) {
              size_t K = ks[k];

              Test(M, N, K, offa, offb);
              Test(M + 1, N, K, offa, offb);
              Test(M, N + 1, K, offa, offb);
              Test(M + 1, N + 1, K, offa, offb);
              Test(M + 3, N + 2, K, offa, offb);
              Test(M + 4, N, K, offa, offb);
              Test(M, N + 4, K, offa, offb);
              Test(M + 4, N + 4, K, offa, offb);
              Test(M + 3, N + 7, K, offa, offb);
              Test(M + 8, N, K, offa, offb);
              Test(M, N + 8, K, offa, offb);
              Test(M + 12, N + 12, K, offa, offb);
              Test(M + 13, N, K, offa, offb);
              Test(M, N + 15, K, offa, offb);
              Test(M + 15, N + 15, K, offa, offb);
            }
          }
          printf("a %zd/%zd b %zd/%zd M %zd\n", a, _countof(zero_points), b, _countof(zero_points), M);
        }
      }
    }

    for (size_t M = 1; M < 160; M++) {
      for (size_t N = 1; N < 160; N++) {
        for (size_t K = 1; K < 160; K++) {
          Test(M, N, K, 18, 24);
        }
      }
      printf("M %zd\n", M);
    }

    for (size_t M = 160; M < 320; M += 24) {
      for (size_t N = 112; N < 320; N += 24) {
        for (size_t K = 1; K < 16; K++) {
          Test(M, N, K, 1, 3);
        }
        for (size_t K = 16; K < 160; K += 32) {
          Test(M, N, K, 5, 7);
        }
      }
      printf("M %zd\n", M);
    }
  }
};

template <typename xint8_t, bool Packed, bool Threaded>
class MlasQgemmU8X8Test<xint8_t, float, Packed, Threaded> : public MlasQgemmU8X8U8X8TestBase<Packed, Threaded> {
 public:
  void Test(size_t M, size_t N, size_t K, uint8_t offa, uint8_t offb) {
    const uint8_t* A = BufferA.GetBuffer(K * M);
    const uint8_t* B = BufferB.GetBuffer(N * K);
    float* C = BufferC.GetBuffer(N * M);
    float* CReference = BufferCReference.GetBuffer(N * M);
    const float* Bias = BufferBias.GetBuffer(N);

    const float AScale = 0.5f;
    float* AFloat = BufferAFloat.GetBuffer(K * M);
    DequantizeLinear(A, AFloat, K * M, AScale, offa);

    const float BScale = 0.25f;
    float* BFloat = BufferBFloat.GetBuffer(N * K);
    DequantizeLinear((xint8_t*)B, BFloat, N * K, BScale, xint8_t(offb));

    const float CScale = AScale * BScale;

    Test(M, N, K, A, AFloat, K, offa, B, BFloat, N, offb, C, CReference, N, CScale, nullptr);
    Test(M, N, K, A, AFloat, K, offa, B, BFloat, N, offb, C, CReference, N, CScale, Bias);
  }

  void Test(size_t M,
            size_t N,
            size_t K,
            const uint8_t* A,
            const float* AFloat,
            size_t lda,
            uint8_t offa,
            const uint8_t* B,
            const float* BFloat,
            size_t ldb,
            uint8_t offb,
            float* C,
            float* CReference,
            size_t ldc,
            float CScale,
            const float* Bias) {
    MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, AFloat, lda, BFloat, ldb, 0.0f,
             CReference, ldc, MlasQgemmU8X8U8X8TestBase<Packed, Threaded>::threadpool_);

    if (Bias != nullptr) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
          CReference[m * ldc + n] += Bias[n];
        }
      }
    }

    this->TestGemm(M, N, K, A, lda, offa, B, ldb, offb, BIsSigned, C, ldc, CScale, Bias);

    for (size_t m = 0, f = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++, f++) {
        // Sensitive to comparing positive/negative zero.
        ASSERT_EQ(C[f], CReference[f]) << "@[" << m << "x" << n << "], "
                                       << "M=" << M << ", N=" << N << ", K=" << K
                                       << ", offa=" << int(offa)
                                       << ", offb=" << int(offb);
      }
    }
  }

 private:
  template <typename qint8_t>
  void DequantizeLinear(const qint8_t* Input,
                        float* Output,
                        size_t N,
                        float scale,
                        qint8_t offset) {
    for (size_t n = 0; n < N; n++) {
      Output[n] = float((int32_t(Input[n]) - offset)) * scale;
    }
  }

  MatrixGuardBuffer<uint8_t> BufferA;
  MatrixGuardBuffer<uint8_t> BufferB;
  MatrixGuardBuffer<float> BufferAFloat;
  MatrixGuardBuffer<float> BufferBFloat;
  MatrixGuardBuffer<float> BufferC;
  MatrixGuardBuffer<float> BufferCReference;
  MatrixGuardBuffer<float> BufferBias;
  const bool BIsSigned = std::is_signed<xint8_t>::value;

 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("QGemmU8") +
                                    (std::is_signed<xint8_t>::value ? "S8" : "U8") +
                                    (Packed ? "_Fp32_Packed" : "_Fp32_NoPack") +
                                    (Threaded ? "_Threaded" : "_SingleThread");
    return suite_name.c_str();
  }
};
