// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_util.h"

template <bool Packed, bool Threaded>
class MlasQgemmTestBase : public MlasTestBase {
 private:
  void* PackB(size_t N, size_t K, const uint8_t* B, size_t ldb, bool AIsSigned, bool BIsSigned) {
    size_t PackedBSize = MlasGemmPackBSize(N, K, AIsSigned, BIsSigned);
    void* PackedB = BufferBPacked.GetBuffer(PackedBSize);
    MlasGemmPackB(N, K, B, ldb, AIsSigned, BIsSigned, PackedB);
    return PackedB;
  }

 protected:
  MLAS_THREADPOOL* threadpool_;

  MlasQgemmTestBase() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

  void TestGemm(size_t M,
                size_t N,
                size_t K,
                size_t BatchSize,
                const uint8_t* A,
                size_t lda,
                uint8_t offa,
                bool AIsSigned,
                const uint8_t* B,
                size_t ldb,
                uint8_t offb,
                bool BIsSigned,
                int32_t* C,
                size_t ldc) {
    MLAS_GEMM_QUANT_SHAPE_PARAMS GemmShape;
    GemmShape.M = M;
    GemmShape.N = N;
    GemmShape.K = K;
    GemmShape.AIsSigned = AIsSigned;
    GemmShape.BIsSigned = BIsSigned;

    std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> GemmParameters(BatchSize);

    for (size_t i = 0; i < GemmParameters.size(); i++) {
      auto& params = GemmParameters[i];
      params.A = A + (M * K * i);
      params.lda = lda;
      params.ZeroPointA = offa;
      params.ZeroPointB = &offb;
      params.C = C + (M * N * i);
      params.ldc = ldc;

      if (Packed) {
        ASSERT_EQ(BatchSize, size_t(1)) << "Packing B not supported in batching yet!";
        params.B = PackB(N, K, B, ldb, AIsSigned, BIsSigned);
        params.BIsPacked = true;
      } else {
        params.B = B + (K * N * i);
        params.ldb = ldb;
      }
    }

    MlasGemmBatch(GemmShape, GemmParameters.data(), BatchSize, threadpool_);
  }

  void TestGemm(size_t M,
                size_t N,
                size_t K,
                size_t BatchSize,
                const uint8_t* A,
                size_t lda,
                uint8_t offa,
                bool AIsSigned,
                const uint8_t* B,
                size_t ldb,
                const uint8_t* offb,
                bool BIsSigned,
                int32_t* C,
                size_t ldc) {
    MLAS_GEMM_QUANT_SHAPE_PARAMS GemmShape;
    GemmShape.M = M;
    GemmShape.N = N;
    GemmShape.K = K;
    GemmShape.AIsSigned = AIsSigned;
    GemmShape.BIsSigned = BIsSigned;

    std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> GemmParameters(BatchSize);

    for (size_t i = 0; i < GemmParameters.size(); i++) {
      auto& params = GemmParameters[i];
      params.A = A + M * K * i;
      params.lda = lda;
      params.ZeroPointA = offa;
      params.ZeroPointB = offb;
      params.PerColumnZeroPoints = true;
      params.C = C + M * N * i;
      params.ldc = ldc;

      if (Packed) {
        ASSERT_EQ(BatchSize, size_t(1)) << "Packing B not supported in batching yet!";
        params.B = PackB(N, K, B, ldb, AIsSigned, BIsSigned);
        params.BIsPacked = true;
      } else {
        params.B = B + K * N * i;
        params.ldb = ldb;
      }
    }

    MlasGemmBatch(GemmShape, GemmParameters.data(), BatchSize, threadpool_);
  }

  void TestGemm(size_t M,
                size_t N,
                size_t K,
                size_t BatchSize,
                const uint8_t* A,
                size_t lda,
                uint8_t offa,
                bool AIsSigned,
                const uint8_t* B,
                size_t ldb,
                uint8_t offb,
                bool BIsSigned,
                float* C,
                size_t ldc,
                float CScale,
                const float* Bias) {
    MLAS_GEMM_QUANT_SHAPE_PARAMS GemmShape;
    GemmShape.M = M;
    GemmShape.N = N;
    GemmShape.K = K;
    GemmShape.AIsSigned = AIsSigned;
    GemmShape.BIsSigned = BIsSigned;

    std::vector<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR> ScaleBiasProcessors;
    ScaleBiasProcessors.reserve(BatchSize);

    std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> GemmParameters(BatchSize);

    for (size_t i = 0; i < BatchSize; i++) {
      auto& params = GemmParameters[i];
      params.A = A + M * K * i;
      params.lda = lda;
      params.ZeroPointA = offa;
      params.ZeroPointB = &offb;
      params.C = reinterpret_cast<int32_t*>(C + M * N * i);
      params.ldc = ldc;

      if (Packed) {
        ASSERT_EQ(BatchSize, size_t(1)) << "Packing B not supported in batching yet!";
        params.B = PackB(N, K, B, ldb, AIsSigned, BIsSigned);
        params.BIsPacked = true;
      } else {
        params.B = B + K * N * i;
        params.ldb = ldb;
      }
      ScaleBiasProcessors.emplace_back(C + M * N * i, ldc, &CScale, Bias);
      params.OutputProcessor = &(ScaleBiasProcessors[i]);
    }

    MlasGemmBatch(GemmShape, GemmParameters.data(), BatchSize, threadpool_);
  }

 private:
  MatrixGuardBuffer<uint8_t> BufferBPacked;
};

template <typename AType, typename BType, typename OutputType, bool Packed, bool Threaded>
class MlasQgemmTest;

template <typename AType, typename BType, bool Packed, bool Threaded>
class MlasQgemmTest<AType, BType, int32_t, Packed, Threaded> : public MlasQgemmTestBase<Packed, Threaded> {
 public:
  void Test(size_t M, size_t N, size_t K, size_t BatchSize, uint8_t offa, uint8_t offb) {
    const uint8_t* A = BufferA.GetBuffer(K * M * BatchSize);
    const uint8_t* B = BufferB.GetBuffer(N * K * BatchSize);
    int32_t* C = BufferC.GetBuffer(N * M * BatchSize);
    int32_t* CReference = BufferCReference.GetBuffer(N * M * BatchSize);

    Test(M, N, K, BatchSize, A, K, offa, B, N, offb, C, CReference, N);
  }

  void Test(size_t M, size_t N, size_t K, size_t BatchSize, uint8_t offa) {
    const uint8_t* A = BufferA.GetBuffer(K * M * BatchSize);
    const uint8_t* B = BufferB.GetBuffer(N * K * BatchSize);
    const uint8_t* ZeroPointB = BufferZeroPointB.GetBuffer(N);
    int32_t* C = BufferC.GetBuffer(N * M * BatchSize);
    int32_t* CReference = BufferCReference.GetBuffer(N * M * BatchSize);

    Test(M, N, K, BatchSize, A, K, offa, B, N, ZeroPointB, C, CReference, N);
  }

  void Test(size_t M,
            size_t N,
            size_t K,
            size_t BatchSize,
            const uint8_t* A,
            size_t lda,
            uint8_t offa,
            const uint8_t* B,
            size_t ldb,
            uint8_t offb,
            int32_t* C,
            int32_t* CReference,
            size_t ldc) {
    std::fill_n(C, M * N * BatchSize, -1);
    std::fill_n(CReference, M * N * BatchSize, -1);

    this->TestGemm(M, N, K, BatchSize, A, lda, offa, AIsSigned, B, ldb, offb, BIsSigned, C, ldc);
    ReferenceQgemm(M, N, K, BatchSize, (const AType*)A, lda, (AType)offa, (const BType*)B, ldb, (BType)offb, CReference, ldc);

    for (size_t batch = 0, f = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++, f++) {
          ASSERT_EQ(C[f], CReference[f]) << "@[" << batch << "x" << m << "x" << n << "], "
                                         << "Batch=" << BatchSize << "M=" << M << ", N=" << N << ", K=" << K
                                         << ", offa=" << int(offa) << ", offb=" << int(offb);
        }
      }
    }
  }

  void Test(size_t M,
            size_t N,
            size_t K,
            size_t BatchSize,
            const uint8_t* A,
            size_t lda,
            uint8_t offa,
            const uint8_t* B,
            size_t ldb,
            const uint8_t* offb,
            int32_t* C,
            int32_t* CReference,
            size_t ldc) {
    std::fill_n(C, M * N * BatchSize, -1);
    std::fill_n(CReference, M * N * BatchSize, -1);

    this->TestGemm(M, N, K, BatchSize, A, lda, offa, AIsSigned, B, ldb, offb, BIsSigned, C, ldc);
    ReferenceQgemm(M, N, K, BatchSize, (const AType*)A, lda, (AType)offa, (const BType*)B, ldb, (const BType*)offb, CReference, ldc);

    for (size_t batch = 0, f = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++, f++) {
          ASSERT_EQ(C[f], CReference[f]) << "@[" << batch << "x" << m << "x" << n << "], "
                                         << "Batch=" << BatchSize << "M=" << M << ", N=" << N << ", K=" << K
                                         << ", offa=" << int(offa) << ", offb=--";
        }
      }
    }
  }

 private:
  void ReferenceQgemm(size_t M,
                      size_t N,
                      size_t K,
                      size_t BatchSize,
                      const AType* A,
                      size_t lda,
                      AType offa,
                      const BType* B,
                      size_t ldb,
                      BType offb,
                      int32_t* C,
                      size_t ldc) {
    for (size_t batch = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
          const AType* a = A + (M * K * batch) + (m * lda);
          const BType* b = B + (K * N * batch) + n;
          int32_t* c = C + (M * N * batch) + (m * ldc) + n;
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
  }

  void ReferenceQgemm(size_t M,
                      size_t N,
                      size_t K,
                      size_t BatchSize,
                      const AType* A,
                      size_t lda,
                      AType offa,
                      const BType* B,
                      size_t ldb,
                      const BType* offb,
                      int32_t* C,
                      size_t ldc) {
    for (size_t batch = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
          const AType* a = A + (M * K * batch) + (m * lda);
          const BType* b = B + (K * N * batch) + n;
          int32_t* c = C + (M * N * batch) + (m * ldc) + n;
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
  }

  MatrixGuardBuffer<uint8_t> BufferA;
  MatrixGuardBuffer<uint8_t> BufferB;
  MatrixGuardBuffer<uint8_t> BufferZeroPointB;
  MatrixGuardBuffer<int32_t> BufferC;
  MatrixGuardBuffer<int32_t> BufferCReference;
  const bool AIsSigned = std::is_signed<AType>::value;
  const bool BIsSigned = std::is_signed<BType>::value;

 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("QGemm") +
                                    (std::is_signed<AType>::value ? "S8" : "U8") +
                                    (std::is_signed<BType>::value ? "S8" : "U8") +
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

              Test(M, N, K, 1, offa, offb);
              Test(M + 1, N, K, 1, offa, offb);
              Test(M, N + 1, K, 1, offa, offb);
              Test(M + 1, N + 1, K, 1, offa, offb);
              Test(M + 3, N + 2, K, 1, offa, offb);
              Test(M + 4, N, K, 1, offa, offb);
              Test(M, N + 4, K, 1, offa, offb);
              Test(M + 4, N + 4, K, 1, offa, offb);
              Test(M + 3, N + 7, K, 1, offa, offb);
              Test(M + 8, N, K, 1, offa, offb);
              Test(M, N + 8, K, 1, offa, offb);
              Test(M + 12, N + 12, K, 1, offa, offb);
              Test(M + 13, N, K, 1, offa, offb);
              Test(M, N + 15, K, 1, offa, offb);
              Test(M + 15, N + 15, K, 1, offa, offb);
              if (!Packed) {
                Test(M, N, K, 7 + a, offa, offb);
                Test(M + 3, N, K, 7 + a, offa, offb);
                Test(M, N + 1, K, 7 + a, offa, offb);
                Test(M + 12, N, K, 7 + a, offa, offb);
                Test(M, N + 15, K, 7 + a, offa, offb);
                Test(M + 15, N + 15, K, 7 + a, offa, offb);
              }
            }
          }
          printf("a %zd/%zd b %zd/%zd M %zd\n", a, _countof(zero_points), b, _countof(zero_points), M);
        }
      }
    }

    for (size_t M = 1; M < 160; M++) {
      for (size_t N = 1; N < 160; N++) {
        for (size_t K = 1; K < 160; K++) {
          Test(M, N, K, 1, 18, 24);
        }
      }
      printf("M %zd\n", M);
    }

    for (size_t M = 160; M < 320; M += 24) {
      for (size_t N = 112; N < 320; N += 24) {
        for (size_t K = 1; K < 16; K++) {
          Test(M, N, K, 1, 1, 3);
        }
        for (size_t K = 16; K < 160; K += 32) {
          Test(M, N, K, 1, 5, 7);
        }
      }
      printf("M %zd\n", M);
    }
  }
};

template <typename AType, typename BType, bool Packed, bool Threaded>
class MlasQgemmTest<AType, BType, float, Packed, Threaded> : public MlasQgemmTestBase<Packed, Threaded> {
 public:
  void Test(size_t M, size_t N, size_t K, size_t BatchSize, uint8_t offa, uint8_t offb) {
    const uint8_t* A = BufferA.GetBuffer(K * M * BatchSize);
    const uint8_t* B = BufferB.GetBuffer(N * K * BatchSize);
    float* C = BufferC.GetBuffer(N * M * BatchSize);
    float* CReference = BufferCReference.GetBuffer(N * M * BatchSize);
    const float* Bias = BufferBias.GetBuffer(N);

    constexpr float AScale = 0.5f;
    float* AFloat = BufferAFloat.GetBuffer(K * M * BatchSize);
    for (size_t b = 0; b < BatchSize; b++) {
      DequantizeLinear((AType*)(A + K * M * b), AFloat + K * M * b, K * M, AScale, (AType)offa);
    }

    constexpr float BScale = 0.25f;
    float* BFloat = BufferBFloat.GetBuffer(N * K * BatchSize);
    for (size_t b = 0; b < BatchSize; b++) {
      DequantizeLinear((BType*)(B + N * K * b), BFloat + N * K * b, N * K, BScale, BType(offb));
    }

    constexpr float CScale = AScale * BScale;

    Test(M, N, K, BatchSize, A, AFloat, K, offa, B, BFloat, N, offb, C, CReference, N, CScale, nullptr);
    Test(M, N, K, BatchSize, A, AFloat, K, offa, B, BFloat, N, offb, C, CReference, N, CScale, Bias);
  }

  void Test(size_t M,
            size_t N,
            size_t K,
            size_t BatchSize,
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
    for (size_t b = 0; b < BatchSize; b++) {
#if 0 // comment for prototype
      MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
               AFloat + K * M * b, lda,
               BFloat + N * K * b, ldb, 0.0f,
               CReference + N * M * b, ldc,
               MlasQgemmTestBase<Packed, Threaded>::threadpool_);
#endif
    }

    if (Bias != nullptr) {
      for (size_t b = 0; b < BatchSize; b++) {
        for (size_t m = 0; m < M; m++) {
          for (size_t n = 0; n < N; n++) {
            CReference[N * M * b + m * ldc + n] += Bias[n];
          }
        }
      }
    }

    this->TestGemm(M, N, K, BatchSize, A, lda, offa, AIsSigned, B, ldb, offb, BIsSigned, C, ldc, CScale, Bias);

    for (size_t batch = 0, f = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++, f++) {
          // Sensitive to comparing positive/negative zero.
          ASSERT_EQ(C[f], CReference[f]) << "@[" << batch << "x" << m << "x" << n << "], "
                                         << "Batch=" << BatchSize << "M=" << M << ", N=" << N << ", K=" << K
                                         << ", offa=" << int(offa) << ", offb=" << offb;
        }
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
  const bool AIsSigned = std::is_signed<AType>::value;
  const bool BIsSigned = std::is_signed<BType>::value;

 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("QGemm") +
                                    (std::is_signed<AType>::value ? "S8" : "U8") +
                                    (std::is_signed<BType>::value ? "S8" : "U8") +
                                    (Packed ? "_Fp32_Packed" : "_Fp32_NoPack") +
                                    (Threaded ? "_Threaded" : "_SingleThread");
    return suite_name.c_str();
  }
};
