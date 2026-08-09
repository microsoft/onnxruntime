// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_util.h"

template <bool Threaded>
class MlasSymmQgemmTestBase : public MlasTestBase {
 protected:
  MLAS_THREADPOOL* threadpool_;

  MlasSymmQgemmTestBase() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

  void TestGemm(size_t M,
                size_t N,
                size_t K,
                size_t BatchSize,
                const uint8_t* A,
                size_t lda,
                int32_t offa,
                bool AIsSigned,
                const int8_t* B,
                size_t ldb,
                int32_t* C,
                size_t ldc) {
    MLAS_GEMM_QUANT_SHAPE_PARAMS GemmShape;
    GemmShape.M = M;
    GemmShape.N = N;
    GemmShape.K = K;
    GemmShape.AIsSigned = AIsSigned;
    GemmShape.BIsSigned = true;

    size_t PackedBSize = MlasSymmQgemmPackBSize(N, K, AIsSigned);
    int8_t* PackedB = (int8_t*)BufferBPacked.GetBuffer(PackedBSize * BatchSize);

    std::vector<MLAS_SYMM_QGEMM_DATA_PARAMS> GemmParameters(BatchSize);

    for (size_t i = 0; i < GemmParameters.size(); i++) {
      auto& params = GemmParameters[i];
      params.A = A + (M * K * i);
      params.lda = lda;
      params.C = C + (M * N * i);
      params.ldc = ldc;

      MlasSymmQgemmPackB(N, K, B + (K * N * i), ldb, AIsSigned, offa, PackedB + PackedBSize * i);
      params.B = PackedB + PackedBSize * i;
    }

    MlasSymmQgemmBatch(GemmShape, GemmParameters.data(), BatchSize, threadpool_);
  }

 private:
  MatrixGuardBuffer<uint8_t> BufferBPacked;
};

template <typename AType, typename OutputType, bool Threaded>
class MlasSymmQgemmTest;

template <typename AType, bool Threaded>
class MlasSymmQgemmTest<AType, int32_t, Threaded> : public MlasSymmQgemmTestBase<Threaded> {
 public:
  void Test(size_t M, size_t N, size_t K, size_t BatchSize, int32_t offa) {
    // Symmetric kernel will have limited buffer overrun when reading the input buffer
    constexpr size_t OVERRUN = 15;
    const uint8_t* A = BufferA.GetBuffer(K * M * BatchSize + OVERRUN);
    const int8_t* B = BufferB.GetBuffer(N * K * BatchSize);
    int32_t* C = BufferC.GetBuffer(N * M * BatchSize);
    int32_t* CReference = BufferCReference.GetBuffer(N * M * BatchSize);

    Test(M, N, K, BatchSize, A, K, offa, B, N, C, CReference, N);
  }

  void Test(size_t M,
            size_t N,
            size_t K,
            size_t BatchSize,
            const uint8_t* A,
            size_t lda,
            int32_t offa,
            const int8_t* B,
            size_t ldb,
            int32_t* C,
            int32_t* CReference,
            size_t ldc) {
    std::fill_n(C, M * N * BatchSize, -1);
    std::fill_n(CReference, M * N * BatchSize, -1);

    this->TestGemm(M, N, K, BatchSize, A, lda, offa, std::is_signed<AType>::value, B, ldb, C, ldc);
    ReferenceQgemm(M, N, K, BatchSize, (const AType*)A, lda, (AType)offa, B, ldb, (const int8_t)0, CReference, ldc);

    for (size_t batch = 0, f = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++, f++) {
          ASSERT_EQ(C[f], CReference[f]) << "@[" << batch << "x" << m << "x" << n << "], "
                                         << "Batch=" << BatchSize << "M=" << M << ", N=" << N << ", K=" << K
                                         << ", offa=" << offa << ", offb=--";
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
                      const int8_t* B,
                      size_t ldb,
                      int8_t offb,
                      int32_t* C,
                      size_t ldc) {
    for (size_t batch = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
          const AType* a = A + (M * K * batch) + (m * lda);
          const int8_t* b = B + (K * N * batch) + n;
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

  MatrixGuardBuffer<uint8_t> BufferA;
  MatrixGuardBuffer<int8_t> BufferB;
  MatrixGuardBuffer<int32_t> BufferC;
  MatrixGuardBuffer<int32_t> BufferCReference;

 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("SymmQgemm") +
                                    (std::is_signed<AType>::value ? "S8" : "U8") +
                                    "_Int32" +
                                    (Threaded ? "_Threaded" : "_SingleThread");
    return suite_name.c_str();
  }

  void ExecuteLong(void) override {
    static const int32_t zero_points[] = {-18, 124};

    for (size_t a = 0; a < _countof(zero_points); a++) {
      int32_t offa = zero_points[a];

      for (size_t M = 16; M < 160; M += 32) {
        for (size_t N = 16; N < 160; N += 32) {
          static const size_t ks[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 20, 32, 48, 64, 118, 119, 120, 121, 122, 160, 240, 320};
          for (size_t k = 0; k < _countof(ks); k++) {
            size_t K = ks[k];

            Test(M, N, K, 1, offa);
            Test(M + 1, N, K, 1, offa);
            Test(M, N + 1, K, 1, offa);
            Test(M + 1, N + 1, K, 1, offa);
            Test(M + 3, N + 2, K, 1, offa);
            Test(M + 4, N, K, 1, offa);
            Test(M, N + 4, K, 1, offa);
            Test(M + 4, N + 4, K, 1, offa);
            Test(M + 3, N + 7, K, 1, offa);
            Test(M + 8, N, K, 1, offa);
            Test(M, N + 8, K, 1, offa);
            Test(M + 12, N + 12, K, 1, offa);
            Test(M + 13, N, K, 1, offa);
            Test(M, N + 15, K, 1, offa);
            Test(M + 15, N + 15, K, 1, offa);
            Test(M, N, K, 7 + a, offa);
            Test(M + 3, N, K, 7 + a, offa);
            Test(M, N + 1, K, 7 + a, offa);
            Test(M + 12, N, K, 7 + a, offa);
            Test(M, N + 15, K, 7 + a, offa);
            Test(M + 15, N + 15, K, 7 + a, offa);
          }
        }
        printf("a %zd/%zd b %zd M %zd\n", a, _countof(zero_points), _countof(zero_points), M);
      }
    }

    for (size_t M = 1; M < 160; M++) {
      for (size_t N = 1; N < 160; N++) {
        for (size_t K = 1; K < 160; K++) {
          Test(M, N, K, 1, 18);
        }
      }
      printf("M %zd\n", M);
    }

    for (size_t M = 160; M < 320; M += 24) {
      for (size_t N = 112; N < 320; N += 24) {
        for (size_t K = 1; K < 16; K++) {
          Test(M, N, K, 1, 1);
        }
        for (size_t K = 16; K < 160; K += 32) {
          Test(M, N, K, 1, -5);
        }
      }
      printf("M %zd\n", M);
    }
  }
};
