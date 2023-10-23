// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_util.h"

template <typename T>
const char* GetGemmTestSuitePrefix();

template <>
const char* GetGemmTestSuitePrefix<float>() {
  return "SGemm";
}

template <>
const char* GetGemmTestSuitePrefix<double>() {
  return "DGemm";
}

template <typename T, bool Packed>
class FgemmPackedContext;

template <>
class FgemmPackedContext<float, false> {
 public:
  void
  TestGemm(
      CBLAS_TRANSPOSE TransA,
      CBLAS_TRANSPOSE TransB,
      size_t M,
      size_t N,
      size_t K,
      size_t BatchSize,
      const float alpha,
      const float* A,
      size_t lda,
      const float* B,
      size_t ldb,
      const float beta,
      float* C,
      size_t ldc,
      MLAS_THREADPOOL* threadpool) {
    std::vector<MLAS_SGEMM_DATA_PARAMS> data(BatchSize);
    for (size_t i = 0; i < BatchSize; i++) {
      data[i].A = A + M * K * i;
      data[i].lda = lda;
#if 0 // comment for prototype
      data[i].B = B + K * N * i;
#endif
      data[i].ldb = ldb;
      data[i].C = C + M * N * i;
      data[i].ldc = ldc;
      data[i].alpha = alpha;
      data[i].beta = beta;
    }
    MlasGemmBatch(TransA, TransB, M, N, K, data.data(), BatchSize, threadpool);
  }
};

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_POWER)
template <>
class FgemmPackedContext<double, false> {
 public:
  void TestGemm(
      CBLAS_TRANSPOSE TransA,
      CBLAS_TRANSPOSE TransB,
      size_t M,
      size_t N,
      size_t K,
      size_t BatchSize,
      double alpha,
      const double* A,
      size_t lda,
      const double* B,
      size_t ldb,
      double beta,
      double* C,
      size_t ldc,
      MLAS_THREADPOOL* threadpool) {
    std::vector<MLAS_DGEMM_DATA_PARAMS> data(BatchSize);
    for (size_t i = 0; i < BatchSize; i++) {
      data[i].A = A + M * K * i;
      data[i].lda = lda;
      data[i].B = B + K * N * i;
      data[i].ldb = ldb;
      data[i].C = C + M * N * i;
      data[i].ldc = ldc;
      data[i].alpha = alpha;
      data[i].beta = beta;
    }
    MlasGemmBatch(TransA, TransB, M, N, K, data.data(), BatchSize, threadpool);
  }
};
#endif

template <>
class FgemmPackedContext<float, true> {
 public:
  void
  TestGemm(
      CBLAS_TRANSPOSE TransA,
      CBLAS_TRANSPOSE TransB,
      size_t M,
      size_t N,
      size_t K,
      size_t BatchSize,
      const float alpha,
      const float* A,
      size_t lda,
      const float* B,
      size_t ldb,
      const float beta,
      float* C,
      size_t ldc,
      MLAS_THREADPOOL* threadpool) {
    size_t PackedBSize = MlasGemmPackBSize(N, K);
    void* PackedB = BufferBPacked.GetBuffer(PackedBSize * BatchSize, true);
    std::vector<MLAS_SGEMM_DATA_PARAMS> data(BatchSize);
    for (size_t i = 0; i < BatchSize; i++) {
      MlasGemmPackB(TransB, N, K, B + K * N * i, ldb, (uint8_t*)PackedB + PackedBSize * i);
      data[i].BIsPacked = true;
      data[i].A = A + M * K * i;
      data[i].lda = lda;
#if 0 // comment for prototype
      data[i].B = (float*)((uint8_t*)PackedB + PackedBSize * i);
#endif
      data[i].ldb = ldb;
      data[i].C = C + M * N * i;
      data[i].ldc = ldc;
      data[i].alpha = alpha;
      data[i].beta = beta;
    }
    MlasGemmBatch(TransA, TransB, M, N, K, data.data(), BatchSize, threadpool);
  }

 private:
  MatrixGuardBuffer<uint8_t> BufferBPacked;
};

template <typename T, bool Packed, bool Threaded>
class MlasFgemmTest : public MlasTestBase {
 private:
  MLAS_THREADPOOL* threadpool_;

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name = std::string(GetGemmTestSuitePrefix<T>()) +
                                          (Packed ? "_Packed" : "_NoPack") +
                                          (Threaded ? "_Threaded" : "_SingleThread");

    return suite_name.c_str();
  }

  MlasFgemmTest() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

  void Test(size_t M, size_t N, size_t K, size_t BatchSize, T alpha, T beta) {
    Test(false, false, M, N, K, BatchSize, alpha, beta);
    Test(false, true, M, N, K, BatchSize, alpha, beta);
    Test(true, false, M, N, K, BatchSize, alpha, beta);
    Test(true, true, M, N, K, BatchSize, alpha, beta);
  }

  void Test(bool trans_a, bool trans_b, size_t M, size_t N, size_t K, size_t BatchSize, T alpha, T beta) {
    //
    // Skip the test if the B buffer cannot be packed.
    //
    if constexpr (Packed) {
      if (N == 0 || K == 0)
        return;
    }

    const T* A = BufferA.GetBuffer(K * M * BatchSize);
    const T* B = BufferB.GetBuffer(N * K * BatchSize);
    T* C = BufferC.GetBuffer(N * M * BatchSize);
    T* CReference = BufferCReference.GetBuffer(N * M * BatchSize);

    Test(trans_a ? CblasTrans : CblasNoTrans,
         trans_b ? CblasTrans : CblasNoTrans,
         M, N, K, BatchSize, alpha, A, trans_a ? M : K, B, trans_b ? K : N,
         beta, C, CReference, N);
  }

  void Test(CBLAS_TRANSPOSE TransA,
            CBLAS_TRANSPOSE TransB,
            size_t M,
            size_t N,
            size_t K,
            size_t BatchSize,
            T alpha,
            const T* A,
            size_t lda,
            const T* B,
            size_t ldb,
            T beta,
            T* C,
            T* CReference,
            size_t ldc) {
    std::fill_n(C, M * N * BatchSize, -0.5f);
    std::fill_n(CReference, M * N * BatchSize, -0.5f);

    PackedContext.TestGemm(TransA, TransB, M, N, K, BatchSize, alpha, A, lda, B, ldb, beta, C, ldc, threadpool_);
    ReferenceGemm(TransA, TransB, M, N, K, BatchSize, alpha, A, lda, B, ldb, beta, CReference, ldc);

    for (size_t batch = 0, f = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++, f++) {
          // Sensitive to comparing positive/negative zero.
          ASSERT_EQ(C[f], CReference[f])
              << " Diff @[" << batch << ", " << m << ", " << n << "] f=" << f << ", "
              << (Packed ? "Packed" : "NoPack") << "."
              << (Threaded ? "SingleThread" : "Threaded") << "/"
              << (TransA == CblasTrans ? "TransA" : "A") << "/"
              << (TransB == CblasTrans ? "TransB" : "B") << "/"
              << "M" << M << "xN" << N << "xK" << K << "/"
              << "Alpha" << alpha << "/"
              << "Beta" << beta;
        }
      }
    }
  }

  void ReferenceGemm(CBLAS_TRANSPOSE TransA,
                     CBLAS_TRANSPOSE TransB,
                     size_t M,
                     size_t N,
                     size_t K,
                     size_t BatchSize,
                     T alpha,
                     const T* A,
                     size_t lda,
                     const T* B,
                     size_t ldb,
                     T beta,
                     T* C,
                     size_t ldc) {
    for (size_t batch = 0; batch < BatchSize; batch++) {
      if (TransA == CblasNoTrans) {
        if (TransB == CblasNoTrans) {
          for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
              const T* a = A + (m * lda);
              const T* b = B + n;
              T* c = C + (m * ldc) + n;
              T sum = 0.0f;

              for (size_t k = 0; k < K; k++) {
                sum += (*b * *a);
                b += ldb;
                a += 1;
              }

              *c = (*c * beta) + (sum * alpha);
            }
          }

        } else {
          for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
              const T* a = A + (m * lda);
              const T* b = B + (n * ldb);
              T* c = C + (m * ldc) + n;
              T sum = 0.0f;

              for (size_t k = 0; k < K; k++) {
                sum += (*b * *a);
                b += 1;
                a += 1;
              }

              *c = (*c * beta) + (sum * alpha);
            }
          }
        }

      } else {
        if (TransB == CblasNoTrans) {
          for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
              const T* a = A + m;
              const T* b = B + n;
              T* c = C + (m * ldc) + n;
              T sum = 0.0f;

              for (size_t k = 0; k < K; k++) {
                sum += (*b * *a);
                b += ldb;
                a += lda;
              }

              *c = (*c * beta) + (sum * alpha);
            }
          }

        } else {
          for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
              const T* a = A + m;
              const T* b = B + (n * ldb);
              T* c = C + (m * ldc) + n;
              T sum = 0.0f;

              for (size_t k = 0; k < K; k++) {
                sum += (*b * *a);
                b += 1;
                a += lda;
              }

              *c = (*c * beta) + (sum * alpha);
            }
          }
        }
      }
      A += M * K;
      B += K * N;
      C += M * N;
    }
  }

  void ExecuteLong() override {
    static const T multipliers[] = {0.0f, -0.0f, 0.25f, -0.5f, 1.0f, -1.0f};

    for (size_t N = 1; N < 128; N++) {
      for (size_t K = 1; K < 128; K++) {
        for (size_t a = 0; a < _countof(multipliers); a++) {
          for (size_t b = 0; b < _countof(multipliers); b++) {
            Test(1, N, K, 1, multipliers[a], multipliers[b]);
            Test(N, 1, K, 1, multipliers[a], multipliers[b]);
            if (!Packed) {
              Test(1, N, K, 3, multipliers[a], multipliers[b]);
            }
          }
        }
      }
    }

    for (size_t a = 0; a < _countof(multipliers); a++) {
      T alpha = multipliers[a];

      for (size_t b = 0; b < _countof(multipliers); b++) {
        T beta = multipliers[b];

        for (size_t M = 16; M < 160; M += 32) {
          for (size_t N = 16; N < 160; N += 32) {
            static const size_t ks[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 32, 48, 64, 118, 119, 120, 121, 122, 160, 240, 320};
            for (size_t k = 0; k < _countof(ks); k++) {
              size_t K = ks[k];

              Test(M, N, K, 1, alpha, beta);
              Test(M + 1, N, K, 1, alpha, beta);
              Test(M, N + 1, K, 1, alpha, beta);
              Test(M + 1, N + 1, K, 1, alpha, beta);
              Test(M + 3, N + 2, K, 1, alpha, beta);
              Test(M + 4, N, K, 1, alpha, beta);
              Test(M, N + 4, K, 1, alpha, beta);
              Test(M + 4, N + 4, K, 1, alpha, beta);
              Test(M + 3, N + 7, K, 1, alpha, beta);
              Test(M + 8, N, K, 1, alpha, beta);
              Test(M, N + 8, K, 1, alpha, beta);
              Test(M + 12, N + 12, K, 1, alpha, beta);
              Test(M + 13, N, K, 1, alpha, beta);
              Test(M, N + 15, K, 1, alpha, beta);
              Test(M + 15, N + 15, K, 1, alpha, beta);
              if (!Packed) {
                Test(M + 3, N + 1, K, 7, multipliers[a], multipliers[b]);
                Test(M + 13, N + 2, K, 9, multipliers[a], multipliers[b]);
              }
            }
          }
          printf("a %zd/%zd b %zd/%zd M %zd\n", a, _countof(multipliers), b, _countof(multipliers), M);
        }
      }
    }

    for (size_t M = 0; M < 160; M++) {
      for (size_t N = 0; N < 160; N++) {
        for (size_t K = 0; K < 160; K++) {
          Test(M, N, K, 1, 1.0f, 0.0f);
        }
      }
      printf("M %zd\n", M);
    }

    for (size_t M = 160; M < 320; M += 24) {
      for (size_t N = 112; N < 320; N += 24) {
        for (size_t K = 0; K < 16; K++) {
          Test(M, N, K, 1, 1.0f, 0.0f);
        }
        for (size_t K = 16; K < 160; K += 32) {
          Test(M, N, K, 1, 1.0f, 0.0f);
        }
      }
      printf("M %zd\n", M);
    }
  }

  MatrixGuardBuffer<T> BufferA;
  MatrixGuardBuffer<T> BufferB;
  MatrixGuardBuffer<T> BufferC;
  MatrixGuardBuffer<T> BufferCReference;
  FgemmPackedContext<T, Packed> PackedContext;
};
