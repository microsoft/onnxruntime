/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the MIT License.

Module Name:

    test_sbgemm.h

Abstract:

    Tests for MLAS bf16 precision GEMM.

--*/

#if defined(__aarch64__) && defined(__linux__)

#pragma once

#include "test_util.h"

template <typename T>
void SmallFloatFill(T* start, size_t size) {
  constexpr float MinimumFillValue = -11.0f;
  auto FillAddress = start;
  size_t offset = size % 23;

  for (size_t i = 0; i < size; i++) {
    offset = (offset + 21) % 23;
    *FillAddress++ = T((MinimumFillValue + offset) / 16.0f);
  }
}

float cosine_similarity(const float* A, const float* B, size_t Vector_Length) {
  float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
  for (size_t i = 0u; i < Vector_Length; ++i) {
    dot += A[i] * B[i];
    denom_a += A[i] * A[i];
    denom_b += B[i] * B[i];
  }
  return dot / (sqrt(denom_a) * sqrt(denom_b));
}

/**
 * @brief Test class for bf16 precision GEMM
 * @tparam AType  Data type of A matrix, need to be float
 * @tparam BType  Data type of b matrix, can be either float or prepacked bf16
 */
template <typename AType, typename BType, bool Packed, bool Threaded>
class MlasSBGemmTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<uint8_t> BufferBPacked;
  MatrixGuardBuffer<AType> BufferA;
  MatrixGuardBuffer<BType> BufferB;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<float> BufferC;
  MatrixGuardBuffer<float> BufferCReference;
  MatrixGuardBuffer<float> BufferFloatC;
  MLAS_THREADPOOL* threadpool_;

  void* PackB(size_t N, size_t K, const BType* B, size_t ldb) {
    size_t PackedBSize = MlasSBGemmPackBSize(N, K);
    if (PackedBSize == 0) {
      return nullptr;
    }
    void* PackedB = BufferBPacked.GetBuffer(PackedBSize);
    if (std::is_same<BType, float>::value) {
      MlasSBGemmConvertPackB(N, K, (const float*)B, ldb, PackedB);
    } else {
    }
    return PackedB;
  }

  void CallSBGemm(size_t M,
                  size_t N,
                  size_t K,
                  size_t BatchSize,
                  const float* A,
                  size_t lda,
                  const BType* B,
                  size_t ldb,
                  const float* Bias,
                  float* C,
                  size_t ldc) {
    std::vector<MLAS_SBGEMM_DATA_PARAMS> GemmParameters(BatchSize);

    for (size_t i = 0; i < GemmParameters.size(); i++) {
      auto& params = GemmParameters[i];
      params.A = A + (M * lda * i);
      params.lda = lda;
      if (nullptr != Bias) {
        params.Bias = reinterpret_cast<const float*>(Bias + N * i);
      } else {
        params.Bias = nullptr;
      }
      params.C = reinterpret_cast<float*>(C + (M * ldc * i));
      params.ldc = ldc;
      params.AIsfp32 = true;
      params.BIsfp32 = true;
      params.BIsPacked = false;

      if (Packed) {
        ASSERT_EQ(BatchSize, size_t(1)) << "Packing B not supported in batching yet!";
        params.B = PackB(N, K, B, ldb);
        params.BIsPacked = true;
        params.ldb = 0;
        params.BIsfp32 = false;
      } else {
        params.B = B + (K * N * i);
        params.ldb = ldb;
      }
    }

    MlasSBGemmBatch(M, N, K, BatchSize, GemmParameters.data(), threadpool_);
  }

  void ReferenceSgemm(size_t M,
                      size_t N,
                      size_t K,
                      size_t BatchSize,
                      const AType* A,
                      const BType* B,
                      const float* Bias,
                      float* C) {
    constexpr size_t KStride = 256;

    for (size_t batch = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
          const AType* a = A + M * K * batch + m * K;
          const BType* b = B + K * N * batch + n;
          float* c = C + (M * N * batch) + (m * N) + n;

          for (size_t k = 0; k < K; k += KStride) {
            float sum = 0.0f;
            if (k == 0 && Bias != nullptr) {
              sum = float(Bias[n]);
            }
            for (size_t kk = 0; kk < std::min(KStride, K - k); kk++) {
              float down(float(*b) * float(*a) + sum);
              sum = float(down);
              b += N;
              a += 1;
            }
            if (k == 0) {
              *c = sum;
            } else {
              float d(sum + *c);
              *c = float(d);
            }
          }
        }
      }
      if (Bias) {
        Bias += N;
      }
    }
  }

 public:
  MlasSBGemmTest() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

  void Test(size_t M, size_t N, size_t K, size_t BatchSize, bool withBias) {
    AType* A = BufferA.GetFilledBuffer(K * M * BatchSize + 16, SmallFloatFill<AType>);
    AType Atail[16];
    std::memcpy(Atail, A + K * M * BatchSize, 16 * sizeof(AType));

    BType* B = BufferB.GetFilledBuffer(N * K * BatchSize + 16, SmallFloatFill<BType>);
    BType Btail[16];
    std::memcpy(Btail, B + N * K * BatchSize, 16 * sizeof(BType));

    float BiasTail[16];
    const float* Bias = nullptr;
    if (withBias) {
      Bias = BufferBias.GetFilledBuffer(N * BatchSize + 16, SmallFloatFill<float>);
      std::memcpy(BiasTail, Bias + N * BatchSize, 16 * sizeof(float));
    }

    float* C = BufferC.GetFilledBuffer(N * M * BatchSize, SmallFloatFill<float>);
    float* CReference = BufferCReference.GetFilledBuffer(
        N * M * BatchSize,
        [](float* start, size_t size) {
          std::fill_n(start, size, -1.0f);
        });
    this->CallSBGemm(M, N, K, BatchSize, A, K, B, N, Bias, C, N);
    ReferenceSgemm(M, N, K, BatchSize, A, B, Bias, CReference);
    const float cosine_similarity_threshold = 0.98;

    for (size_t batch = 0, f = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++, f++) {
          if (!(CloseEnough(float(C[f]), CReference[f]))) {
            float cos_sim = cosine_similarity(C, CReference, (BatchSize * M * N));
            if (abs(cos_sim) < cosine_similarity_threshold) {
              ASSERT_TRUE(false) << "cosine similarity check failed" << cos_sim;
            } else {
              break;
            }
          }
        }
      }
    }

    ASSERT_EQ(std::memcmp(Atail, A + K * M * BatchSize, 16 * sizeof(AType)), 0) << "Matrix A buffer overwritten!";
    ASSERT_EQ(std::memcmp(Btail, B + N * K * BatchSize, 16 * sizeof(BType)), 0) << "Matrix B buffer overwritten!";
    if (withBias) {
      ASSERT_EQ(std::memcmp(BiasTail, Bias + N * BatchSize, 16 * sizeof(float)), 0) << "Bias buffer overwritten!";
    }
  }

 private:
 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("SBGemmFP") +
                                    (std::is_same<AType, float>::value ? "32" : "16") +
                                    (std::is_same<BType, float>::value ? "32" : "16") +
                                    (Packed ? "_Packed" : "_NoPack") +
                                    (Threaded ? "_Threaded" : "_SingleThread");
    return suite_name.c_str();
  }

  void ExecuteLong(void) override {
    for (size_t M = 16; M < 160; M += 32) {
      for (size_t N = 16; N < 160; N += 32) {
        static const size_t ks[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 20, 32, 48, 64, 118, 119, 120, 121, 122, 160, 240, 320};
        for (size_t k = 0; k < _countof(ks); k++) {
          size_t K = ks[k];

          Test(M, N, K, 1, false);
          Test(M, N, K, 1, true);
          Test(M + 1, N, K, 1, false);
          Test(M, N + 1, K, 1, true);
          Test(M + 1, N + 1, K, 1, false);
          Test(M + 3, N + 2, K, 1, true);
          Test(M + 4, N, K, 1, false);
          Test(M, N + 4, K, 1, true);
          Test(M + 4, N + 4, K, 1, false);
          Test(M + 3, N + 7, K, 1, true);
          Test(M + 8, N, K, 1, false);
          Test(M, N + 8, K, 1, true);
          Test(M + 12, N + 12, K, 1, false);
          Test(M + 13, N, K, 1, true);
          Test(M, N + 15, K, 1, false);
          Test(M + 15, N + 15, K, 1, false);
          if (!Packed) {
            Test(M, N, K, 7, false);
            Test(M + 3, N, K, 8, true);
            Test(M, N + 1, K, 9, false);
            Test(M + 12, N, K, 10, true);
            Test(M, N + 15, K, 11, false);
            Test(M + 15, N + 15, K, 12, true);
          }
        }
      }
      printf("M %zd\n", M);
    }

    for (size_t M = 1; M < 160; M++) {
      for (size_t N = 1; N < 160; N++) {
        for (size_t K = 1; K < 160; K++) {
          Test(M, N, K, 1, true);
        }
      }
      printf("M %zd\n", M);
    }

    for (size_t M = 160; M < 320; M += 24) {
      for (size_t N = 112; N < 320; N += 24) {
        for (size_t K = 1; K < 16; K++) {
          Test(M, N, K, 1, true);
        }
        for (size_t K = 16; K < 160; K += 32) {
          Test(M, N, K, 1, false);
        }
      }
      printf("M %zd\n", M);
    }
  }
};

#endif  // defined(__aarch64__) && defined(__linux__)
