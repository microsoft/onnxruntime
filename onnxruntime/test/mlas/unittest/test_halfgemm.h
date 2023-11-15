/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_halfgemm.h

Abstract:

    Tests for MLAS half precision GEMM.

--*/

#pragma once

#include "test_fp16.h"

/**
 * @brief Test class for half precision GEMM
 * @tparam AType  Data type of A matrix, can be either float or MLFp16
 * @tparam BType  Data type of b matrix, can be either float or MLFp16
 */
template <typename AType, typename BType, bool Packed, bool Threaded>
class MlasHalfGemmTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<uint8_t> BufferBPacked;
  MatrixGuardBuffer<AType> BufferA;
  MatrixGuardBuffer<BType> BufferB;
  MatrixGuardBuffer<MLFp16> BufferBias;
  MatrixGuardBuffer<MLFp16> BufferC;
  MatrixGuardBuffer<float> BufferCReference;
  MatrixGuardBuffer<float> BufferFloatC;
  MLAS_THREADPOOL* threadpool_;

  void* PackB(size_t N, size_t K, const BType* B, size_t ldb) {
    size_t PackedBSize = MlasHalfGemmPackBSize(N, K, std::is_same<BType, float>::value);
    if (PackedBSize == 0) {
      return nullptr;
    }
    void* PackedB = BufferBPacked.GetBuffer(PackedBSize);
    if (std::is_same<BType, float>::value) {
      MlasHalfGemmConvertPackB(N, K, (const float*)B, ldb, PackedB);
    } else {
      MlasHalfGemmPackB(N, K, (const MLAS_FP16*)B, ldb, PackedB);
    }
    return PackedB;
  }

  void CallGemm(size_t M,
                size_t N,
                size_t K,
                size_t BatchSize,
                const AType* A,
                size_t lda,
                const BType* B,
                size_t ldb,
                const MLFp16* Bias,
                MLFp16* C,
                size_t ldc,
                float* Cfloat) {
    MLAS_ACTIVATION act;
    act.ActivationKind = MlasIdentityActivation;
    std::vector<MLAS_HALF_GEMM_2FLOAT_PROCESSOR> Converters;
    Converters.reserve(BatchSize);

    std::vector<MLAS_HALF_GEMM_DATA_PARAMS> GemmParameters(BatchSize);

    for (size_t i = 0; i < GemmParameters.size(); i++) {
      auto& params = GemmParameters[i];
      params.A = A + (M * lda * i);
      params.lda = lda;
      if (nullptr != Bias) {
        params.Bias = reinterpret_cast<const MLAS_FP16*>(Bias + N * i);
      } else {
        params.Bias = nullptr;
      }
      params.C = reinterpret_cast<MLAS_FP16*>(C + (M * ldc * i));
      params.ldc = ldc;

      if (Packed) {
        ASSERT_EQ(BatchSize, size_t(1)) << "Packing B not supported in batching yet!";
        params.B = PackB(N, K, B, ldb);
        params.ldb = 0;
      } else {
        params.B = B + (K * N * i);
        params.ldb = ldb;
      }
      params.AIsfp32 = std::is_same<AType, float>::value;
      params.BIsfp32 = std::is_same<BType, float>::value;
      Converters.emplace_back(act, Cfloat + (M * N * i), N);
      params.OutputProcessor = &(Converters[i]);
    }

    MlasHalfGemmBatch(M, N, K, BatchSize, GemmParameters.data(), threadpool_);
  }

  void ReferenceQgemm(size_t M,
                      size_t N,
                      size_t K,
                      size_t BatchSize,
                      const AType* A,
                      const BType* B,
                      const MLFp16* Bias,
                      float* C) {
    // TODO!! deal with half precision accumulation error
    // Most CPUs does not support mixed precision accumulation,
    // only mul & add fuse. As a result, different striding
    // on the K dimension may lead to rounding error.
    // Accumulation of these rounding error maybe very significant.
    // So setting a approximation ratio does NOT work.
    //
    // Currently this test require a manual efforts:
    // 1. Change the K stride of the kernel under test to be 16;
    // 2. Force the K stride of the fp16 kernel to 16
    // 3. Change the test oracle to be exact match.
    // 4. Pass this test and then change it back :-(.
    //
    constexpr size_t KStride = 512;

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
              MLFp16 down(float(*b) * float(*a) + sum);
              sum = float(down);
              b += N;
              a += 1;
            }
            if (k == 0) {
              *c = sum;
            } else {
              MLFp16 d(sum + *c);
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
  MlasHalfGemmTest() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

  void Test(size_t M, size_t N, size_t K, size_t BatchSize, bool withBias) {
    const AType* A = BufferA.GetFilledBuffer(K * M * BatchSize + 16, SmallFloatFill<AType>);
    AType Atail[16];
    std::memcpy(Atail, A + K * M * BatchSize, 16 * sizeof(AType));

    const BType* B = BufferB.GetFilledBuffer(N * K * BatchSize + 16, SmallFloatFill<BType>);
    BType Btail[16];
    std::memcpy(Btail, B + N * K * BatchSize, 16 * sizeof(BType));

    MLFp16 BiasTail[16];
    const MLFp16* Bias = nullptr;
    if (withBias) {
      Bias = BufferBias.GetFilledBuffer(N * BatchSize + 16, SmallFloatFill<MLFp16>);
      std::memcpy(BiasTail, Bias + N * BatchSize, 16 * sizeof(MLFp16));
    }

    MLFp16* C = BufferC.GetFilledBuffer(N * M * BatchSize, SmallFloatFill<MLFp16>);
    float* Cfloat = BufferFloatC.GetBuffer(N * M * BatchSize, true);
    float* CReference = BufferCReference.GetFilledBuffer(
        N * M * BatchSize,
        [](float* start, size_t size) {
          std::fill_n(start, size, -1.0f);
        });

    this->CallGemm(M, N, K, BatchSize, A, K, B, N, Bias, C, N, Cfloat);
    ReferenceQgemm(M, N, K, BatchSize, A, B, Bias, CReference);

    for (size_t batch = 0, f = 0; batch < BatchSize; batch++) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++, f++) {
          ASSERT_TRUE(CloseEnough(float(C[f]), CReference[f])) << "@[" << batch << "x" << m << "x" << n << "], "
                                                               << "Batch=" << BatchSize << "M=" << M << ", N=" << N << ", K=" << K;
          ASSERT_TRUE(CloseEnough(Cfloat[f], CReference[f])) << "Converted@[" << batch << "x" << m << "x" << n << "], "
                                                             << "Batch=" << BatchSize << "M=" << M << ", N=" << N << ", K=" << K;
        }
      }
    }
    ASSERT_EQ(std::memcmp(Atail, A + K * M * BatchSize, 16 * sizeof(AType)), 0) << "Matrix A buffer overwritten!";
    ASSERT_EQ(std::memcmp(Btail, B + N * K * BatchSize, 16 * sizeof(BType)), 0) << "Matrix B buffer overwritten!";
    if (withBias) {
      ASSERT_EQ(std::memcmp(BiasTail, Bias + N * BatchSize, 16 * sizeof(MLFp16)), 0) << "Bias buffer overwritten!";
    }
  }

 private:
 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("HalfGemmFP") +
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
