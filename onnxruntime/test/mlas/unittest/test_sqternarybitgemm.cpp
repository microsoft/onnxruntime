/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_sqnbitgemm.h

Abstract:

    Tests for MLAS n-bit int block quantized GEMM.

--*/

#include "test_util.h"
#include "mlas_q4.h"
#include "mlas_qnbit.h"
#include <cstring>

/**
 * @brief Test class for n-bit int block quantized GEMM
 *        Note: only 2-D matmul supported for now
 */
class MlasSQTernaryBitGemmTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferA;
  MatrixGuardBuffer<int8_t> BufferQuantAData;
  MatrixGuardBuffer<float> BufferQuantAScale;
  MatrixGuardBuffer<float> BufferB;
  MatrixGuardBuffer<uint8_t> BufferQuantBData;
  MatrixGuardBuffer<std::byte> BufferPackedQuantBData;
  MatrixGuardBuffer<float> BufferQuantBScale;
  MatrixGuardBuffer<float> BufferDequantizedB;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<std::byte> BufferWorkspace;
  MatrixGuardBuffer<float> BufferC;
  MatrixGuardBuffer<float> BufferCReference;

  void CallGemm(size_t M,
                size_t N,
                size_t K,
                const float* A,
                size_t lda,
                const void* QuantBData,
                const float* Bias,
                float* C,
                size_t ldc,
                void* Workspace,
                MLAS_THREADPOOL* Threadpool) {
    MLAS_QNBIT_GEMM_DATA_PARAMS<float> params;
    params.A = A;
    params.lda = lda;
    params.Bias = Bias;
    params.C = C;
    params.ldc = ldc;
    params.PackedQuantBData = (std::byte*)QuantBData;
    params.QuantBScale = nullptr;
    params.PostProcessor = nullptr;

    MlasQNBitGemmBatch(M, N, K, 1, 0, 256, SQNBIT_CompInt8, &params, Workspace, Threadpool, TQ1_0);
  }

  void CallReferenceGemm(
    size_t M,
    size_t N,
    size_t K,
    const float* A,
    const float* B,
    const float* Bias,
    float* C) {
    float* c = C;
    for (size_t m = 0; m < M; ++m) {
        const float* a = A + m * K;
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a[k] * B[n * K + k]; // column major
            }
            if (Bias != nullptr) {
                sum += Bias[n];
            }
            c[m * N + n] = sum;
        }
    }
  }

 public:
  void Test(size_t M, size_t N, size_t K, bool WithThreadpool, bool WithBias) {
    MLAS_THREADPOOL* Threadpool = WithThreadpool ? GetMlasThreadPool() : nullptr;

    const float* A = BufferA.GetBuffer(K * M);

    const float* B = BufferB.GetBuffer(N * K);

    const float* Bias = nullptr;
    if (WithBias) {
      Bias = BufferBias.GetBuffer(N);
    }

    float* C = BufferC.GetBuffer(N * M, true);
    float* CReference = BufferCReference.GetBuffer(N * M, true);

    // quantize B
    uint8_t* QuantBData = nullptr;
    size_t QuantBDataSizeInBytes = QTernaryBitGemmPerGemmWorkspaceSize(M, N, K, QK_K, SQNBIT_CompInt8);
    QuantBData = BufferQuantBData.GetBuffer(QuantBDataSizeInBytes);
    MlasQuantizeBlockwise_Q1_0(QuantBData, B, static_cast<int>(K), static_cast<int>(N));
    void* Workspace = nullptr;
    const size_t WorkspaceSize = MlasQNBitGemmBatchWorkspaceSize(M, N, K, 1, 0, QK_K, SQNBIT_CompInt8, TQ1_0);
    Workspace = BufferWorkspace.GetBuffer(WorkspaceSize);

#if 1
    // do round trip such that it shall be an idempotent mapping after inital quantization
    std::vector<std::byte> QuantA(WorkspaceSize), QuantA2(WorkspaceSize);
    std::vector<float> A2(K * M);
    //QuantizeARow_Q8_K(QK_K, A, K, &QuantA[0]);
    Quantize_Q8_K(QK_K, A, M, K, K, &QuantA[0]);
    //DequantizeARow_Q8_K(QK_K, &A2[0], K, &QuantA[0]);
    Dequantize_Q8_K(QK_K, &A2[0], M, K, K, &QuantA[0]);
    //QuantizeARow_Q8_K(QK_K, &A2[0], K, &QuantA2[0]);
    Quantize_Q8_K(QK_K, &A2[0], M, K, K, &QuantA2[0]);
    block_q8_K* x0 = reinterpret_cast<block_q8_K*>(&QuantA[0]);
    block_q8_K* x1 = reinterpret_cast<block_q8_K*>(&QuantA2[0]);
    bool pass_round_trip_test =
        std::memcmp(x0->qs, x1->qs, sizeof(x0->qs)) == 0 &&
        std::memcmp(x0->bsums, x1->bsums, sizeof(x0->bsums)) == 0 &&
        x0->d == x1->d;
    assert(pass_round_trip_test);

    std::vector<float> DequantBData(N * K);
    std::vector<uint8_t> QuantBData2(QuantBDataSizeInBytes);
    MlasDequantizeBlockwise_Q1_0(&DequantBData[0], QuantBData, static_cast<int>(K), static_cast<int>(N));

    MlasQuantizeBlockwise_Q1_0(&QuantBData2[0], &DequantBData[0], static_cast<int>(K), static_cast<int>(N));
    block_tq1_0* y0 = reinterpret_cast<block_tq1_0*>(&QuantBData[0]);
    block_tq1_0* y1 = reinterpret_cast<block_tq1_0*>(&QuantBData2[0]);
    pass_round_trip_test =
        std::memcmp(y0->qs, y1->qs, sizeof(y0->qs)) == 0 &&
        std::memcmp(y0->qh, y1->qh, sizeof(y0->qh)) == 0 &&
        y0->d == y1->d;
    assert(pass_round_trip_test);

    CallGemm(M, N, K,
             &A2[0],
             /* lda */ K,
             QuantBData,
             Bias,
             C,
             /* ldc */ N,
             Workspace,
             Threadpool);

    CallReferenceGemm(M, N, K, &A2[0], &DequantBData[0], Bias, CReference);
#else
    CallGemm(M, N, K,
             A, /* lda */ K,
             QuantBData,
             Bias,
             C, /* ldc */ N,
             Workspace,
             Threadpool);

    CallReferenceGemm(M, N, K, A, B, Bias, CReference);
#endif

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
    static std::string suite_name = std::string("SQTernaryBitGemm");
    return suite_name.c_str();
  }
};

//
// Short Execute() test helper to register each test separately by all parameters.
//
class SQNBitGemmShortExecuteTest : public MlasTestFixture<MlasSQTernaryBitGemmTest> {
 public:
  explicit SQNBitGemmShortExecuteTest(size_t M, size_t N, size_t K, bool WithThreadpool, bool WithBias)
      : M_(M),
        N_(N),
        K_(K),
        WithThreadpool_(WithThreadpool),
        WithBias_(WithBias) {
  }

  void TestBody() override {
    MlasTestFixture<MlasSQTernaryBitGemmTest>::mlas_tester->Test(
        M_, N_, K_, WithThreadpool_, WithBias_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K,
                                   bool WithThreadpool, bool WithBias) {
    size_t tests_registered = 0;

    if (MlasIsQNBitGemmAvailable(0, QK_K, SQNBIT_CompInt8, TQ1_0)) {
      std::stringstream ss;
      ss << (WithThreadpool ? "SingleThread" : "Threaded")
         << "/M" << M << "xN" << N << "xK" << K
         << "/hasBias" << WithBias;
      auto test_name = ss.str();

      testing::RegisterTest(
          MlasSQTernaryBitGemmTest::GetTestSuiteName(),
          test_name.c_str(),
          nullptr,
          test_name.c_str(),
          __FILE__,
          __LINE__,
          // Important to use the fixture type as the return type here.
          [=]() -> MlasTestFixture<MlasSQTernaryBitGemmTest>* {
            return new SQNBitGemmShortExecuteTest(M, N, K, WithThreadpool, WithBias);
          });

      tests_registered += 1;
    }

    return tests_registered;
  }

  static size_t RegisterShortExecuteTests() {
    size_t tests_registered = 0;
    for (size_t M : {1, 2}) {
      for (size_t N : {1, 2}) {
        for (size_t K : {256, 512}) {
          for (bool use_thread_pool : {true, false}) {
            for (bool has_bias : {false}) {
              tests_registered += RegisterSingleTest(M, N, K, use_thread_pool, has_bias);
            }
          }
        }
      }
    }
    return tests_registered;
  }

 private:
  size_t M_, N_, K_;
  MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType_;
  bool WithThreadpool_, Symmetric_, WithBias_;
};

static size_t SQNBitGemmRegisterAllShortExecuteTests() {
  size_t count = 0;
  // TODO: enable these test for 2bit development.
  //count += SQNBitGemmShortExecuteTest<2, 16>::RegisterShortExecuteTests();
  //count += SQNBitGemmShortExecuteTest<2, 32>::RegisterShortExecuteTests();
  //count += SQNBitGemmShortExecuteTest<2, 64>::RegisterShortExecuteTests();
  //count += SQNBitGemmShortExecuteTest<2, 128>::RegisterShortExecuteTests();
  //count += SQNBitGemmShortExecuteTest<2, 256>::RegisterShortExecuteTests();

  count += SQNBitGemmShortExecuteTest::RegisterShortExecuteTests();
  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return SQNBitGemmRegisterAllShortExecuteTests();
      }
      return 0;
    });
