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

static constexpr const char* ComputeTypeName(MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType) {
  switch (ComputeType) {
    case SQNBIT_CompFp32:
      return "Fp32";
    case SQNBIT_CompInt8:
      return "Int8";
    default:
      return "unknown";
  }
}

/**
 * @brief Test class for n-bit int block quantized GEMM
 *        Note: only 2-D matmul supported for now
 */
template <size_t BlkBitWidth, size_t BlkLen>
class MlasSQNBitGemmTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferA;
  MatrixGuardBuffer<int8_t> BufferQuantAData;
  MatrixGuardBuffer<float> BufferQuantAScale;
  MatrixGuardBuffer<float> BufferB;
  MatrixGuardBuffer<uint8_t> BufferQuantBData;
  MatrixGuardBuffer<std::byte> BufferPackedQuantBData;
  MatrixGuardBuffer<uint8_t> BufferQuantBZeroPoint;
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
                const void* /*QuantBData*/,
                const void* PackedQuantBDataWorkspace,
                const float* QuantBScale,
                const void* QuantBZeroPoint,
                const float* Bias,
                float* C,
                size_t ldc,
                void* Workspace,
                MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
                MLAS_THREADPOOL* Threadpool,
                const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig) {
    MLAS_QNBIT_GEMM_DATA_PARAMS<float> params;
    params.A = A;
    params.lda = lda;
    params.Bias = Bias;
    params.C = C;
    params.ldc = ldc;
#ifdef MLAS_TARGET_AMD64_IX86
    if (ComputeType == SQNBIT_CompInt8) {
      params.QuantBDataWorkspace = PackedQuantBDataWorkspace;
    }
#endif
    params.PackedQuantBData = static_cast<const std::byte*>(PackedQuantBDataWorkspace);
    params.QuantBScale = QuantBScale;
    params.QuantBZeroPoint = QuantBZeroPoint;
    params.PostProcessor = nullptr;

    MlasQNBitGemmBatch(M, N, K, 1, BlkBitWidth, BlkLen, ComputeType, &params, Workspace, Threadpool,
                       BackendKernelSelectorConfig);
  }

  void QuantizeA(size_t M, size_t K, const float* A, int8_t* QuantAData, float* QuantAScale) {
    const size_t BlockCountK = (K + BlkLen - 1) / BlkLen;
    const size_t lda = K;
    for (size_t m = 0; m < M; ++m) {
      for (size_t k = 0, k_blk = 0; k < K; k += BlkLen, ++k_blk) {
        const size_t local_blk_len = std::min(K - k, BlkLen);
        float blk_a[BlkLen]{};
        std::copy_n(A + m * lda + k, local_blk_len, blk_a);

        float amax = 0.0f;  // max of absolute values of A block
        for (size_t kk = 0; kk < local_blk_len; ++kk) {
          float a = blk_a[kk];
          amax = std::max(amax, fabsf(a));
        }

        constexpr float range_max = (1 << 7) - 1;
        const float scale = amax / range_max;
        const float scale_reciprocal = scale != 0.0f ? 1.0f / scale : 0.0f;

        QuantAScale[m * BlockCountK + k_blk] = scale;

        for (size_t kk = 0; kk < BlkLen; ++kk) {
          const float q = roundf(blk_a[kk] * scale_reciprocal);
          QuantAData[m * BlockCountK * BlkLen + k + kk] =
              static_cast<int8_t>(
                  std::clamp(q,
                             static_cast<float>(std::numeric_limits<int8_t>::min()),
                             static_cast<float>(std::numeric_limits<int8_t>::max())));
        }
      }
    }
  }

  void CallReferenceGemm_CompInt8(size_t M,
                                  size_t N,
                                  size_t K,
                                  const float* A,
                                  const uint8_t* QuantBData,
                                  const float* QuantBScale,
                                  const uint8_t* QuantBZeroPoint,
                                  const float* Bias,
                                  float* C) {
    const size_t BlockCountK = (K + BlkLen - 1) / BlkLen;

    int8_t* QuantAData = BufferQuantAData.GetBuffer(M * BlockCountK * BlkLen);
    float* QuantAScale = BufferQuantAScale.GetBuffer(M * BlockCountK);
    QuantizeA(M, K, A, QuantAData, QuantAScale);

    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        float sum = Bias == nullptr ? 0.0f : Bias[n];
        for (size_t k = 0, k_blk = 0; k < K; k += BlkLen, ++k_blk) {
          const size_t k_blk_len = std::min(K - k, BlkLen);

          const float a_scale = QuantAScale[m * BlockCountK + k_blk];

          const float b_scale = QuantBScale[n * BlockCountK + k_blk];

          static_assert(BlkBitWidth == 4, "only implemented for 4-bit quantized B");

          uint8_t b_zp = 8;
          if (QuantBZeroPoint != nullptr) {
            const uint8_t b_zp_byte = QuantBZeroPoint[n * ((BlockCountK + 1) / 2) + k_blk / 2];
            b_zp = (k_blk & 1) ? (b_zp_byte >> 4) : (b_zp_byte & 0x0F);
          }

          int32_t qsum = 0;

          for (size_t kk = 0; kk < k_blk_len; ++kk) {
            const int8_t qa = QuantAData[m * BlockCountK * BlkLen + k + kk];
            const uint8_t qb_byte = QuantBData[(n * BlockCountK * BlkLen + k + kk) / 2];
            const int8_t qb = ((kk & 1) == 1 ? (qb_byte >> 4) : (qb_byte & 0x0F)) - b_zp;
            qsum += qa * qb;
          }

          sum += static_cast<float>(qsum) * a_scale * b_scale;
        }

        C[m * N + n] = sum;
      }
    }
  }

  void CallReferenceGemm_CompFp32(size_t M,
                                  size_t N,
                                  size_t K,
                                  const float* A,
                                  const uint8_t* QuantBData,
                                  const float* QuantBScale,
                                  const uint8_t* QuantBZeroPoint,
                                  const float* Bias,
                                  float* C) {
    float* DequantizedBData = BufferDequantizedB.GetBuffer(K * N);
    MlasDequantizeBlockwise<float, BlkBitWidth>(
        DequantizedBData, QuantBData, QuantBScale, QuantBZeroPoint, BlkLen, /* columnwise */ true,
        static_cast<int>(K), static_cast<int>(N), GetMlasThreadPool());
    // Note: DequantizedBData is in column major layout.

    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        const float* a = A + m * K;
        const float* b = DequantizedBData + n * K;
        float* c = C + (m * N) + n;

        float sum = Bias == nullptr ? 0.0f : Bias[n];
        for (size_t k = 0; k < K; k++) {
          sum += (*a) * (*b);
          b += 1;
          a += 1;
        }
        *c = sum;
      }
    }
  }

 public:
  void Test(size_t M, size_t N, size_t K,
            MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
            bool WithThreadpool, bool Symmetric, bool WithBias,
            const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig = nullptr) {
    MLAS_THREADPOOL* Threadpool = WithThreadpool ? GetMlasThreadPool() : nullptr;

    const float* A = BufferA.GetBuffer(K * M);

    const float* B = BufferB.GetBuffer(N * K);

    const float* Bias = nullptr;
    if (WithBias) {
      Bias = BufferBias.GetBuffer(N);
    }

#if 0
    auto print_matrix = [](size_t nrows, size_t ncols, const float* data) {
      for (size_t row = 0; row < nrows; ++row) {
        for (size_t col = 0; col < ncols; ++col) {
          std::cout << data[row * ncols + col] << ", ";
        }
        std::cout << "\n";
      }
    };

    auto print_matrix_col = [](size_t nrows, size_t ncols, size_t col, const float* data) {
      for (size_t row = 0; row < nrows; ++row) {
        std::cout << data[row * ncols + col] << ", ";
      }
      std::cout << "\n";
    };

    std::cout << "A:\n";
    print_matrix(M, K, A);
    std::cout << "B:\n";
    print_matrix(K, N, B);
#endif

    float* C = BufferC.GetBuffer(N * M, true);
    float* CReference = BufferCReference.GetBuffer(N * M, true);

    // quantize B
    uint8_t* QuantBData = nullptr;
    float* QuantBScale = nullptr;
    uint8_t* QuantBZeroPoint = nullptr;
    {
      size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
      MlasBlockwiseQuantizedBufferSizes<BlkBitWidth>(BlkLen, /* columnwise */ true,
                                                     static_cast<int>(K), static_cast<int>(N),
                                                     QuantBDataSizeInBytes, QuantBScaleSize, &QuantBZeroPointSizeInBytes);

      QuantBData = BufferQuantBData.GetBuffer(QuantBDataSizeInBytes);
      QuantBScale = BufferQuantBScale.GetBuffer(QuantBScaleSize);
      if (!Symmetric) {
        QuantBZeroPoint = BufferQuantBZeroPoint.GetBuffer(QuantBZeroPointSizeInBytes);
      }

      MlasQuantizeBlockwise<float, BlkBitWidth>(QuantBData, QuantBScale, QuantBZeroPoint,
                                                B, BlkLen,
                                                /* columnwise */ true,
                                                static_cast<int>(K), static_cast<int>(N),
                                                static_cast<int>(N),
                                                GetMlasThreadPool());
    }

    void* Workspace = nullptr;
    if (const auto WorkspaceSize = MlasQNBitGemmBatchWorkspaceSize(M, N, K, 1, BlkBitWidth, BlkLen, !Symmetric,
                                                                   ComputeType, BackendKernelSelectorConfig);
        WorkspaceSize > 0) {
      Workspace = BufferWorkspace.GetBuffer(WorkspaceSize);
    }

    void* PackedQuantBDataWorkspace = nullptr;
    if (const auto PackedQuantBDataSize = MlasQNBitGemmPackQuantBDataSize(N, K, BlkBitWidth, BlkLen, !Symmetric,
                                                                          ComputeType, BackendKernelSelectorConfig);
        PackedQuantBDataSize > 0) {
      PackedQuantBDataWorkspace = BufferPackedQuantBData.GetBuffer(PackedQuantBDataSize);
      bool has_zp_input = QuantBZeroPoint != nullptr;
      MlasQNBitGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, ComputeType, QuantBData, PackedQuantBDataWorkspace,
                                  QuantBScale, has_zp_input, QuantBZeroPoint,
                                  GetMlasThreadPool(), BackendKernelSelectorConfig);
    }

    CallGemm(M, N, K,
             A, /* lda */ K,
             QuantBData, PackedQuantBDataWorkspace, QuantBScale, QuantBZeroPoint,
             Bias,
             C, /* ldc */ N,
             Workspace,
             ComputeType,
             Threadpool,
             BackendKernelSelectorConfig);

    if (ComputeType == SQNBIT_CompFp32) {
      CallReferenceGemm_CompFp32(M, N, K, A, QuantBData, QuantBScale, QuantBZeroPoint, Bias, CReference);
    } else if (ComputeType == SQNBIT_CompInt8) {
      CallReferenceGemm_CompInt8(M, N, K, A, QuantBData, QuantBScale, QuantBZeroPoint, Bias, CReference);
    } else {
      FAIL() << "Test is not implemented for compute type "
             << ComputeType << " (" << ComputeTypeName(ComputeType) << ")";
    }

    size_t f = 0;
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++, f++) {
        ASSERT_TRUE(CloseEnough(C[f], CReference[f]))
            << "Expected: " << CReference[f] << " Actual: " << C[f] << "@[" << m << "x" << n << "], "
            << "M=" << M << ", N=" << N << ", K=" << K;
      }
    }
  }

  /**
   * @brief Test CompInt8 with an explicitly built quantized B that uses the full 0..15 range of
   *        4-bit weights and, more importantly, of zero points.
   *
   * Test() above derives the zero points from random input data with MlasQuantizeBlockwise(),
   * which yields values very close to 8 for every block. That leaves the asymmetric
   * (per-element zero point subtraction) kernels effectively unverified. In particular the AVX2
   * M=1 CompInt8 kernels subtract the zero point inside the dot product instead of applying it
   * through the precomputed block sums, so they are the only ones sensitive to the actual zero
   * point value.
   */
  void TestFullRangeZeroPointCompInt8(size_t M, size_t N, size_t K, bool WithThreadpool, bool WithBias) {
    if (!MlasIsQNBitGemmAvailable(BlkBitWidth, BlkLen, SQNBIT_CompInt8)) {
      GTEST_SKIP() << "CompInt8 is not available for this block size.";
    }

    static_assert(BlkBitWidth == 4, "only implemented for 4-bit quantized B");

    MLAS_THREADPOOL* Threadpool = WithThreadpool ? GetMlasThreadPool() : nullptr;

    const size_t BlockCountK = (K + BlkLen - 1) / BlkLen;

    size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
    MlasBlockwiseQuantizedBufferSizes<BlkBitWidth>(BlkLen, /* columnwise */ true,
                                                   static_cast<int>(K), static_cast<int>(N),
                                                   QuantBDataSizeInBytes, QuantBScaleSize, &QuantBZeroPointSizeInBytes);

    float* A = BufferA.GetBuffer(K * M);
    uint8_t* QuantBData = BufferQuantBData.GetBuffer(QuantBDataSizeInBytes);
    float* QuantBScale = BufferQuantBScale.GetBuffer(QuantBScaleSize);
    uint8_t* QuantBZeroPoint = BufferQuantBZeroPoint.GetBuffer(QuantBZeroPointSizeInBytes);
    float* Bias = WithBias ? BufferBias.GetBuffer(N) : nullptr;

    // deterministic pseudo random values so that a failure is always reproducible
    uint32_t rng = 42;
    auto next_rand = [&rng]() {
      rng = rng * 1664525 + 1013904223;
      return (rng >> 16) & 0x7FFF;
    };

    for (size_t i = 0; i < K * M; ++i) {
      A[i] = static_cast<float>(static_cast<int>(next_rand() % 2001) - 1000) / 1000.0f;
    }
    for (size_t i = 0; i < QuantBDataSizeInBytes; ++i) {
      QuantBData[i] = static_cast<uint8_t>(next_rand() & 0xFF);
    }
    for (size_t i = 0; i < QuantBScaleSize; ++i) {
      QuantBScale[i] = 0.01f + 0.001f * static_cast<float>(next_rand() % 100);
    }
    // full 0..15 zero point range
    for (size_t i = 0; i < QuantBZeroPointSizeInBytes; ++i) {
      QuantBZeroPoint[i] = static_cast<uint8_t>(next_rand() & 0xFF);
    }
    if (Bias != nullptr) {
      for (size_t n = 0; n < N; ++n) {
        Bias[n] = static_cast<float>(static_cast<int>(next_rand() % 201) - 100) / 100.0f;
      }
    }

    float* C = BufferC.GetBuffer(N * M, true);

    void* Workspace = nullptr;
    if (const auto WorkspaceSize = MlasQNBitGemmBatchWorkspaceSize(M, N, K, 1, BlkBitWidth, BlkLen,
                                                                   /* has zero point */ true, SQNBIT_CompInt8, nullptr);
        WorkspaceSize > 0) {
      Workspace = BufferWorkspace.GetBuffer(WorkspaceSize);
    }

    void* PackedQuantBDataWorkspace = nullptr;
    if (const auto PackedQuantBDataSize = MlasQNBitGemmPackQuantBDataSize(N, K, BlkBitWidth, BlkLen,
                                                                          /* has zero point */ true,
                                                                          SQNBIT_CompInt8, nullptr);
        PackedQuantBDataSize > 0) {
      PackedQuantBDataWorkspace = BufferPackedQuantBData.GetBuffer(PackedQuantBDataSize);
      MlasQNBitGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, SQNBIT_CompInt8, QuantBData, PackedQuantBDataWorkspace,
                                  QuantBScale, /* has zero point */ true, QuantBZeroPoint,
                                  GetMlasThreadPool(), nullptr);
    }

    CallGemm(M, N, K,
             A, /* lda */ K,
             QuantBData, PackedQuantBDataWorkspace, QuantBScale, QuantBZeroPoint,
             Bias,
             C, /* ldc */ N,
             Workspace,
             SQNBIT_CompInt8,
             Threadpool,
             nullptr);

    // reference: same integer arithmetic as the kernels, accumulated in double
    int8_t* QuantAData = BufferQuantAData.GetBuffer(M * BlockCountK * BlkLen);
    float* QuantAScale = BufferQuantAScale.GetBuffer(M * BlockCountK);
    QuantizeA(M, K, A, QuantAData, QuantAScale);

    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        double sum = Bias == nullptr ? 0.0 : Bias[n];
        // sum of the absolute per block contributions, used to size the tolerance: the kernels
        // may compute the zero point correction as a separate (large) term that cancels against
        // the main term, so the accumulation error scales with the block magnitudes.
        double magnitude = Bias == nullptr ? 0.0 : std::abs(Bias[n]);

        for (size_t k = 0, k_blk = 0; k < K; k += BlkLen, ++k_blk) {
          const size_t k_blk_len = std::min(K - k, BlkLen);
          const float a_scale = QuantAScale[m * BlockCountK + k_blk];
          const float b_scale = QuantBScale[n * BlockCountK + k_blk];

          const uint8_t b_zp_byte = QuantBZeroPoint[n * ((BlockCountK + 1) / 2) + k_blk / 2];
          const uint8_t b_zp = (k_blk & 1) ? (b_zp_byte >> 4) : (b_zp_byte & 0x0F);

          int32_t qsum = 0;
          int32_t qmagnitude = 0;
          for (size_t kk = 0; kk < k_blk_len; ++kk) {
            const int8_t qa = QuantAData[m * BlockCountK * BlkLen + k + kk];
            const uint8_t qb_byte = QuantBData[(n * BlockCountK * BlkLen + k + kk) / 2];
            const int32_t qb = ((kk & 1) == 1 ? (qb_byte >> 4) : (qb_byte & 0x0F));
            qsum += qa * (qb - b_zp);
            qmagnitude += std::abs(qa) * std::max(qb, static_cast<int32_t>(b_zp));
          }

          sum += static_cast<double>(qsum) * a_scale * b_scale;
          magnitude += static_cast<double>(qmagnitude) * a_scale * b_scale;
        }

        const double tolerance = 1e-4 * magnitude + 1e-5;
        ASSERT_NEAR(static_cast<double>(C[m * N + n]), sum, tolerance)
            << "@[" << m << "x" << n << "], M=" << M << ", N=" << N << ", K=" << K
            << ", BlkLen=" << BlkLen;
      }
    }
  }

  void TestAsymmetricKleidiAICompInt8(size_t M, size_t N, size_t K, bool WithThreadpool, bool WithBias) {
#if defined(MLAS_TARGET_ARM64)
    MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
    config.use_kleidiai = true;
    constexpr bool HasZeroPoint = true;

    if (!MlasIsQNBitGemmAvailable(BlkBitWidth, BlkLen, SQNBIT_CompInt8) ||
        !MlasQNBitGemmScalesPacked(K, BlkBitWidth, BlkLen, SQNBIT_CompInt8, HasZeroPoint, &config)) {
      GTEST_SKIP() << "KleidiAI packed asymmetric SQ4 path is unavailable.";
    }

    ASSERT_GT(MlasQNBitGemmPackQuantBDataSize(N, K, BlkBitWidth, BlkLen, HasZeroPoint,
                                              SQNBIT_CompInt8, &config),
              0u);
    ASSERT_GT(MlasQNBitGemmBatchWorkspaceSize(M, N, K, 1, BlkBitWidth, BlkLen, HasZeroPoint,
                                              SQNBIT_CompInt8, &config),
              0u);

    Test(M, N, K, SQNBIT_CompInt8, WithThreadpool, /*Symmetric=*/false, WithBias, &config);
#else
    (void)M;
    (void)N;
    (void)K;
    (void)WithThreadpool;
    (void)WithBias;
    GTEST_SKIP() << "KleidiAI Q4 tests require ARM64.";
#endif
  }

 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("SQNBitGemm") +
                                    "BlkBitWidth" + std::to_string(BlkBitWidth) +
                                    "BlkLen" + std::to_string(BlkLen);
    return suite_name.c_str();
  }
};

//
// Short Execute() test helper to register each test separately by all parameters.
//
template <size_t BlkBitWidth, size_t BlkLen>
class SQNBitGemmShortExecuteTest : public MlasTestFixture<MlasSQNBitGemmTest<BlkBitWidth, BlkLen>> {
 public:
  explicit SQNBitGemmShortExecuteTest(size_t M, size_t N, size_t K,
                                      MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
                                      bool WithThreadpool, bool Symmetric, bool WithBias)
      : M_(M),
        N_(N),
        K_(K),
        ComputeType_(ComputeType),
        WithThreadpool_(WithThreadpool),
        Symmetric_(Symmetric),
        WithBias_(WithBias) {
  }

  void TestBody() override {
    MlasTestFixture<MlasSQNBitGemmTest<BlkBitWidth, BlkLen>>::mlas_tester->Test(
        M_, N_, K_, ComputeType_, WithThreadpool_, Symmetric_, WithBias_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K,
                                   MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
                                   bool WithThreadpool, bool Symmetric, bool WithBias) {
    size_t tests_registered = 0;

    if (MlasIsQNBitGemmAvailable(BlkBitWidth, BlkLen, ComputeType)) {
      std::stringstream ss;
      ss << (WithThreadpool ? "SingleThread" : "Threaded")
         << "/isSymmetric" << Symmetric
         << "/M" << M << "xN" << N << "xK" << K
         << "/hasBias" << WithBias
         << "/computeType" << ComputeTypeName(ComputeType);
      auto test_name = ss.str();

      testing::RegisterTest(
          MlasSQNBitGemmTest<BlkBitWidth, BlkLen>::GetTestSuiteName(),
          test_name.c_str(),
          nullptr,
          test_name.c_str(),
          __FILE__,
          __LINE__,
          // Important to use the fixture type as the return type here.
          [=]() -> MlasTestFixture<MlasSQNBitGemmTest<BlkBitWidth, BlkLen>>* {
            return new SQNBitGemmShortExecuteTest(
                M, N, K, ComputeType, WithThreadpool, Symmetric, WithBias);
          });

      tests_registered += 1;
    }

    return tests_registered;
  }

  static size_t RegisterShortExecuteTests() {
    size_t tests_registered = 0;

    for (MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType : {SQNBIT_CompFp32, SQNBIT_CompInt8}) {
      for (bool WithThreadpool : {false, true}) {
        for (bool Symmetric : {false, true}) {
          for (size_t b = 1; b < 16; b++) {
            tests_registered += RegisterSingleTest(b, b, b, ComputeType, WithThreadpool, Symmetric, false);
            tests_registered += RegisterSingleTest(b, b, b, ComputeType, WithThreadpool, Symmetric, true);
          }
          for (size_t b = 16; b <= 256; b <<= 1) {
            tests_registered += RegisterSingleTest(b, b, b, ComputeType, WithThreadpool, Symmetric, false);
            tests_registered += RegisterSingleTest(b, b, b, ComputeType, WithThreadpool, Symmetric, true);
          }
          for (size_t b = 256; b < 320; b += 32) {
            tests_registered += RegisterSingleTest(b, b, b, ComputeType, WithThreadpool, Symmetric, true);
          }
          for (size_t b = 1; b < 96; b++) {
            tests_registered += RegisterSingleTest(1, b, 32, ComputeType, WithThreadpool, Symmetric, false);
            tests_registered += RegisterSingleTest(1, 32, b, ComputeType, WithThreadpool, Symmetric, true);
            tests_registered += RegisterSingleTest(1, b, b, ComputeType, WithThreadpool, Symmetric, false);
          }
          tests_registered += RegisterSingleTest(43, 500, 401, ComputeType, WithThreadpool, Symmetric, true);
          tests_registered += RegisterSingleTest(1, 2, 16, ComputeType, WithThreadpool, Symmetric, true);
          tests_registered += RegisterSingleTest(1, 2, 16, ComputeType, WithThreadpool, Symmetric, false);
          tests_registered += RegisterSingleTest(1, 1027, 1031, ComputeType, WithThreadpool, Symmetric, false);
          tests_registered += RegisterSingleTest(11, 1027, 1031, ComputeType, WithThreadpool, Symmetric, false);
          tests_registered += RegisterSingleTest(1, 1027, 1031, ComputeType, WithThreadpool, Symmetric, true);
          tests_registered += RegisterSingleTest(11, 1027, 1031, ComputeType, WithThreadpool, Symmetric, true);
          tests_registered += RegisterSingleTest(1, 527, 2131, ComputeType, WithThreadpool, Symmetric, false);
          tests_registered += RegisterSingleTest(11, 527, 2131, ComputeType, WithThreadpool, Symmetric, false);
          tests_registered += RegisterSingleTest(1, 527, 2131, ComputeType, WithThreadpool, Symmetric, true);
          tests_registered += RegisterSingleTest(11, 527, 2131, ComputeType, WithThreadpool, Symmetric, true);
          // tests_registered += RegisterSingleTest(1001, 1027, 1031, ComputeType, WithThreadpool, Symmetric, false);
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

class SQNBitGemmKleidiAIShortExecuteTest : public MlasTestFixture<MlasSQNBitGemmTest<4, 128>> {
 public:
  explicit SQNBitGemmKleidiAIShortExecuteTest(size_t M, size_t N, size_t K,
                                              bool WithThreadpool, bool WithBias)
      : M_(M),
        N_(N),
        K_(K),
        WithThreadpool_(WithThreadpool),
        WithBias_(WithBias) {
  }

  void TestBody() override {
    MlasTestFixture<MlasSQNBitGemmTest<4, 128>>::mlas_tester->TestAsymmetricKleidiAICompInt8(
        M_, N_, K_, WithThreadpool_, WithBias_);
  }

  static size_t RegisterSingleTest(const char* test_name, size_t M, size_t N, size_t K,
                                   bool WithThreadpool, bool WithBias) {
    testing::RegisterTest(
        MlasSQNBitGemmTest<4, 128>::GetTestSuiteName(),
        test_name,
        nullptr,
        test_name,
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<MlasSQNBitGemmTest<4, 128>>* {
          return new SQNBitGemmKleidiAIShortExecuteTest(M, N, K, WithThreadpool, WithBias);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t tests_registered = 0;

    tests_registered += RegisterSingleTest(
        "KleidiAIAsymGemv_M1_N257_K128", 1, 257, 128, /*WithThreadpool=*/false, /*WithBias=*/true);
    tests_registered += RegisterSingleTest(
        "KleidiAIAsymGemm_M5_N257_K128", 5, 257, 128, /*WithThreadpool=*/true, /*WithBias=*/true);
    tests_registered += RegisterSingleTest(
        "KleidiAIAsymGemv_M1_N288_K1024_NoBias", 1, 288, 1024, /*WithThreadpool=*/false, /*WithBias=*/false);

    return tests_registered;
  }

 private:
  size_t M_, N_, K_;
  bool WithThreadpool_, WithBias_;
};

/**
 * @brief Registers CompInt8 tests with an explicitly built quantized B that covers the full
 *        zero point range. See MlasSQNBitGemmTest::TestFullRangeZeroPointCompInt8().
 */
template <size_t BlkBitWidth, size_t BlkLen>
class SQNBitGemmFullRangeZeroPointShortExecuteTest : public MlasTestFixture<MlasSQNBitGemmTest<BlkBitWidth, BlkLen>> {
 public:
  explicit SQNBitGemmFullRangeZeroPointShortExecuteTest(size_t M, size_t N, size_t K,
                                                        bool WithThreadpool, bool WithBias)
      : M_(M), N_(N), K_(K), WithThreadpool_(WithThreadpool), WithBias_(WithBias) {
  }

  void TestBody() override {
    MlasTestFixture<MlasSQNBitGemmTest<BlkBitWidth, BlkLen>>::mlas_tester->TestFullRangeZeroPointCompInt8(
        M_, N_, K_, WithThreadpool_, WithBias_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, bool WithThreadpool, bool WithBias) {
    std::stringstream ss;
    ss << "FullRangeZeroPoint/" << (WithThreadpool ? "Threaded" : "SingleThread")
       << "/M" << M << "xN" << N << "xK" << K
       << "/hasBias" << WithBias
       << "/computeTypeInt8";
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasSQNBitGemmTest<BlkBitWidth, BlkLen>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<MlasSQNBitGemmTest<BlkBitWidth, BlkLen>>* {
          return new SQNBitGemmFullRangeZeroPointShortExecuteTest(M, N, K, WithThreadpool, WithBias);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t tests_registered = 0;

    // M=1 uses dedicated GEMV kernels on some platforms (e.g. the AVX2 CompInt8 M=1 kernels,
    // which are the only ones applying the zero point per element inside the dot product).
    // The N values cover the 4 column blocking and its 1..3 column remainder, the K values
    // cover even and odd block counts and a partial trailing block.
    for (size_t N : {1, 2, 3, 4, 7, 16, 96, 130}) {
      for (size_t K : {32, 96, 129, 256}) {
        tests_registered += RegisterSingleTest(1, N, K, /*WithThreadpool=*/false, /*WithBias=*/false);
      }
    }
    tests_registered += RegisterSingleTest(1, 96, 256, /*WithThreadpool=*/false, /*WithBias=*/true);
    tests_registered += RegisterSingleTest(1, 1027, 1031, /*WithThreadpool=*/true, /*WithBias=*/true);

    // M > 1 for comparison: these go through the block sum based zero point correction.
    tests_registered += RegisterSingleTest(2, 96, 256, /*WithThreadpool=*/false, /*WithBias=*/false);
    tests_registered += RegisterSingleTest(11, 96, 129, /*WithThreadpool=*/false, /*WithBias=*/true);

    return tests_registered;
  }

 private:
  size_t M_, N_, K_;
  bool WithThreadpool_, WithBias_;
};

static size_t SQNBitGemmRegisterAllShortExecuteTests() {
  size_t count = 0;

  count += SQNBitGemmShortExecuteTest<4, 16>::RegisterShortExecuteTests();
  count += SQNBitGemmShortExecuteTest<4, 32>::RegisterShortExecuteTests();
  count += SQNBitGemmShortExecuteTest<4, 64>::RegisterShortExecuteTests();
  count += SQNBitGemmShortExecuteTest<4, 128>::RegisterShortExecuteTests();
  count += SQNBitGemmShortExecuteTest<4, 256>::RegisterShortExecuteTests();
  count += SQNBitGemmFullRangeZeroPointShortExecuteTest<4, 16>::RegisterShortExecuteTests();
  count += SQNBitGemmFullRangeZeroPointShortExecuteTest<4, 32>::RegisterShortExecuteTests();
  count += SQNBitGemmFullRangeZeroPointShortExecuteTest<4, 64>::RegisterShortExecuteTests();
  count += SQNBitGemmFullRangeZeroPointShortExecuteTest<4, 128>::RegisterShortExecuteTests();
  count += SQNBitGemmFullRangeZeroPointShortExecuteTest<4, 256>::RegisterShortExecuteTests();
  count += SQNBitGemmKleidiAIShortExecuteTest::RegisterShortExecuteTests();

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return SQNBitGemmRegisterAllShortExecuteTests();
      }
      return 0;
    });
