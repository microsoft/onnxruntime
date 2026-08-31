/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_sqnbitgemm.h

Abstract:

    Tests for MLAS n-bit int block quantized GEMM.

--*/

#include <limits>
#include <stdexcept>

#include "test_util.h"
#include "mlas_q4.h"
#include "mlas_qnbit.h"

#if defined(MLAS_TARGET_ARM64) && defined(USE_KLEIDIAI)
#include "core/mlas/lib/mlasi.h"
#if defined(MLAS_ENABLE_TEST_HOOKS)
#include "core/mlas/lib/kleidiai/mlasi_kleidiai.h"
#endif
#endif

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
  class AffinePostProcessor final : public MLAS_GEMM_POSTPROCESSOR<float> {
   public:
    void Process(float* C, size_t RangeStartM, size_t RangeStartN,
                 size_t RangeCountM, size_t RangeCountN, size_t ldc) const override {
      for (size_t m = RangeStartM; m < RangeStartM + RangeCountM; ++m) {
        for (size_t n = RangeStartN; n < RangeStartN + RangeCountN; ++n) {
          C[m * ldc + n] = C[m * ldc + n] * 2.0f + 1.0f;
        }
      }
    }
  };

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
                size_t BatchN,
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
                const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig,
                MLAS_GEMM_POSTPROCESSOR<float>* PostProcessor) {
    auto params = std::make_unique<MLAS_QNBIT_GEMM_DATA_PARAMS<float>[]>(BatchN);
    for (size_t batch = 0; batch < BatchN; ++batch) {
      auto& param = params[batch];
      param.A = A + batch * M * lda;
      param.lda = lda;
      param.Bias = Bias;
      param.C = C + batch * M * ldc;
      param.ldc = ldc;
#ifdef MLAS_TARGET_AMD64_IX86
      if (ComputeType == SQNBIT_CompInt8) {
        param.QuantBDataWorkspace = PackedQuantBDataWorkspace;
      }
#endif
      param.PackedQuantBData = static_cast<const std::byte*>(PackedQuantBDataWorkspace);
      param.QuantBScale = QuantBScale;
      param.QuantBZeroPoint = QuantBZeroPoint;
      param.PostProcessor = PostProcessor;
    }

    MlasQNBitGemmBatch(M, N, K, BatchN, BlkBitWidth, BlkLen, ComputeType, params.get(), Workspace, Threadpool,
                       BackendKernelSelectorConfig);
  }

  void QuantizeA(size_t M, size_t K, const float* A, size_t lda,
                 int8_t* QuantAData, float* QuantAScale) {
    const size_t BlockCountK = (K + BlkLen - 1) / BlkLen;
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
                                  size_t lda,
                                  const uint8_t* QuantBData,
                                  const float* QuantBScale,
                                  const uint8_t* QuantBZeroPoint,
                                  const float* Bias,
                                  float* C,
                                  size_t ldc) {
    const size_t BlockCountK = (K + BlkLen - 1) / BlkLen;

    int8_t* QuantAData = BufferQuantAData.GetBuffer(M * BlockCountK * BlkLen);
    float* QuantAScale = BufferQuantAScale.GetBuffer(M * BlockCountK);
    QuantizeA(M, K, A, lda, QuantAData, QuantAScale);

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

        C[m * ldc + n] = sum;
      }
    }
  }

  void CallReferenceGemm_CompFp32(size_t M,
                                  size_t N,
                                  size_t K,
                                  const float* A,
                                  size_t lda,
                                  const uint8_t* QuantBData,
                                  const float* QuantBScale,
                                  const uint8_t* QuantBZeroPoint,
                                  const float* Bias,
                                  float* C,
                                  size_t ldc) {
    float* DequantizedBData = BufferDequantizedB.GetBuffer(K * N);
    MlasDequantizeBlockwise<float, BlkBitWidth>(
        DequantizedBData, QuantBData, QuantBScale, QuantBZeroPoint, BlkLen, /* columnwise */ true,
        static_cast<int>(K), static_cast<int>(N), GetMlasThreadPool());
    // Note: DequantizedBData is in column major layout.

    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        const float* a = A + m * lda;
        const float* b = DequantizedBData + n * K;
        float* c = C + (m * ldc) + n;

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
            const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig = nullptr,
            size_t BatchN = 1, size_t Lda = 0, size_t Ldc = 0, bool WithPostProcessor = false) {
    MLAS_THREADPOOL* Threadpool = WithThreadpool ? GetMlasThreadPool() : nullptr;
    const size_t lda = Lda == 0 ? K : Lda;
    const size_t ldc = Ldc == 0 ? N : Ldc;

    const float* A = BufferA.GetBuffer(lda * M * BatchN);

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

    float* C = BufferC.GetBuffer(ldc * M * BatchN, true);
    float* CReference = BufferCReference.GetBuffer(ldc * M * BatchN, true);

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
    if (const auto WorkspaceSize = MlasQNBitGemmBatchWorkspaceSize(M, N, K, BatchN, BlkBitWidth, BlkLen,
                                                                   !Symmetric, ComputeType,
                                                                   BackendKernelSelectorConfig);
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

    AffinePostProcessor post_processor;
    auto* post_processor_ptr = WithPostProcessor ? &post_processor : nullptr;
    CallGemm(M, N, K, BatchN,
             A, lda,
             QuantBData, PackedQuantBDataWorkspace, QuantBScale, QuantBZeroPoint,
             Bias,
             C, ldc,
             Workspace,
             ComputeType,
             Threadpool,
             BackendKernelSelectorConfig,
             post_processor_ptr);

    for (size_t batch = 0; batch < BatchN; ++batch) {
      const float* BatchA = A + batch * M * lda;
      float* BatchCReference = CReference + batch * M * ldc;
      if (ComputeType == SQNBIT_CompFp32) {
        CallReferenceGemm_CompFp32(
            M, N, K, BatchA, lda, QuantBData, QuantBScale, QuantBZeroPoint, Bias, BatchCReference, ldc);
      } else if (ComputeType == SQNBIT_CompInt8) {
        CallReferenceGemm_CompInt8(
            M, N, K, BatchA, lda, QuantBData, QuantBScale, QuantBZeroPoint, Bias, BatchCReference, ldc);
      } else {
        FAIL() << "Test is not implemented for compute type "
               << ComputeType << " (" << ComputeTypeName(ComputeType) << ")";
      }

      if (post_processor_ptr != nullptr) {
        post_processor.Process(BatchCReference, 0, 0, M, N, ldc);
      }
    }

    for (size_t batch = 0; batch < BatchN; ++batch) {
      for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
          const size_t f = batch * M * ldc + m * ldc + n;
          ASSERT_TRUE(CloseEnough(C[f], CReference[f]))
              << "Expected: " << CReference[f] << " Actual: " << C[f] << "@[" << m << "x" << n << "], "
              << "M=" << M << ", N=" << N << ", K=" << K << ", lda=" << lda << ", ldc=" << ldc
              << ", batch=" << batch;
        }
      }
    }
  }

  void TestAsymmetricKleidiAICompInt8(size_t M, size_t N, size_t K,
                                      bool WithThreadpool, bool WithBias, size_t BatchN = 1,
                                      size_t Lda = 0, size_t Ldc = 0, bool WithPostProcessor = false) {
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
    ASSERT_GT(MlasQNBitGemmBatchWorkspaceSize(M, N, K, BatchN, BlkBitWidth, BlkLen, HasZeroPoint,
                                              SQNBIT_CompInt8, &config),
              0u);

    Test(M, N, K, SQNBIT_CompInt8, WithThreadpool, /*Symmetric=*/false, WithBias, &config, BatchN, Lda, Ldc,
         WithPostProcessor);
#else
    (void)M;
    (void)N;
    (void)K;
    (void)BatchN;
    (void)Lda;
    (void)Ldc;
    (void)WithPostProcessor;
    (void)WithThreadpool;
    (void)WithBias;
    GTEST_SKIP() << "KleidiAI Q4 tests require ARM64.";
#endif
  }

  void TestSymmetricKleidiAICompInt8(size_t M, size_t N, size_t K, bool WithThreadpool, bool WithBias,
                                     size_t Lda = 0, size_t Ldc = 0) {
#if defined(MLAS_TARGET_ARM64) && defined(USE_KLEIDIAI) && defined(MLAS_ENABLE_TEST_HOOKS)
    const auto& cpuid = MLAS_CPUIDINFO::GetCPUIDInfo();
    if (!cpuid.HasArm_SME2() && !cpuid.HasArmNeon_I8MM() && !cpuid.HasArmNeonDot()) {
      GTEST_SKIP() << "KleidiAI symmetric Q4 tests require SME2, I8MM, or DotProd.";
    }

    MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
    config.use_kleidiai = true;
    constexpr bool HasZeroPoint = false;

    ASSERT_TRUE(MlasIsQNBitGemmAvailable(BlkBitWidth, BlkLen, SQNBIT_CompInt8));
    ASSERT_TRUE(MlasQNBitGemmScalesPacked(K, BlkBitWidth, BlkLen, SQNBIT_CompInt8, HasZeroPoint, &config));

    const char* selected_kernel = M == 1
                                      ? ArmKleidiAI::GetKleidiAIQ4GemvKernelNameForTesting()
                                      : ArmKleidiAI::GetKleidiAIQ4GemmKernelNameForTesting();
    const char* expected_kernel;
    if (cpuid.HasArm_SME2()) {
      expected_kernel = M == 1
                            ? "kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot"
                            : "kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa";
    } else if (cpuid.HasArmNeon_I8MM()) {
      expected_kernel = M == 1
                            ? "kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod"
                            : "kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm";
    } else {
      expected_kernel = M == 1
                            ? "kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod"
                            : "kai_run_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod";
    }
    ASSERT_STREQ(selected_kernel, expected_kernel);

    ASSERT_GT(MlasQNBitGemmPackQuantBDataSize(N, K, BlkBitWidth, BlkLen, HasZeroPoint,
                                              SQNBIT_CompInt8, &config),
              0u);
    ASSERT_GT(MlasQNBitGemmBatchWorkspaceSize(M, N, K, 1, BlkBitWidth, BlkLen, HasZeroPoint,
                                              SQNBIT_CompInt8, &config),
              0u);

    Test(M, N, K, SQNBIT_CompInt8, WithThreadpool, /*Symmetric=*/true, WithBias, &config,
         /*BatchN=*/1, Lda, Ldc);
#else
    (void)M;
    (void)N;
    (void)K;
    (void)WithThreadpool;
    (void)WithBias;
    (void)Lda;
    (void)Ldc;
    GTEST_SKIP() << "KleidiAI symmetric Q4 tests require an ARM64 KleidiAI test-hook build.";
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

TEST(SQNBitGemmKleidiAIValidation, ZeroKRejectsKleidiAIOverride) {
#if defined(MLAS_TARGET_ARM64) && defined(USE_KLEIDIAI)
  constexpr size_t BlkBitWidth = 4;
  constexpr size_t BlkLen = 128;
  constexpr size_t SupportedK = BlkLen;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  config.use_kleidiai = true;

  const auto is_supported = GetMlasPlatform().MlasQNBitGemmIsSupportedOverride;
  if (is_supported == nullptr ||
      !is_supported(SupportedK, BlkBitWidth, BlkLen, false, SQNBIT_CompInt8, &config)) {
    GTEST_SKIP() << "KleidiAI symmetric SQ4 path is unavailable.";
  }

  for (bool has_zero_point : {false, true}) {
    EXPECT_FALSE(is_supported(
        0, BlkBitWidth, BlkLen, has_zero_point, SQNBIT_CompInt8, &config));
  }
#else
  GTEST_SKIP() << "KleidiAI Q4 tests require an Arm64 KleidiAI build.";
#endif
}

TEST(SQNBitGemmKleidiAIValidation, ZeroMOrNIsNoOp) {
#if defined(MLAS_TARGET_ARM64) && defined(USE_KLEIDIAI)
  constexpr size_t BlkBitWidth = 4;
  constexpr size_t BlkLen = 128;
  constexpr size_t M = 1;
  constexpr size_t N = 16;
  constexpr size_t K = BlkLen;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  config.use_kleidiai = true;

  const auto is_supported = GetMlasPlatform().MlasQNBitGemmIsSupportedOverride;
  if (is_supported == nullptr ||
      !is_supported(K, BlkBitWidth, BlkLen, false, SQNBIT_CompInt8, &config)) {
    GTEST_SKIP() << "KleidiAI symmetric SQ4 path is unavailable.";
  }

  MLAS_THREADPOOL* thread_pool = GetMlasThreadPool();
  if (thread_pool == nullptr) {
    GTEST_SKIP() << "Threaded KleidiAI Q4 test requires an MLAS thread pool.";
  }

  MLAS_QNBIT_GEMM_DATA_PARAMS<float> params{};
  // Successful return from each call is the condition under test.
  MlasQNBitGemmBatch(
      0, N, K, 1, BlkBitWidth, BlkLen, SQNBIT_CompInt8,
      &params, nullptr, thread_pool, &config);
  MlasQNBitGemmBatch(
      M, 0, K, 1, BlkBitWidth, BlkLen, SQNBIT_CompInt8,
      &params, nullptr, thread_pool, &config);
#else
  GTEST_SKIP() << "KleidiAI Q4 tests require an Arm64 KleidiAI build.";
#endif
}

#if !defined(ORT_NO_EXCEPTIONS)
TEST(SQNBitGemmKleidiAIValidation, MixedZeroPointBatchRejected) {
#if defined(MLAS_TARGET_ARM64) && defined(USE_KLEIDIAI)
  constexpr size_t BlkBitWidth = 4;
  constexpr size_t BlkLen = 128;
  constexpr size_t M = 1;
  constexpr size_t N = 1;
  constexpr size_t K = 128;
  constexpr size_t BatchN = 2;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  config.use_kleidiai = true;

  if (!MlasIsQNBitGemmAvailable(BlkBitWidth, BlkLen, SQNBIT_CompInt8) ||
      !MlasQNBitGemmScalesPacked(K, BlkBitWidth, BlkLen, SQNBIT_CompInt8, false, &config)) {
    GTEST_SKIP() << "KleidiAI packed symmetric SQ4 path is unavailable.";
  }

  MLAS_QNBIT_GEMM_DATA_PARAMS<float> params[BatchN]{};
  std::byte zero_point{};
  params[1].QuantBZeroPoint = &zero_point;

  EXPECT_THROW(
      MlasQNBitGemmBatch(
          M, N, K, BatchN, BlkBitWidth, BlkLen, SQNBIT_CompInt8, params, nullptr, nullptr, &config),
      std::invalid_argument);
#else
  GTEST_SKIP() << "KleidiAI Q4 tests require an Arm64 KleidiAI build.";
#endif
}

TEST(SQNBitGemmKleidiAIValidation, SizeOverflowRejected) {
#if defined(MLAS_TARGET_ARM64) && defined(USE_KLEIDIAI)
  constexpr size_t BlkBitWidth = 4;
  constexpr size_t BlkLen = 128;
  constexpr size_t K = 256;
  constexpr bool HasZeroPoint = false;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config;
  config.use_kleidiai = true;

  if (!MlasIsQNBitGemmAvailable(BlkBitWidth, BlkLen, SQNBIT_CompInt8) ||
      !MlasQNBitGemmScalesPacked(K, BlkBitWidth, BlkLen, SQNBIT_CompInt8, HasZeroPoint, &config)) {
    GTEST_SKIP() << "KleidiAI packed symmetric SQ4 path is unavailable.";
  }

  std::byte quant_b_data{};
  std::byte packed_quant_b_data{};
  float quant_b_scale = 1.0f;
  EXPECT_THROW(
      MlasQNBitGemmPackQuantBData(
          std::numeric_limits<size_t>::max() / 2 + 1, K, BlkBitWidth, BlkLen, SQNBIT_CompInt8,
          &quant_b_data, &packed_quant_b_data,
          &quant_b_scale, HasZeroPoint, nullptr, nullptr, &config),
      std::overflow_error);

  EXPECT_THROW(
      (void)MlasQNBitGemmBatchWorkspaceSize(
          1, 1, K, std::numeric_limits<size_t>::max(), BlkBitWidth, BlkLen, HasZeroPoint,
          SQNBIT_CompInt8, &config),
      std::overflow_error);
#else
  GTEST_SKIP() << "KleidiAI Q4 tests require an Arm64 KleidiAI build.";
#endif
}
#endif  // !defined(ORT_NO_EXCEPTIONS)

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
                                              bool WithThreadpool, bool Symmetric, bool WithBias,
                                              size_t BatchN = 1, size_t Lda = 0, size_t Ldc = 0,
                                              bool WithPostProcessor = false)
      : M_(M),
        N_(N),
        K_(K),
        BatchN_(BatchN),
        Lda_(Lda),
        Ldc_(Ldc),
        WithThreadpool_(WithThreadpool),
        Symmetric_(Symmetric),
        WithBias_(WithBias),
        WithPostProcessor_(WithPostProcessor) {
  }

  void TestBody() override {
    if (Symmetric_) {
      MlasTestFixture<MlasSQNBitGemmTest<4, 128>>::mlas_tester->TestSymmetricKleidiAICompInt8(
          M_, N_, K_, WithThreadpool_, WithBias_, Lda_, Ldc_);
    } else {
      MlasTestFixture<MlasSQNBitGemmTest<4, 128>>::mlas_tester->TestAsymmetricKleidiAICompInt8(
          M_, N_, K_, WithThreadpool_, WithBias_, BatchN_, Lda_, Ldc_, WithPostProcessor_);
    }
  }

  static size_t RegisterSingleTest(const char* test_name, size_t M, size_t N, size_t K,
                                   bool WithThreadpool, bool Symmetric, bool WithBias, size_t BatchN = 1,
                                   size_t Lda = 0, size_t Ldc = 0, bool WithPostProcessor = false) {
    testing::RegisterTest(
        MlasSQNBitGemmTest<4, 128>::GetTestSuiteName(),
        test_name,
        nullptr,
        test_name,
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<MlasSQNBitGemmTest<4, 128>>* {
          return new SQNBitGemmKleidiAIShortExecuteTest(
              M, N, K, WithThreadpool, Symmetric, WithBias, BatchN, Lda, Ldc, WithPostProcessor);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t tests_registered = 0;

    tests_registered += RegisterSingleTest(
        "KleidiAIAsymGemv_M1_N257_K128", 1, 257, 128,
        /*WithThreadpool=*/false, /*Symmetric=*/false, /*WithBias=*/true);
    tests_registered += RegisterSingleTest(
        "KleidiAIAsymGemm_M5_N257_K128", 5, 257, 128,
        /*WithThreadpool=*/true, /*Symmetric=*/false, /*WithBias=*/true);
    tests_registered += RegisterSingleTest(
        "KleidiAIAsymGemm_M5_N257_K128_PostProcessor", 5, 257, 128,
        /*WithThreadpool=*/true, /*Symmetric=*/false, /*WithBias=*/true, /*BatchN=*/1,
        /*Lda=*/0, /*Ldc=*/0, /*WithPostProcessor=*/true);
    tests_registered += RegisterSingleTest(
        "KleidiAIAsymGemmBatch2_M5_N257_K128", 5, 257, 128,
        /*WithThreadpool=*/true, /*Symmetric=*/false, /*WithBias=*/true, /*BatchN=*/2);
    tests_registered += RegisterSingleTest(
        "KleidiAIAsymGemv_M1_N288_K1024_NoBias", 1, 288, 1024,
        /*WithThreadpool=*/false, /*Symmetric=*/false, /*WithBias=*/false);
    tests_registered += RegisterSingleTest(
        "KleidiAISymGemv_M1_N257_K128_NoBias", 1, 257, 128,
        /*WithThreadpool=*/false, /*Symmetric=*/true, /*WithBias=*/false);
    tests_registered += RegisterSingleTest(
        "KleidiAISymGemm_M5_N257_K1024", 5, 257, 1024,
        /*WithThreadpool=*/true, /*Symmetric=*/true, /*WithBias=*/true);
    tests_registered += RegisterSingleTest(
        "KleidiAIAsymGemm_M5_N257_K128_Lda145_NoThreadpool", 5, 257, 128,
        /*WithThreadpool=*/false, /*Symmetric=*/false, /*WithBias=*/false, /*BatchN=*/1, /*Lda=*/145);
    tests_registered += RegisterSingleTest(
        "KleidiAIAsymGemm_M5_N257_K128_Lda145_Threadpool", 5, 257, 128,
        /*WithThreadpool=*/true, /*Symmetric=*/false, /*WithBias=*/false, /*BatchN=*/1, /*Lda=*/145);
    tests_registered += RegisterSingleTest(
        "KleidiAISymGemm_M5_N257_K128_Lda145_NoThreadpool", 5, 257, 128,
        /*WithThreadpool=*/false, /*Symmetric=*/true, /*WithBias=*/false, /*BatchN=*/1, /*Lda=*/145);
    tests_registered += RegisterSingleTest(
        "KleidiAISymGemm_M5_N257_K128_Lda145_Threadpool", 5, 257, 128,
        /*WithThreadpool=*/true, /*Symmetric=*/true, /*WithBias=*/false, /*BatchN=*/1, /*Lda=*/145);
    tests_registered += RegisterSingleTest(
        "KleidiAIAsymGemmBatch2_M5_N257_K128_Lda145_Threadpool", 5, 257, 128,
        /*WithThreadpool=*/true, /*Symmetric=*/false, /*WithBias=*/false, /*BatchN=*/2, /*Lda=*/145);
    tests_registered += RegisterSingleTest(
        "KleidiAISymGemm_M5_N257_K128_Ldc274_Threadpool", 5, 257, 128,
        /*WithThreadpool=*/true, /*Symmetric=*/true, /*WithBias=*/false, /*BatchN=*/1, /*Lda=*/0, /*Ldc=*/274);

    return tests_registered;
  }

 private:
  size_t M_, N_, K_, BatchN_, Lda_, Ldc_;
  bool WithThreadpool_, Symmetric_, WithBias_, WithPostProcessor_;
};

static size_t SQNBitGemmRegisterAllShortExecuteTests() {
  size_t count = 0;

  count += SQNBitGemmShortExecuteTest<4, 16>::RegisterShortExecuteTests();
  count += SQNBitGemmShortExecuteTest<4, 32>::RegisterShortExecuteTests();
  count += SQNBitGemmShortExecuteTest<4, 64>::RegisterShortExecuteTests();
  count += SQNBitGemmShortExecuteTest<4, 128>::RegisterShortExecuteTests();
  count += SQNBitGemmShortExecuteTest<4, 256>::RegisterShortExecuteTests();
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
