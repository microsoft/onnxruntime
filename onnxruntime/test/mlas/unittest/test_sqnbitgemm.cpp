/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_sqnbitgemm.h

Abstract:

    Tests for MLAS n-bit int block quantized GEMM.

--*/

#include "test_util.h"
#include "test_fp16.h"
#include "core/framework/float16.h"
#include "mlas_q4.h"
#include "mlas.h"
#include "mlas_qnbit.h"

#include <type_traits>

static constexpr const char* ComputeTypeName(MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType) {
  switch (ComputeType) {
    case CompFp32:
      return "Fp32";
    case CompInt8:
      return "Int8";
    default:
      return "unknown";
  }
}

/**
 * @brief Test class for n-bit int block quantized GEMM
 *        Note: only 2-D matmul supported for now
 */
template <typename AType, size_t BlkBitWidth, size_t BlkLen>
class MlasSQNBitGemmTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<AType> BufferA;
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
                const AType* A,
                size_t lda,
                const void* /*QuantBData*/,
                const void* PackedQuantBDataWorkspace,
                const float* QuantBScale,
                const void* QuantBZeroPoint,
                const float* Bias,
                float* C,
                size_t ldc,
                void* Workspace,
                MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
                MLAS_THREADPOOL* Threadpool) {
    MLAS_SQNBIT_GEMM_DATA_PARAMS params;
    params.A = A;
    params.lda = lda;
    params.Bias = Bias;
    params.C = C;
    params.ldc = ldc;
#ifdef MLAS_TARGET_AMD64_IX86
    if (ComputeType == CompInt8) {
      params.QuantBDataWorkspace = PackedQuantBDataWorkspace;
    }
#endif
    params.PackedQuantBData = static_cast<const std::byte*>(PackedQuantBDataWorkspace);
    params.QuantBScale = QuantBScale;
    params.QuantBZeroPoint = QuantBZeroPoint;
    params.PostProcessor = nullptr;

    if constexpr (std::is_same<AType, MLFp16>::value) {
      MlasSQNBitGemmBatch<MLAS_FP16>(M, N, K, 1, BlkBitWidth, BlkLen, ComputeType, &params, Workspace, Threadpool);
    } else {
      MlasSQNBitGemmBatch<AType>(M, N, K, 1, BlkBitWidth, BlkLen, ComputeType, &params, Workspace, Threadpool);
    }
  }

  void QuantizeA(size_t M, size_t K, const AType* A, int8_t* QuantAData, float* QuantAScale) {
    const size_t BlockCountK = (K + BlkLen - 1) / BlkLen;
    const size_t lda = K;
    for (size_t m = 0; m < M; ++m) {
      for (size_t k = 0, k_blk = 0; k < K; k += BlkLen, ++k_blk) {
        const size_t local_blk_len = std::min(K - k, BlkLen);
        AType blk_a[BlkLen]{};
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

  template <typename AType>
  void CallReferenceGemm_CompInt8(size_t M,
                                  size_t N,
                                  size_t K,
                                  const AType* A,
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

  template <typename AType>
  void CallReferenceGemm_CompFp32(size_t M,
                                  size_t N,
                                  size_t K,
                                  const AType* A,
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
        const AType* a = A + m * K;
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
            MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
            bool WithThreadpool, bool Symmetric, bool WithBias) {
    MLAS_THREADPOOL* Threadpool = WithThreadpool ? GetMlasThreadPool() : nullptr;

    const AType* A = BufferA.GetBuffer(K * M);

    const float* B = BufferB.GetBuffer(N * K);

    const float* Bias = nullptr;
    if (WithBias) {
      Bias = BufferBias.GetBuffer(N);
    }

#if 0
    auto print_matrix1 = [](size_t nrows, size_t ncols, const MLFp16* data) {
      for (size_t row = 0; row < nrows; ++row) {
        for (size_t col = 0; col < ncols; ++col) {
          std::cout << data[row * ncols + col] << ", ";
        }
        std::cout << "\n";
      }
    };

    auto print_matrix2 = [](size_t nrows, size_t ncols, const float* data) {
      for (size_t row = 0; row < nrows; ++row) {
        for (size_t col = 0; col < ncols; ++col) {
          std::cout << data[row * ncols + col] << ", ";
        }
        std::cout << "\n";
      }
    };

    auto print_matrix_col = [](size_t nrows, size_t ncols, size_t col, const MLFp16* data) {
      for (size_t row = 0; row < nrows; ++row) {
        std::cout << data[row * ncols + col] << ", ";
      }
      std::cout << "\n";
    };

    std::cout << "A:\n";
    print_matrix1(M, K, A);
    std::cout << "B:\n";
    print_matrix2(K, N, B);
#endif

    float* C = BufferC.GetBuffer(N * M, true);
    float* CReference = BufferCReference.GetBuffer(N * M, true);

    // quantize B
    uint8_t* QuantBData = nullptr;
    float* QuantBScale = nullptr;
    uint8_t* QuantBZeroPoint = nullptr;
    {
      size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
      MlasBlockwiseQuantizedBufferSizes(BlkBitWidth, BlkLen, /* columnwise */ true,
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
    if (const auto WorkspaceSize = MlasSQNBitGemmBatchWorkspaceSize(M, N, K, 1, BlkBitWidth, BlkLen, ComputeType);
        WorkspaceSize > 0) {
      Workspace = BufferWorkspace.GetBuffer(WorkspaceSize);
    }

    void* PackedQuantBDataWorkspace = nullptr;
    if (const auto PackedQuantBDataSize = MlasSQNBitGemmPackQuantBDataSize(N, K, BlkBitWidth, BlkLen, ComputeType);
        PackedQuantBDataSize > 0) {
      PackedQuantBDataWorkspace = BufferPackedQuantBData.GetBuffer(PackedQuantBDataSize);
      bool has_zp_input = QuantBZeroPoint != nullptr;
      MlasSQNBitGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, ComputeType, QuantBData, PackedQuantBDataWorkspace,
                                   QuantBScale, has_zp_input, QuantBZeroPoint,
                                   GetMlasThreadPool());
    }

    CallGemm(M, N, K,
             A, /* lda */ K,
             QuantBData, PackedQuantBDataWorkspace, QuantBScale, QuantBZeroPoint,
             Bias,
             C, /* ldc */ N,
             Workspace,
             ComputeType,
             Threadpool);

    if (ComputeType == CompFp32) {
      CallReferenceGemm_CompFp32(M, N, K, A, QuantBData, QuantBScale, QuantBZeroPoint, Bias, CReference);
    } else if (ComputeType == CompInt8) {
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

 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("SQNBitGemm") +
                                    "BlkBitWidth" + std::to_string(BlkBitWidth) +
                                    "BlkLen" + std::to_string(BlkLen) +
                                    "AType" + std::string(std::is_same<AType, float>::value ? "Fp32" : "Fp16");
    return suite_name.c_str();
  }
};

//
// Short Execute() test helper to register each test separately by all parameters.
//
template <typename AType, size_t BlkBitWidth, size_t BlkLen>
class SQNBitGemmShortExecuteTest : public MlasTestFixture<MlasSQNBitGemmTest<AType, BlkBitWidth, BlkLen>> {
 public:
  explicit SQNBitGemmShortExecuteTest(size_t M, size_t N, size_t K,
                                      MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
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
    MlasTestFixture<MlasSQNBitGemmTest<AType, BlkBitWidth, BlkLen>>::mlas_tester->Test(
        M_, N_, K_, ComputeType_, WithThreadpool_, Symmetric_, WithBias_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K,
                                   MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
                                   bool WithThreadpool, bool Symmetric, bool WithBias) {
    size_t tests_registered = 0;

    if (MlasIsSQNBitGemmAvailable(BlkBitWidth, BlkLen, ComputeType)) {
      std::stringstream ss;
      ss << (WithThreadpool ? "SingleThread" : "Threaded")
         << "/isSymmetric" << Symmetric
         << "/M" << M << "xN" << N << "xK" << K
         << "/hasBias" << WithBias
         << "/AType" << (std::is_same<AType, float>::value ? "Fp32" : "Fp16")
         << "/computeType" << ComputeTypeName(ComputeType);
      auto test_name = ss.str();

      testing::RegisterTest(
          MlasSQNBitGemmTest<AType, BlkBitWidth, BlkLen>::GetTestSuiteName(),
          test_name.c_str(),
          nullptr,
          test_name.c_str(),
          __FILE__,
          __LINE__,
          // Important to use the fixture type as the return type here.
          [=]() -> MlasTestFixture<MlasSQNBitGemmTest<AType, BlkBitWidth, BlkLen>>* {
            return new SQNBitGemmShortExecuteTest(
                M, N, K, ComputeType, WithThreadpool, Symmetric, WithBias);
          });

      tests_registered += 1;
    }

    return tests_registered;
  }

  static size_t RegisterShortExecuteTests() {
    size_t tests_registered = 0;

    for (MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType : {CompFp32, CompInt8}) {
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
  MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType_;
  bool WithThreadpool_, Symmetric_, WithBias_;
};

template <typename AType>
static size_t SQNBitGemmRegisterAllShortExecuteTests() {
  size_t count = 0;

  count += SQNBitGemmShortExecuteTest<AType, 4, 16>::RegisterShortExecuteTests();
  count += SQNBitGemmShortExecuteTest<AType, 4, 32>::RegisterShortExecuteTests();
  count += SQNBitGemmShortExecuteTest<AType, 4, 64>::RegisterShortExecuteTests();
  count += SQNBitGemmShortExecuteTest<AType, 4, 128>::RegisterShortExecuteTests();
  count += SQNBitGemmShortExecuteTest<AType, 4, 256>::RegisterShortExecuteTests();

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        // using MLAS_FP16 = onnxruntime::MLFloat16;
        return SQNBitGemmRegisterAllShortExecuteTests<float>() + SQNBitGemmRegisterAllShortExecuteTests<MLFp16>();
        //return SQNBitGemmRegisterAllShortExecuteTests<onnxruntime::MLFloat16>();
      }
      return 0;
    });
