/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_hqnbitgemm_neon.cpp

Abstract:

    Tests for MLAS n-bit int block quantized GEMM on ARM CPU with input A type T1 fp16.

--*/

#include <vector>
#include <random>

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/lib/qnbitgemm.h"
#include "mlas_qnbit.h"

template <size_t K, size_t N, size_t BlkLen, size_t SubBlkLen>
class MlasSQ8BitPrepackTest : public MlasTestBase {
 private:
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int> distrib_u8_;
  std::uniform_real_distribution<float> distrib_f32_;
  MatrixGuardBuffer<uint8_t> inputB_, inputZp_, refB_, packedBuffer_;
  MatrixGuardBuffer<float> inputScale_, refScale_;
  MatrixGuardBuffer<float> inputBlkSum_, refBlkSum_;

  void PrepackB(const uint8_t* src, uint8_t* dst) {
    constexpr size_t ldb = (K + BlkLen - 1) & (~(BlkLen - 1));
    size_t n = 0;
    for (; n + 4 <= N; n += 4) {
      size_t k = 0;
      for (; k + SubBlkLen <= ldb; k += SubBlkLen) {
        for (size_t i = 0; i < 4; ++i) {
          std::copy(src + (n + i) * ldb + k, src + (n + i) * ldb + k + SubBlkLen, dst + n * ldb + 4 * k + i * SubBlkLen);
        }
      }

      for (size_t kk = 0; kk + k + BlkLen <= ldb; kk += BlkLen) {
        for (size_t i = 0; i < 4; ++i) {
          std::copy(src + (n + i) * ldb + k + kk, src + (n + i) * ldb + k + kk + BlkLen, dst + n * ldb + 4 * k + 4 * kk + i * BlkLen);
        }
      }
    }

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif
    for (; n < N; ++n) {
      std::copy(src + n * ldb, src + n * ldb + ldb, dst + n * ldb);
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
  }

  void PrepackBlkSumAndScale(const float* scale, const uint8_t* zp, float* packedScale, float* blkSum) {
    constexpr size_t BlkCount = (K + BlkLen - 1) / BlkLen;
    constexpr size_t BlkPerSubBlk = SubBlkLen > BlkLen ? SubBlkLen / BlkLen : 1;

    size_t n = 0;
    for (; n + 4 <= N; n += 4) {
      size_t k = 0;
      for (; k + BlkPerSubBlk <= BlkCount; k += BlkPerSubBlk) {
        for (size_t i = 0; i < 4; ++i) {
          for (size_t j = 0; j < BlkPerSubBlk; ++j) {
            auto srcOffset = (n + i) * BlkCount + k + j;
            auto scaleDstOffset = n * BlkCount + 4 * k + i * BlkPerSubBlk + j;
            auto sumDstOffset = (((n + i) / 16) * BlkCount + k + j) * 16 + (n + i) % 16;

            auto vSum = -scale[srcOffset] * (zp ? static_cast<float>(zp[srcOffset]) : 128.f);

            packedScale[scaleDstOffset] = scale[srcOffset];
            blkSum[sumDstOffset] = vSum;
          }
        }
      }
      for (size_t kk = 0; k + kk < BlkCount; ++kk) {
        for (size_t i = 0; i < 4; ++i) {
          auto srcOffset = (n + i) * BlkCount + k + kk;
          auto scaleDstOffset = n * BlkCount + 4 * k + 4 * kk + i;
          auto sumDstOffset = (((n + i) / 16) * BlkCount + k + kk) * 16 + (n + i) % 16;

          auto vSum = -scale[srcOffset] * (zp ? static_cast<float>(zp[srcOffset]) : 128.f);

          packedScale[scaleDstOffset] = scale[srcOffset];
          blkSum[sumDstOffset] = vSum;
        }
      }
    }

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif
    for (; n < N; ++n) {
      for (size_t k = 0; k < BlkCount; ++k) {
        auto srcOffset = n * BlkCount + k;
        auto scaleDstOffset = n * BlkCount + k;
        auto sumDstOffset = (((n) / 16) * BlkCount + k) * 16 + (n) % 16;

        auto vSum = -scale[srcOffset] * (zp ? static_cast<float>(zp[srcOffset]) : 128.f);

        packedScale[scaleDstOffset] = scale[srcOffset];
        blkSum[sumDstOffset] = vSum;
      }
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
  }

  void CheckB(const uint8_t* packedB, const uint8_t* refB) {
    size_t ldb = (K + BlkLen - 1) & (~(BlkLen - 1));
    size_t n = 0, N4 = N & (~3), ldbSub = ldb & (~(SubBlkLen - 1));
    for (; n < N4; ++n) {
      size_t k = 0;
      for (; k < ldbSub && k < K; ++k) {
        size_t idx = (n & (~3)) * ldb + (k & (~(SubBlkLen - 1))) * 4 + (n & 3) * SubBlkLen + (k & (SubBlkLen - 1));
        ASSERT_EQ(packedB[idx], refB[idx])
            << " n " << n << " k " << k;
      }
      for (; k < K; ++k) {
        size_t idx = (n & (~3)) * ldb + (k & (~(BlkLen - 1))) * 4 + (n & 3) * BlkLen + (k & (BlkLen - 1));
        ASSERT_EQ(packedB[idx], refB[idx])
            << " n " << n << " k " << k;
      }
    }

    for (; n < N; ++n) {
      for (size_t k = 0; k < K; ++k) {
        ASSERT_EQ(packedB[n * ldb + k], refB[n * ldb + k])
            << " n " << n << " k " << k;
      }
    }
  }

  void CheckScale(const float* packedScale, const float* refScale) {
    size_t BlkCount = (K + BlkLen - 1) / BlkLen;
    size_t BlkPerSubBlk = SubBlkLen > BlkLen ? SubBlkLen / BlkLen : 1;
    size_t n = 0, N4 = N & (~3), BlkCountSub = BlkCount & (~(BlkPerSubBlk - 1));

    for (; n < N4; ++n) {
      size_t k = 0;
      for (; k < BlkCountSub; ++k) {
        size_t idx = (n & (~3)) * BlkCount + (k & (~(BlkPerSubBlk - 1))) * 4 + (n & 3) * BlkPerSubBlk + (k & (BlkPerSubBlk - 1));
        ASSERT_EQ(packedScale[idx], refScale[idx])
            << " n " << n << " k " << k;
      }
      for (; k < BlkCount; ++k) {
        size_t idx = (n & (~3)) * BlkCount + k * 4 + (n & 3);
        ASSERT_EQ(packedScale[idx], refScale[idx])
            << " n " << n << " k " << k;
      }
    }

    for (; n < N; ++n) {
      for (size_t k = 0; k < BlkCount; ++k) {
        ASSERT_EQ(packedScale[n * BlkCount + k], refScale[n * BlkCount + k])
            << " n " << n << " k " << k;
      }
    }
  }

  void CheckBlkSum(const float* packedBlkSum, const float* refBlkSum) {
    size_t BlkCount = (K + BlkLen - 1) / BlkLen;

    for (size_t n = 0; n < N; ++n) {
      for (size_t k = 0; k < BlkCount; ++k) {
        size_t idx = (((n) / 16) * BlkCount + k) * 16 + (n) % 16;
        ASSERT_EQ(packedBlkSum[idx], refBlkSum[idx])
            << " n " << n << " k " << k;
      }
    }
  }

  template <bool hasZp>
  void TestPrepack() {
    if (!MlasIsQNBitGemmAvailable(8, BlkLen, SQNBIT_CompInt8)) return;

    constexpr size_t Bits = 8;
    constexpr size_t BlkCount = (K + BlkLen - 1) / BlkLen;
    constexpr size_t Ldb = (((K + BlkLen - 1) & (~(BlkLen - 1))) * Bits + 7) / 8;
    constexpr size_t PackBCount = N * Ldb;
    constexpr size_t ScaleCount = BlkCount * N;
    const size_t BufferSize = MlasQNBitGemmPackQuantBDataSize(N, K, Bits, BlkLen, hasZp, SQNBIT_CompInt8);

    const auto* inputB = inputB_.GetFilledBuffer(PackBCount, [this](uint8_t* p, size_t t) {
      for (size_t i = 0; i < t; i++) {
        p[i] = static_cast<uint8_t>(this->distrib_u8_(this->gen_));
      }
    });

    const auto* inputScale = inputScale_.GetFilledBuffer(ScaleCount, [this](float* p, size_t t) {
      for (size_t i = 0; i < t; i++) {
        p[i] = this->distrib_f32_(this->gen_);
      }
    });

    const auto* inputZp = hasZp ? inputZp_.GetFilledBuffer(ScaleCount, [this](uint8_t* p, size_t t) {
      for (size_t i = 0; i < t; i++) {
        p[i] = static_cast<uint8_t>(this->distrib_u8_(this->gen_));
      }
    })
                                : nullptr;

    auto* packedBuffer = packedBuffer_.GetBuffer(BufferSize, true);
    auto* refB = refB_.GetBuffer(PackBCount, true);
    auto* refScale = refScale_.GetBuffer(ScaleCount, true);
    auto* refBlkSum = refBlkSum_.GetBuffer(((N + 15) & (~15)) * BlkCount, true);

    MlasQNBitGemmPackQuantBData(
        N, K, Bits, BlkLen, MLAS_QNBIT_GEMM_COMPUTE_TYPE::SQNBIT_CompInt8, inputB, packedBuffer,
        inputScale, hasZp, nullptr, nullptr);
    MlasQNBitGemmPackQuantBData(
        N, K, Bits, BlkLen, MLAS_QNBIT_GEMM_COMPUTE_TYPE::SQNBIT_CompInt8, nullptr, packedBuffer,
        inputScale, hasZp, nullptr, nullptr);
    MlasQNBitGemmPackQuantBData(
        N, K, Bits, BlkLen, MLAS_QNBIT_GEMM_COMPUTE_TYPE::SQNBIT_CompInt8, nullptr, packedBuffer,
        nullptr, hasZp, inputZp, nullptr);

    PackedQuantBDataStruct<float, 8> packedQuantB(packedBuffer, N, BlkCount, BlkLen);

    PrepackB(inputB, refB);
    PrepackBlkSumAndScale(inputScale, inputZp, refScale, refBlkSum);

    CheckB(refB, reinterpret_cast<const uint8_t*>(packedQuantB.PackedQuantBData));
    CheckScale(refScale, packedQuantB.PackedQuantBScale);
    CheckBlkSum(refBlkSum, packedQuantB.QuantBBlkSum);
  }

 public:
  MlasSQ8BitPrepackTest()
      : seed_(19287), gen_(seed_), distrib_u8_(0, 255), distrib_f32_(-10.f, 10.f) {
  }

  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("SQ8BitPrepack_") +
                                    "K" + std::to_string(K) +
                                    "N" + std::to_string(N) +
                                    "BlkLen" + std::to_string(BlkLen) +
                                    "SubBlkLen" + std::to_string(SubBlkLen);
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    TestPrepack<false>();
    TestPrepack<true>();
  }
};

template <size_t M, size_t K, size_t N, size_t BlkLen>
class MlasSQ8BitGemmKernelTest : public MlasTestBase {
 private:
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distrib_f32_;
  std::normal_distribution<float> norm_distrib_f32_;
  MatrixGuardBuffer<uint8_t> packedBuffer_, workspace_, packedB_, Zp_;
  MatrixGuardBuffer<float> A_, B_, C_, ref_, bias_, scale_;

  bool FloatEqual(float v0, float v1, float rtol, float atol) {
    return std::abs(v0 - v1) <= std::abs(v1 * rtol) + atol;
  }

  void MatMul(const float* A, size_t lda, const float* B, const float* bias, float* C, size_t ldc) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        float accu = bias ? bias[n] : 0.0f;
        for (size_t k = 0; k < K; ++k) {
          float a = A[m * lda + k];
          float b = B[n * K + k];
          accu += a * b;
        }
        C[m * ldc + n] = accu;
      }
    }
  }

  void Check(const float* target, const float* ref, size_t ldc, float rtol, float atol) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        size_t i = m * ldc + n;
        ASSERT_TRUE(FloatEqual(target[i], ref[i], rtol, atol))
            << " v0 " << target[i] << " v1 " << ref[i]
            << " m " << m << " n " << n;
      }
    }
  }

  template <bool HasBias, bool HasZp>
  void TestSQ8BitGemmKernel() {
    if (!MlasIsQNBitGemmAvailable(8, BlkLen, SQNBIT_CompInt8)) return;

    constexpr size_t BlkCount = (K + BlkLen - 1) / BlkLen;
    constexpr size_t ldb = BlkCount * BlkLen;
    constexpr size_t lda = ldb;
    constexpr size_t ldc = (N + 15) & (~15);
    const auto* A = A_.GetFilledBuffer(M * lda, [this](float* p, size_t t) {
      for (size_t i = 0; i < t; i++) {
        p[i] = this->norm_distrib_f32_(this->gen_);
      }
    });

    auto* B = B_.GetFilledBuffer(K * N, [this](float* p, size_t t) {
      for (size_t i = 0; i < t; i++) {
        p[i] = this->norm_distrib_f32_(this->gen_);
      }
    });

    int q_rows, q_cols;
    MlasBlockwiseQuantizedShape<float, 8>((int)BlkLen, true, (int)K, (int)N, q_rows, q_cols);

    size_t q_data_size_in_bytes, q_scale_size, q_zp_size_in_bytes;
    MlasBlockwiseQuantizedBufferSizes<8>((int)(BlkLen), true, (int)K, (int)N,
                                         q_data_size_in_bytes, q_scale_size, &q_zp_size_in_bytes);

    auto* inputB = packedB_.GetBuffer(q_data_size_in_bytes, true);
    auto* inputScale = scale_.GetBuffer(q_scale_size, true);
    auto* inputZp = HasZp ? Zp_.GetBuffer(q_zp_size_in_bytes, true) : nullptr;

    MlasQuantizeBlockwise<float, 8>(
        inputB,
        inputScale,
        inputZp,
        B,
        BlkLen,
        true,
        K,
        N,
        N,
        nullptr);

    MlasDequantizeBlockwise<float, 8>(
        B,
        inputB,
        inputScale,
        inputZp,
        BlkLen,
        true,
        K,
        N,
        nullptr);

    size_t bufferSize = MlasQNBitGemmPackQuantBDataSize(N, K, 8, BlkLen, HasZp, SQNBIT_CompInt8);
    auto* packedBuffer = packedBuffer_.GetBuffer(bufferSize, true);

    MlasQNBitGemmPackQuantBData(
        N, K, 8, BlkLen, MLAS_QNBIT_GEMM_COMPUTE_TYPE::SQNBIT_CompInt8, inputB, packedBuffer,
        inputScale, HasZp, nullptr, nullptr);
    MlasQNBitGemmPackQuantBData(
        N, K, 8, BlkLen, MLAS_QNBIT_GEMM_COMPUTE_TYPE::SQNBIT_CompInt8, nullptr, packedBuffer,
        inputScale, HasZp, nullptr, nullptr);
    MlasQNBitGemmPackQuantBData(
        N, K, 8, BlkLen, MLAS_QNBIT_GEMM_COMPUTE_TYPE::SQNBIT_CompInt8, nullptr, packedBuffer,
        nullptr, HasZp, inputZp, nullptr);

    PackedQuantBDataStruct<float, 8> packedQuantB(packedBuffer, N, BlkCount, BlkLen);

    auto* C = C_.GetBuffer(M * ldc, true);
    auto* ref = ref_.GetBuffer(M * ldc, true);

    auto* bias = HasBias ? bias_.GetFilledBuffer(N, [this](float* p, size_t t) {
      for (size_t i = 0; i < t; i++) {
        p[i] = this->distrib_f32_(this->gen_);
      }
    })
                         : nullptr;

    const size_t workspace_size = MlasQNBitGemmBatchWorkspaceSize(M, N, K, 1, 8, BlkLen, HasZp, SQNBIT_CompInt8);
    auto* workspace = workspace_.GetBuffer(workspace_size, true);

    MLAS_QNBIT_GEMM_DATA_PARAMS<float> data;
    data.A = A;
    data.lda = lda;
    data.QuantBDataWorkspace = packedBuffer;
    data.PackedQuantBData = packedQuantB.PackedQuantBData;
    data.QuantBScale = inputScale;
    data.QuantBZeroPoint = inputZp;
    data.Bias = bias;
    data.C = C;
    data.ldc = ldc;

    MlasQNBitGemmBatch(M, N, K, 1, 8, BlkLen, SQNBIT_CompInt8, &data, workspace, nullptr);

    MatMul(A, lda, B, bias, ref, ldc);
    Check(C, ref, ldc, 0.05f, 0.02f);
  }

 public:
  MlasSQ8BitGemmKernelTest()
      : seed_(1234), gen_(seed_), distrib_f32_(1.f, 5.f), norm_distrib_f32_(0.f, 0.25f) {
  }

  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("SQ8BitGemmKernel_") +
                                    "M" + std::to_string(M) +
                                    "N" + std::to_string(N) +
                                    "K" + std::to_string(K) +
                                    "BlkLen" + std::to_string(BlkLen);
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    TestSQ8BitGemmKernel<false, false>();
    TestSQ8BitGemmKernel<false, true>();
    TestSQ8BitGemmKernel<true, false>();
    TestSQ8BitGemmKernel<true, true>();
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  auto& platform = GetMlasPlatform();

  if (is_short_execute) {
    if (platform.Avx512Supported_) {
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 16, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 32, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 64, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 128, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 256, 128>>::RegisterShortExecute();

      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<16, 4, 16, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<32, 4, 16, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<64, 4, 16, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<128, 4, 16, 128>>::RegisterShortExecute();

      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<15, 5, 16, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<15, 5, 32, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<15, 5, 64, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<15, 5, 128, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<15, 5, 256, 128>>::RegisterShortExecute();

      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<17, 8, 16, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<17, 8, 32, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<17, 8, 64, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<17, 8, 128, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<17, 8, 256, 128>>::RegisterShortExecute();

      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<256, 16, 16, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<257, 17, 32, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<255, 15, 64, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<256, 17, 128, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<257, 16, 256, 128>>::RegisterShortExecute();
    } else {
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 16, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 32, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 64, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 128, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 256, 64>>::RegisterShortExecute();

      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<16, 4, 16, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<32, 4, 16, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<64, 4, 16, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<128, 4, 16, 64>>::RegisterShortExecute();

      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<15, 5, 16, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<15, 5, 32, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<15, 5, 64, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<15, 5, 128, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<15, 5, 256, 64>>::RegisterShortExecute();

      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<17, 8, 16, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<17, 8, 32, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<17, 8, 64, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<17, 8, 128, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<17, 8, 256, 64>>::RegisterShortExecute();

      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<159, 16, 16, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<160, 17, 32, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<161, 15, 64, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<160, 17, 128, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<159, 16, 256, 64>>::RegisterShortExecute();
    }

    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 1, 1, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 16, 4, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 32, 4, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 64, 4, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 128, 4, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 16, 4, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 32, 4, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 64, 4, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 128, 4, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 16, 3, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 32, 3, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 64, 3, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 128, 3, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 16, 3, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 32, 3, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 64, 3, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 128, 3, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 128, 3, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<7, 128, 4, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<8, 497, 5, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 127, 4, 16>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 129, 4, 16>>::RegisterShortExecute();

    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 1, 1, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 16, 4, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 32, 4, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 64, 4, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 128, 4, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 256, 4, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 16, 4, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 32, 4, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 64, 4, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 128, 4, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 256, 4, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 16, 3, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 32, 3, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 64, 3, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 128, 3, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 256, 3, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 16, 3, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 32, 3, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 64, 3, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 128, 3, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 256, 3, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<8, 33, 5, 32>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<8, 513, 9, 32>>::RegisterShortExecute();

    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 1, 1, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 16, 4, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 32, 4, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 64, 4, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 128, 4, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 256, 4, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 16, 4, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 32, 4, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 64, 4, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 128, 4, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 256, 4, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 16, 3, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 32, 3, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 64, 3, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 128, 3, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 256, 3, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 16, 3, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 32, 3, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 64, 3, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 128, 3, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 256, 3, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<7, 96, 5, 64>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<8, 497, 9, 64>>::RegisterShortExecute();

    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 1, 1, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 16, 4, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 32, 4, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 64, 4, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 128, 4, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 256, 4, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 16, 4, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 32, 4, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 64, 4, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 128, 4, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 256, 4, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 16, 3, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 32, 3, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 64, 3, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 128, 3, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 256, 3, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 16, 3, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 32, 3, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 64, 3, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 128, 3, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 256, 3, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<6, 255, 7, 128>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<5, 257, 9, 128>>::RegisterShortExecute();

    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 1, 1, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 16, 4, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 32, 4, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 64, 4, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 128, 4, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 256, 4, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 16, 4, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 32, 4, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 64, 4, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 128, 4, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 256, 4, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 16, 3, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 32, 3, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 64, 3, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 128, 3, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<1, 256, 3, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 16, 3, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 32, 3, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 64, 3, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 128, 3, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<2, 256, 3, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<7, 255, 7, 256>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSQ8BitGemmKernelTest<6, 257, 7, 256>>::RegisterShortExecute();
  }
  return count;
});
