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
#include "core/mlas/lib/qnbitgemm.h"
#include "mlas_qnbit.h"

template <size_t K, size_t N, size_t BlkLen, size_t SubBlkLen>
class MlasSQ8BitPrepackTest : public MlasTestBase {
 private:
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<uint8_t> distrib_u8_;
  std::uniform_real_distribution<float> distrib_f32_;
  MatrixGuardBuffer<uint8_t> inputB_, inputZp_, refB_, packedBuffer_;
  MatrixGuardBuffer<float> inputScale_, refScale_;
  MatrixGuardBuffer<float> inputBlkSum_, refBlkSum_;

  void PrepackB(const uint8_t* src, uint8_t* dst) {
    size_t ldb = (K + BlkLen - 1) & (~(BlkLen - 1));
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

    for (; n < N; ++n) {
      std::copy(src + n * ldb, src + n * ldb + ldb, dst + n * ldb);
    }
  }

  void PrepackBlkSumAndScale(const float* scale, const uint8_t* zp, float* packedScale, float* blkSum) {
    size_t BlkCount = (K + BlkLen - 1) / BlkLen;
    size_t BlkPerSubBlk = SubBlkLen > BlkLen ? SubBlkLen / BlkLen : 1;

    size_t n = 0;
    for (; n + 4 <= N; n += 4) {
      size_t k = 0;
      for (; k + BlkPerSubBlk <= BlkCount; k += BlkPerSubBlk) {
        for (size_t i = 0; i < 4; ++i) {
          for (size_t j = 0; j < BlkPerSubBlk; ++j) {
            auto srcOffset = (n + i) * BlkCount + k + j;
            auto scaleDstOffset = n * BlkCount + 4 * k * BlkPerSubBlk + i * BlkPerSubBlk + j;
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
        ASSERT_EQ(packedB[n * ldb + k], ref[n * ldb + k])
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
    constexpr size_t ScaleCount = (K + BlkLen - 1) / BlkLen * N;
    constexpr size_t BufferSize = MlasQNBitGemmPackQuantBDataSize(N, K, Bits, BlkLen, hasZp, SQNBIT_CompInt8);

    const auto* inputB = inputB_.GetFilledBuffer(PackBCount, [this](uint8_t* p, size_t t) {
      for (size_t i = 0; i < PackBCount; i++) {
        p[i] = static_cast<uint8_t>(this->distrib_u8_(this->gen_));
      }
    });

    const auto* inputScale = inputScale_.GetFilledBuffer(ScaleCount, [this](float* p, size_t t) {
      for (size_t i = 0; i < ScaleCount; i++) {
        p[i] = this->distrib_f32_(this->gen_);
      }
    });

    const auto* inputZp = hasZp ? inputZp_.GetFilledBuffer(ScaleCount, [this](uint8_t* p, size_t t) {
      for (size_t i = 0; i < PackBCount; i++) {
        p[i] = static_cast<uint8_t>(this->distrib_u8_(this->gen_));
      }
    }) : nullptr;

    auto* packedBuffer = packedBuffer_.GetBuffer(BufferSize, true);
    auto* refB = refB_.GetBuffer(PackBCount, true);
    auto* refScale = refScale_.GetBuffer(ScaleCount, true);
    auto* refBlkSum = refBlkSum_.GetBuffer(ScaleCount, true);

    MlasQNBitGemmPackQuantBData(
        N, K, Bits, BlkLen, MLAS_QNBIT_GEMM_COMPUTE_TYPE::SQNBIT_CompInt8, inputB, packedBuffer,
        inputScale, hasZp, nullptr, nullptr);
    MlasQNBitGemmPackQuantBData(
        N, K, Bits, BlkLen, MLAS_QNBIT_GEMM_COMPUTE_TYPE::SQNBIT_CompInt8, nullptr, packedBuffer,
        inputScale, hasZp, nullptr, nullptr);
    MlasQNBitGemmPackQuantBData(
        N, K, Bits, BlkLen, MLAS_QNBIT_GEMM_COMPUTE_TYPE::SQNBIT_CompInt8, nullptr, nullptr,
        inputScale, hasZp, inputZp, nullptr);

    PackedQuantBDataStruct<float, 8> packedQuantB(packedBuffer, N, BlkCount, BlkLen);

    PrepackB(inputB, refB);
    PrepackBlkSumAndScale(inputScale, inputZp, refScale, refBlkSum);

    CheckB(refB, packedQuantB.PackedQuantBData);
    CheckScale(refScale, packedQuantB.PackedQuantBScale);
    CheckBlkSum(refBlkSum, packedQuantB.QuantBBlkSum);
  }

 public:
  MlasSQ8BitPrepackTest()
      : seed_(19287), gen_(seed_), distrib_u8_(0, 255), distrib_f32_(-10.f, 10.f) {
  }

  static const char* GetTestSuiteName() {
    return "SQ8BitPrepack_K*N*BlkLen*SubBlkLen_" + std::to_string(K) + "x" + std::to_string(N) + "x" +
           std::to_string(BlkLen) + "x" + std::to_string(SubBlkLen);
  }

  void ExecuteShort(void) override {
    TestPrepack<false>();
    TestPrepack<true>();
  }
};

template <size_t M, size_t N, size_t K, size_t BlkLen>
class MlasSQ8BitGemmKernelTest : public MlasTestBase {
 private:
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distrib_f32_;
  std::uniform_int_distribution<uint8_t> distrib_u8_, workspace_;
  MatrixGuardBuffer<uint8_t> packedBuffer_;
  MatrixGuardBuffer<float> A_, C_, ref_, bias_;

  void InitializeBuffer(uint8_t* packedB, float* packedScale, float* blkSum) {
    size_t ldb = (K + BlkLen - 1) & (~(BlkLen - 1));
    size_t BlkCount = (K + BlkLen - 1) / BlkLen;

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < ldb; ++j) {
        packedB[i * ldb + j] = this->distrib_u8_(this->gen_);
      }
    }

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < BlkCount; ++j) {
        packedScale[i * BlkCount + j] = this->distrib_f32_(this->gen_);
      }
    }

    size_t ldSum = (N + 15) & (~15);
    for (size_t i = 0; i < ldSum; ++i) {
      for (size_t j = 0; j < BlkCount; ++j) {
        blkSum[i * BlkCount + j] = this->distrib_f32_(this->gen_);
      }
    }
  }

  bool FloatEqual(MLAS_FP16 v0, MLAS_FP16 v1, float rtol, float atol) {
    float f0 = v0.ToFloat(), f1 = v1.ToFloat();
    return std::abs(f0 - f1) <= std::abs(f1 * rtol) + atol;
  }

  template <size_t ldb, size_t N, size_t K>
  float GetBVal(const MLAS_FP16* B, size_t n, size_t k) {
    size_t i;
    if ((N & (~7)) > n) {
      size_t full8 = n & (~7);
      i = full8 * ldb + 8 * k + (n - full8);
    } else {
      i = n * ldb + k;
    }
    return B[i].ToFloat();
  }

  template <size_t M, size_t N, size_t K, size_t ldb, bool UseBias>
  void MatMul(const MLAS_FP16* A, const MLAS_FP16* B, const MLAS_FP16* bias, MLAS_FP16* C) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        float accu = UseBias ? bias[n] : 0.0f;
        for (size_t k = 0; k < K; ++k) {
          float a = A[m * K + k].ToFloat();
          float b = GetBVal<ldb, N, K>(B, n, k);
          accu = accu + a * b;
        }
        C[m * N + n] = MLAS_FP16(accu);
      }
    }
  }

  template <size_t Ldc, size_t M, size_t N>
  MLAS_FORCEINLINE void Check(const MLAS_FP16* target, const MLAS_FP16* ref) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        size_t i = m * Ldc + n;
        ASSERT_TRUE(FloatEqual(target[i], ref[i], 0.02f, 0.055f))
            << " seed " << seed_
            << " v0 " << target[i] << " v1 " << ref[i]
            << " m " << m << " n " << n;
      }
    }
  }

  template <bool HasBias, bool hasZp>
  void TestSQ8BitGemmKernel() {
    constexpr size_t BlkCount = (K + BlkLen - 1) / BlkLen;
    constexpr size_t ldb = BlkCount * BlkLen;
    constexpr size_t lda = ldb;
    constexpr size_t ldc = (N + 15) & (~15);
    size_t bufferSize = MlasQNBitGemmPackQuantBDataSize(N, K, 8, BlkLen, hasZp, SQNBIT_CompInt8);

    const auto* A = A_.GetFilledBuffer(M * lda, [this, lda](float* p, size_t t) {
      for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
          p[i * lda + j] = this->distrib_f32_(this->gen_);
        }
      }
    });

    auto* packedBuffer = packedBuffer_.GetBuffer(bufferSize, true);
    PackedQuantBDataStruct<float, 8> packedQuantB(packedBuffer, N, BlkCount, BlkLen);
    InitializeBuffer((uint8_t*)packedQuantB.PackedQuantBData, packedQuantB.PackedQuantBScale, packedQuantB.QuantBBlkSum);

    auto* C = C_.GetBuffer(M * ldc, true);
    auto* ref = ref_.GetBuffer(M * ldc, true);

    auto* bias = bias_.GetFilledBuffer(N, [this](float* p, size_t t) {
      for (size_t i = 0; i < N; i++) {
        p[i] = this->distrib_f32_(this->gen_);
      }
    });

    MlasQNBitGemmBatch(M, N, K, batch_count, nbits_, block_size_, compute_type_, data.data(), workspace.get(), thread_pool);

    MatMul<M, N, K, ldb, UseBias>(A, B, bias, ref);
    Check<N, M, N>(C, ref);
  }

 public:
  MlasSQ8BitGemmKernelTest()
      : seed_(19287), gen_(seed_), distrib_f32_(-1.25f, 1.25f), distrib_u8_(0, 255) {
  }

  static const char* GetTestSuiteName() {
    return "SQ8BitGemmKernel_M*N*K*BlkLen_" + std::to_string(M) + "x" + std::to_string(N) + "x" +
           std::to_string(K) + "x" + std::to_string(BlkLen);
  }

  template <size_t M>
  void ExecuteShort_T(void) {
    TestHQ4BitGemmKernel<M, 1, 1, 16, false>();
    TestHQ4BitGemmKernel<M, 1, 1, 16, true>();
    TestHQ4BitGemmKernel<M, 1, 15, 16, false>();
    TestHQ4BitGemmKernel<M, 1, 15, 16, true>();
    TestHQ4BitGemmKernel<M, 1, 31, 16, false>();
    TestHQ4BitGemmKernel<M, 1, 31, 16, true>();
    TestHQ4BitGemmKernel<M, 31, 1, 16, false>();
    TestHQ4BitGemmKernel<M, 31, 1, 16, true>();
    TestHQ4BitGemmKernel<M, 31, 15, 16, false>();
    TestHQ4BitGemmKernel<M, 31, 15, 16, true>();
    TestHQ4BitGemmKernel<M, 31, 31, 16, false>();
    TestHQ4BitGemmKernel<M, 31, 31, 16, true>();
    TestHQ4BitGemmKernel<M, 31, 63, 128, false>();
    TestHQ4BitGemmKernel<M, 31, 63, 128, true>();
    TestHQ4BitGemmKernel<M, 31, 511, 128, false>();
    TestHQ4BitGemmKernel<M, 31, 511, 128, true>();
    TestHQ4BitGemmKernel<M, 128, 1, 16, false>();
    TestHQ4BitGemmKernel<M, 128, 1, 16, true>();
    TestHQ4BitGemmKernel<M, 128, 15, 16, false>();
    TestHQ4BitGemmKernel<M, 128, 15, 16, true>();
    TestHQ4BitGemmKernel<M, 128, 31, 16, false>();
    TestHQ4BitGemmKernel<M, 128, 31, 16, true>();
    TestHQ4BitGemmKernel<M, 128, 63, 128, false>();
    TestHQ4BitGemmKernel<M, 128, 63, 128, true>();
    TestHQ4BitGemmKernel<M, 128, 511, 128, false>();
    TestHQ4BitGemmKernel<M, 128, 511, 128, true>();
  }

  void ExecuteShort(void) override {
    ExecuteShort_T<1>();
    ExecuteShort_T<2>();
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

      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<159, 16, 16, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<160, 17, 32, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<161, 15, 64, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<160, 17, 128, 128>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<159, 16, 256, 128>>::RegisterShortExecute();
    } else {
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 16, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 32, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 64, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 128, 64>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasSQ8BitPrepackTest<1, 1, 256, 64>>::RegisterShortExecute();

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
  }
  return count;
});
