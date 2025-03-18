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

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

class MlasNeonFp16CastTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> fp32Buffer_;
  MatrixGuardBuffer<unsigned short> fp16Buffer_;

  template <size_t count>
  void TestFp16ToFp32() {
    const auto* src = fp16Buffer_.GetFilledBuffer(count, [](unsigned short* start, size_t size) {
      for (size_t i = 0; i < size; i++) {
        start[i] = static_cast<unsigned short>(i);
      }
    });
    auto* dest = fp32Buffer_.GetBuffer(count, true);

    MlasCastF16ToF32KernelNeon(src, dest, count);

    for (size_t i = 0; i < count; i++) {
      if ((src[i] & 0x1c00) == 0x1c00) continue;  // skip inf and nan
      ASSERT_EQ(dest[i], MLAS_FP16::FromBits(src[i]).ToFloat());
    }
  }

  template <size_t count>
  void TestFp32ToFp16() {
    const auto* src = fp32Buffer_.GetFilledBuffer(count, [](float* p, size_t size) {
      for (size_t i = 0; i < size; i++) {
        p[i] = static_cast<float>(i) + 0.125f;
      }
    });
    auto* dest = fp16Buffer_.GetBuffer(count, true);

    MlasCastF32ToF16KernelNeon(src, dest, count);

    for (size_t i = 0; i < count; i++) {
      ASSERT_EQ(dest[i], MLAS_FP16(src[i]).val);
    }
  }

 public:
  static const char* GetTestSuiteName() {
    return "NeonFp16Cast";
  }

  void ExecuteShort(void) override {
    TestFp16ToFp32<(1 << 16)>();
    TestFp16ToFp32<1>();
    TestFp16ToFp32<4>();
    TestFp16ToFp32<7>();
    TestFp32ToFp16<(1 << 16)>();
    TestFp32ToFp16<3>();
    TestFp32ToFp16<4>();
    TestFp32ToFp16<6>();
  }
};

class MlasNeonFp16PrepackTest : public MlasTestBase {
 private:
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib_;
  MatrixGuardBuffer<uint8_t> input_, ref_, packed_;

  template <size_t Ldb>
  MLAS_FORCEINLINE void Transpose8x8(const uint8_t* src, size_t n, size_t k, uint8_t* dst) {
    for (size_t c = 0; c < 8; c++) {
      for (size_t r = 0; r < 8; r++) {
        size_t i = (n + c) * Ldb + r + k;
        size_t j = n * Ldb + (r + k) * 8 + c;
        dst[j] = src[i];
      }
    }
  }

  MLAS_FORCEINLINE
  uint8_t GetInt4(uint8_t v, size_t i) {
    return (i & 1) ? (v >> 4) : (v & 0x0f);
  }

  MLAS_FORCEINLINE
  void PrepackSlice(const uint8_t* src, size_t j, uint8_t* dst) {
    for (size_t i = 0; i < 8; i++) {
      uint8_t v0 = GetInt4(src[j + (i >> 1)], i);
      uint8_t v1 = GetInt4(src[j + ((8 + i) >> 1)], i + 8);
      dst[j + i] = v0 | (v1 << 4);
    }
  }

  template <size_t Ldb, size_t N, size_t K>
  MLAS_FORCEINLINE void Prepack(const uint8_t* src, uint8_t* dst) {
    size_t n = 0;
    for (; n + 8 <= N; n += 8) {
      for (size_t k = 0; k < Ldb; k += 8) {
        Transpose8x8<Ldb>(src, n, k, dst);
      }
    }

    for (; n < N; ++n) {
      for (size_t k = 0; k < Ldb; k += 8) {
        PrepackSlice(src, n * Ldb + k, dst);
      }
    }
  }

  template <size_t Ldb, size_t N, size_t K>
  MLAS_FORCEINLINE void Check(const uint8_t* packed, const uint8_t* ref) {
    size_t n = 0;
    for (; n + 8 <= N; n += 8) {
      for (size_t i = 0; i < K; i += 2) {
        for (size_t j = 0; j < 8; ++j) {
          ASSERT_EQ(packed[n * Ldb + (i >> 1) * 8 + j], ref[n * Ldb + (i >> 1) * 8 + j])
              << " seed " << seed_
              << " n " << n << " i " << i << " j " << j;
        }
      }
    }

    for (; n < N; ++n) {
      for (size_t i = 0; i < K; i += 2) {
        ASSERT_EQ(packed[n * Ldb + (i >> 1)], ref[n * Ldb + (i >> 1)])
            << " seed " << seed_
            << " n " << n << " i " << i;
      }
    }
  }

  template <size_t N, size_t K, size_t BlkLen>
  void TestPrepack() {
    constexpr size_t Bits = 4;
    constexpr size_t Ldb = (((K + BlkLen - 1) & (~(BlkLen - 1))) * Bits + 7) / 8;
    constexpr size_t BufferSize = N * Ldb;
    auto InitializeBuffer = [this](uint8_t* buffer, size_t count) {
      for (size_t i = 0; i < count; i++) {
        buffer[i] = static_cast<uint8_t>(distrib_(gen_));
      }
    };

    const auto* input = input_.GetFilledBuffer(BufferSize, InitializeBuffer);
    auto* packed = packed_.GetBuffer(BufferSize, true);
    auto* ref = ref_.GetBuffer(BufferSize, true);
    MlasQNBitGemmPackQuantBData(
        N, K, Bits, BlkLen, MLAS_QNBIT_GEMM_COMPUTE_TYPE::HQNBIT_CompFp16, input, packed,
        nullptr, false, nullptr, nullptr);
    Prepack<Ldb, N, K>(input, ref);
    Check<Ldb, N, K>(packed, ref);
  }

 public:
  MlasNeonFp16PrepackTest()
      : seed_(19287), gen_(seed_), distrib_(0, 255) {
  }

  static const char* GetTestSuiteName() {
    return "NeonFp16Prepack";
  }

  void ExecuteShort(void) override {
    TestPrepack<1, 1, 16>();
    TestPrepack<1, 15, 16>();
    TestPrepack<1, 31, 16>();
    TestPrepack<8, 1, 16>();
    TestPrepack<8, 16, 16>();
    TestPrepack<9, 31, 16>();
    TestPrepack<9, 33, 32>();
    TestPrepack<15, 33, 16>();
    TestPrepack<17, 67, 16>();
    TestPrepack<17, 96, 128>();
    TestPrepack<263, 263, 16>();
  }
};

class MlasNeonFp16DequantBTest : public MlasTestBase {
 private:
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib_;
  std::uniform_real_distribution<float> _distribFp;
  MatrixGuardBuffer<uint8_t> input_, zero_points_;
  MatrixGuardBuffer<MLAS_FP16> dequant_, ref_, scales_;

  MLAS_FORCEINLINE
  uint8_t GetInt4(uint8_t v, size_t i) {
    return (i & 1) ? (v >> 4) : (v & 0x0f);
  }

  template <size_t N, size_t K, size_t BlkLen, bool UseZeroPoints>
  void DequantB(const uint8_t* src, MLAS_FP16* dst, const MLAS_FP16* scales, const uint8_t* zero_points) {
    constexpr size_t blkNum = (K + BlkLen - 1) / BlkLen;
    constexpr size_t ld_src = (blkNum * BlkLen + 1) / 2;
    constexpr size_t ld_dst = blkNum * BlkLen;
    constexpr size_t ld_zp = (blkNum + 1) / 2;
    size_t n = 0;
    for (; n + 8 <= N; n += 8) {
      size_t i_src = n * ld_src, i_dst = n * ld_dst, i_scale = n * blkNum, i_zp = n * ld_zp;
      for (size_t blk = 0; blk < blkNum; i_zp += (blk & 1), ++blk, ++i_scale) {
        for (size_t i = 0; i < BlkLen; i += 2, i_dst += 8) {
          for (size_t j = 0; j < 8; ++j, ++i_src, ++i_dst) {
            uint8_t v = src[i_src];
            float v0 = static_cast<float>(GetInt4(v, 0));
            float v1 = static_cast<float>(GetInt4(v, 1));
            float zp = static_cast<float>(UseZeroPoints ? GetInt4(zero_points[i_zp + ld_zp * j], blk) : 8);
            float scale = scales[i_scale + blkNum * j];
            dst[i_dst] = MLAS_FP16(v0 * scale - zp * scale);
            dst[i_dst + 8] = MLAS_FP16(v1 * scale - zp * scale);
          }
        }
      }
    }

    for (; n < N; ++n) {
      size_t i_src = n * ld_src, i_dst = n * ld_dst, i_scale = n * blkNum, i_zp = n * ld_zp;
      for (size_t blk = 0; blk < blkNum; i_zp += (blk & 1), ++blk, ++i_scale) {
        float zp = static_cast<float>(UseZeroPoints ? GetInt4(zero_points[i_zp], blk) : 8);
        float scale = scales[i_scale];
        for (size_t i = 0; i < BlkLen; i += 16, i_dst += 8) {
          for (size_t j = 0; j < 16; j += 2, ++i_src, ++i_dst) {
            uint8_t v = src[i_src];
            float v0 = static_cast<float>(GetInt4(v, 0));
            float v1 = static_cast<float>(GetInt4(v, 1));
            dst[i_dst] = MLAS_FP16(v0 * scale - zp * scale);
            dst[i_dst + 8] = MLAS_FP16(v1 * scale - zp * scale);
          }
        }
      }
    }
  }

  MLAS_FORCEINLINE
  bool FloatEqual(MLAS_FP16 v0, MLAS_FP16 v1, float rtol, float atol) {
    float f0 = std::abs(v0.ToFloat()), f1 = std::abs(v1.ToFloat());
    return std::abs(f0 - f1) <= f1 * rtol + atol;
  }

  template <size_t Ldb, size_t N, size_t K>
  MLAS_FORCEINLINE void Check(const MLAS_FP16* target, const MLAS_FP16* ref) {
    size_t n = 0;
    for (; n + 8 <= N; n += 8) {
      for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < 8; ++j) {
          size_t idx = n * Ldb + i * 8 + j;
          ASSERT_TRUE(FloatEqual(target[idx], ref[idx], 0.01f, 0.01f))
              << " seed " << seed_
              << " v0 " << target[idx] << " v1 " << ref[idx]
              << " n " << n << " i " << i << " j " << j;
        }
      }
    }

    for (; n < N; ++n) {
      for (size_t i = 0; i < K; ++i) {
        size_t idx = n * Ldb + i;
        ASSERT_TRUE(FloatEqual(target[idx], ref[idx], 0.01f, 0.01f))
            << " seed " << seed_
            << " v0 " << target[idx] << " v1 " << ref[idx]
            << " n " << n << " i " << i;
      }
    }
  }

  template <size_t N, size_t K, size_t BlkLen, bool UseZeroPoints>
  void TestDequant() {
    constexpr size_t BlkNum = (K + BlkLen - 1) / BlkLen;
    constexpr size_t BCount = BlkNum * BlkLen * N;
    constexpr size_t ScaleCount = N * BlkNum;
    constexpr size_t ZpSize = N * ((BlkNum + 1) / 2);

    auto InitializeBuffer_i8 = [this](uint8_t* buffer, size_t count) {
      for (size_t i = 0; i < count; i++) {
        buffer[i] = static_cast<uint8_t>(distrib_(gen_));
      }
    };

    auto InitializeBuffer_fp16 = [this](MLAS_FP16* buffer, size_t count) {
      for (size_t i = 0; i < count; i++) {
        buffer[i] = MLAS_FP16(_distribFp(gen_));
      }
    };

    const auto* input = input_.GetFilledBuffer(BCount / 2, InitializeBuffer_i8);
    const auto* zero_points = zero_points_.GetFilledBuffer(ZpSize, InitializeBuffer_i8);
    auto* dequant = dequant_.GetBuffer(BCount);
    auto* ref = ref_.GetBuffer(BCount);
    const auto* scales = scales_.GetFilledBuffer(ScaleCount, InitializeBuffer_fp16);
    GetMlasPlatform().QNBitGemmDispatch->HQ4BitBlkDequantBForHgemm_CompFp16(
        BlkLen, dequant, reinterpret_cast<const std::byte*>(input), scales,
        UseZeroPoints ? reinterpret_cast<const std::byte*>(zero_points) : nullptr,
        N, K, BlkNum);
    DequantB<N, K, BlkLen, UseZeroPoints>(input, ref, scales, zero_points);
    Check<BlkLen * BlkNum, N, K>(dequant, ref);
  }

 public:
  MlasNeonFp16DequantBTest()
      : seed_(19287), gen_(seed_), distrib_(0, 255), _distribFp(0.5f, 2.0f) {
  }

  static const char* GetTestSuiteName() {
    return "NeonFp16DequantB";
  }

  void ExecuteShort(void) override {
    TestDequant<1, 1, 16, false>();
    TestDequant<1, 1, 16, true>();
    TestDequant<1, 15, 16, false>();
    TestDequant<1, 15, 16, true>();
    TestDequant<1, 31, 16, false>();
    TestDequant<1, 31, 16, true>();
    TestDequant<8, 1, 16, false>();
    TestDequant<8, 1, 16, true>();
    TestDequant<8, 16, 16, false>();
    TestDequant<8, 16, 16, true>();
    TestDequant<9, 31, 16, false>();
    TestDequant<9, 31, 16, true>();
    TestDequant<9, 33, 32, false>();
    TestDequant<9, 33, 32, true>();
    TestDequant<15, 33, 16, false>();
    TestDequant<15, 33, 16, true>();
    TestDequant<17, 67, 16, false>();
    TestDequant<17, 67, 16, true>();
    TestDequant<17, 96, 128, false>();
    TestDequant<17, 96, 128, true>();
    TestDequant<263, 263, 16, false>();
    TestDequant<263, 263, 16, true>();
  }
};

class MlasNeonFp16HQ4BitGemmKernelTest : public MlasTestBase {
 private:
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  MatrixGuardBuffer<MLAS_FP16> A_, B_, C_, ref_, bias_;

  MLAS_FORCEINLINE
  void InitializeBuffer(MLAS_FP16* buffer, float min, float max, size_t count) {
    std::uniform_real_distribution<float> distrib(min, max);
    for (size_t i = 0; i < count; i++) {
      buffer[i] = MLAS_FP16(distrib(gen_));
    }
  }

  MLAS_FORCEINLINE
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

  template <size_t M, size_t N, size_t K, size_t BlkLen, bool UseBias>
  void TestHQ4BitGemmKernel() {
    static_assert(M <= 2);
    constexpr size_t BlkNum = (K + BlkLen - 1) / BlkLen;
    constexpr size_t ldb = BlkNum * BlkLen;

    const auto* A = A_.GetFilledBuffer(M * K, [this](MLAS_FP16* p, size_t t) {
      InitializeBuffer(p, -0.25f, 0.25f, t);
    });
    const auto* B = B_.GetFilledBuffer(ldb * N, [this](MLAS_FP16* p, size_t t) {
      InitializeBuffer(p, -0.25f, 0.25f, t);
    });
    auto* C = C_.GetBuffer(M * N, true);
    auto* ref = ref_.GetBuffer(M * N, true);
    auto* bias = bias_.GetFilledBuffer(N, [this](MLAS_FP16* p, size_t t) {
      InitializeBuffer(p, -5.0f, 5.0f, t);
    });

    GetMlasPlatform().QNBitGemmDispatch->HQ4BitGemmKernel_CompFp16(
        A, B, UseBias ? bias : nullptr, C, M, N, K, K, ldb, N);

    MatMul<M, N, K, ldb, UseBias>(A, B, bias, ref);
    Check<N, M, N>(C, ref);
  }

 public:
  MlasNeonFp16HQ4BitGemmKernelTest()
      : seed_(19287), gen_(seed_) {
  }

  static const char* GetTestSuiteName() {
    return "NeonFp16HQ4BitGemmKernel";
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
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasNeonFp16CastTest>::RegisterShortExecute();
    if (GetMlasPlatform().QNBitGemmDispatch) {
      if (GetMlasPlatform().QNBitGemmDispatch->HQ4BitGemmPackQuantBData) {
        count += MlasDirectShortExecuteTests<MlasNeonFp16PrepackTest>::RegisterShortExecute();
      }
      if (GetMlasPlatform().QNBitGemmDispatch->HQ4BitBlkDequantBForHgemm_CompFp16) {
        count += MlasDirectShortExecuteTests<MlasNeonFp16DequantBTest>::RegisterShortExecute();
      }
      if (GetMlasPlatform().QNBitGemmDispatch->HQ4BitGemmKernel_CompFp16) {
        count += MlasDirectShortExecuteTests<MlasNeonFp16HQ4BitGemmKernelTest>::RegisterShortExecute();
      }
    }
  }
  return count;
});

#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
