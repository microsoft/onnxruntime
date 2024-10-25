/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_sqnbitgemm_neon_fp16.cpp

Abstract:

    Tests for MLAS n-bit int block quantized GEMM on ARM CPU with input A type T1 fp16.

--*/

#include <vector>
#include <random>

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/sqnbitgemm.h"
#include "mlas_qnbit.h"

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

class MlasNeonFp16CastTest : public MlasTestBase {
 private:

  template <size_t count>
  void TestFp16ToFp32() {
    std::vector<unsigned short> src(count);
    std::vector<float> dest(count);

    for (size_t i = 0; i < count; i++) {
      src[i] = static_cast<unsigned short>(i);
    }

    MlasCastF16ToF32KernelNeon(src.data(), dest.data(), count);

    for (size_t i = 0; i < count; i++) {
      if ((src[i] & 0x1c00) == 0x1c00) continue;  // skip inf and nan
      ASSERT_EQ(dest[i], MLAS_FP16::FromBits(src[i]).ToFloat());
    }
  }

  template <size_t count>
  void TestFp32ToFp16() {
    std::vector<float> src(count);
    std::vector<unsigned short> dest(count);

    for (size_t i = 0; i < count; i++) {
      src[i] = static_cast<float>(i) + 0.125f;
    }

    MlasCastF32ToF16KernelNeon(src.data(), dest.data(), count);

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
  std::random_device _rd;  // a seed source for the random number engine
  std::mt19937 _gen; // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> _distrib;

  MLAS_FORCEINLINE
  void InitializeBuffer(std::vector<uint8_t>& buffer) {
    size_t count = buffer.size();
    for (size_t i = 0; i < count; i++) {
      buffer[i] = static_cast<uint8_t>(_distrib(_gen));
    }
  }

  template<size_t Ldb>
  MLAS_FORCEINLINE
  void Transpose8x8(std::vector<uint8_t>& src, size_t n, size_t k, std::vector<uint8_t>& dst) {
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
  void PrepackSlice(std::vector<uint8_t>& src, size_t j, std::vector<uint8_t>& dst) {
    for (size_t i = 0; i < 8; i++) {
      uint8_t v0 = GetInt4(src[j + (i >> 1)], i);
      uint8_t v1 = GetInt4(src[j + ((8 + i) >> 1)], i + 8);
      dst[j + i] = v0 | (v1 << 4);
    }
  }

  template <size_t Ldb, size_t N, size_t K>
  MLAS_FORCEINLINE
  void Prepack(std::vector<uint8_t>& src, std::vector<uint8_t>& dst) {
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

  template<size_t Ldb, size_t N, size_t K>
  MLAS_FORCEINLINE
  void Check(std::vector<uint8_t>& packed, std::vector<uint8_t>& ref) {
    size_t n = 0;
    for (; n + 8 <= N; n += 8) {
      for (size_t i = 0; i < K; i += 2) {
        for (size_t j = 0; j < 8; ++j) {
          ASSERT_EQ(packed[n * Ldb + (i >> 1) * 8 + j], ref[n * Ldb + (i >> 1) * 8 + j])
              << " n " << n << " i " << i << " j " << j;
        }
      }
    }

    for (; n < N; ++n) {
      for (size_t i = 0; i < K; i += 2) {
        ASSERT_EQ(packed[n * Ldb + (i >> 1)], ref[n * Ldb + (i >> 1)])
            << " n " << n << " i " << i;
      }
    }
  }

  template <size_t N, size_t K, size_t BlkLen>
  void TestPrepack() {
    constexpr size_t Bits = 4;
    constexpr size_t Ldb = (((K + BlkLen - 1) & (~(BlkLen - 1))) * Bits + 7) / 8;
    constexpr size_t BufferSize = N * Ldb;

    std::vector<uint8_t> input(BufferSize), packed(BufferSize), ref(BufferSize);
    InitializeBuffer(input);
    MlasSQNBitGemmPackQuantBData(
        N, K, Bits, BlkLen, MLAS_SQNBIT_GEMM_COMPUTE_TYPE::CompFp16, input.data(), packed.data(), nullptr
    );
    Prepack<Ldb, N, K>(input, ref);
    Check<Ldb, N, K>(packed, ref);
  }

 public:
  MlasNeonFp16PrepackTest()
    : _gen(_rd()), _distrib(0, 255) {
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
  std::random_device _rd;  // a seed source for the random number engine
  std::mt19937 _gen; // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> _distrib;
  std::uniform_real_distribution<float> _distribFp;

  MLAS_FORCEINLINE
  void InitializeBuffer(std::vector<uint8_t>& buffer) {
    size_t count = buffer.size();
    for (size_t i = 0; i < count; i++) {
      buffer[i] = static_cast<uint8_t>(_distrib(_gen));
    }
  }

  MLAS_FORCEINLINE
  void InitializeBuffer(std::vector<MLAS_FP16>& buffer) {
    size_t count = buffer.size();
    for (size_t i = 0; i < count; i++) {
      buffer[i] = MLAS_FP16(_distribFp(_gen));
    }
  }

  MLAS_FORCEINLINE
  uint8_t GetInt4(uint8_t v, size_t i) {
    return (i & 1) ? (v >> 4) : (v & 0x0f);
  }

  template <size_t N, size_t K, size_t BlkLen, bool UseZeroPoints>
  void DequantB(const std::vector<uint8_t>& src, std::vector<MLAS_FP16>& dst,
                const std::vector<MLAS_FP16>& scales, const std::vector<uint8_t>& zero_points) {
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

  template<size_t Ldb, size_t N, size_t K>
  MLAS_FORCEINLINE
  void Check(std::vector<MLAS_FP16>& target, std::vector<MLAS_FP16>& ref) {
    size_t n = 0;
    for (; n + 8 <= N; n += 8) {
      for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < 8; ++j) {
          size_t idx = n * Ldb + i * 8 + j;
          ASSERT_TRUE(FloatEqual(target[idx], ref[idx], 0.01f, 0.01f))
              << " v0 " << target[idx] << " v1 " << ref[idx]
              << " n " << n << " i " << i << " j " << j;
        }
      }
    }

    for (; n < N; ++n) {
      for (size_t i = 0; i < K; ++i) {
        size_t idx = n * Ldb + i;
        ASSERT_TRUE(FloatEqual(target[idx], ref[idx], 0.01f, 0.01f))
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

    std::vector<uint8_t> input(BCount / 2), zero_points(ZpSize);
    std::vector<MLAS_FP16> dequant(BCount), ref(BCount), scales(ScaleCount);
    InitializeBuffer(input);
    InitializeBuffer(zero_points);
    InitializeBuffer(scales);
    GetMlasPlatform().SQNBitGemmDispatch->Q4BitBlkDequantBForSgemm_CompFp16(
        BlkLen, dequant.data(), reinterpret_cast<std::byte*>(input.data()), scales.data(),
        UseZeroPoints ? reinterpret_cast<std::byte*>(zero_points.data()) : nullptr,
        N, K, BlkNum
    );
    DequantB<N, K, BlkLen, UseZeroPoints>(input, ref, scales, zero_points);
    Check<BlkLen * BlkNum, N, K>(dequant, ref);
  }

 public:
  MlasNeonFp16DequantBTest()
    : _gen(_rd()), _distrib(0, 255), _distribFp(0.5f, 2.0f) {
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

class MlasNeonFp16SQ4BitGemmKernelTest : public MlasTestBase {
 private:
  std::random_device _rd;  // a seed source for the random number engine
  std::mt19937 _gen; // mersenne_twister_engine seeded with rd()

  MLAS_FORCEINLINE
  void InitializeBuffer(std::vector<MLAS_FP16>& buffer, float min, float max) {
    std::uniform_real_distribution<float> distrib(min, max);
    size_t count = buffer.size();
    for (size_t i = 0; i < count; i++) {
      buffer[i] = MLAS_FP16(distrib(_gen));
    }
  }


  MLAS_FORCEINLINE
  bool FloatEqual(MLAS_FP16 v0, MLAS_FP16 v1, float rtol, float atol) {
    float f0 = v0.ToFloat(), f1 = v1.ToFloat();
    return std::abs(f0 - f1) <= std::abs(f1 * rtol) + atol;
  }

  template <size_t ldb, size_t N, size_t K>
  float GetBVal(std::vector<MLAS_FP16>& B, size_t n, size_t k) {
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
  void MatMul(std::vector<MLAS_FP16>& A, std::vector<MLAS_FP16>& B,
              std::vector<MLAS_FP16>& bias, std::vector<MLAS_FP16>& C) {
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

  template<size_t Ldc, size_t M, size_t N>
  MLAS_FORCEINLINE
  void Check(std::vector<MLAS_FP16>& target, std::vector<MLAS_FP16>& ref) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        size_t i = m * Ldc + n;
        ASSERT_TRUE(FloatEqual(target[i], ref[i], 0.01f, 0.02f))
            << " v0 " << target[i] << " v1 " << ref[i]
            << " m " << m << " n " << n;
      }
    }
  }

  template <size_t M, size_t N, size_t K, size_t BlkLen, bool UseBias>
  void TestSQ4BitGemmKernel() {
    constexpr size_t BlkNum = (K + BlkLen - 1) / BlkLen;
    constexpr size_t ldb = BlkNum * BlkLen;

    std::vector<MLAS_FP16> A(M * K), B(ldb * N), C(M * N), ref(M * N), bias(N);
    InitializeBuffer(A, -0.25f, 0.25f);
    InitializeBuffer(B, -0.25f, 0.25f);
    InitializeBuffer(bias, -5.0f, 5.0f);

    GetMlasPlatform().SQNBitGemmDispatch->SQ4BitGemmKernel_CompFp16(
        A.data(), B.data(), UseBias ? bias.data() : nullptr, C.data(), M, N, K, K, ldb, N, 64, 32
    );
    MatMul<M, N, K, ldb, UseBias>(A, B, bias, ref);
    Check<N, M, N>(C, ref);
  }

 public:
  MlasNeonFp16SQ4BitGemmKernelTest()
    : _gen(_rd()) {
  }

  static const char* GetTestSuiteName() {
    return "NeonFp16SQ4BitGemmKernel";
  }

  template <size_t M>
  void ExecuteShort_T(void) {
    TestSQ4BitGemmKernel<M, 1, 1, 16, false>();
    TestSQ4BitGemmKernel<M, 1, 1, 16, true>();
    TestSQ4BitGemmKernel<M, 1, 15, 16, false>();
    TestSQ4BitGemmKernel<M, 1, 15, 16, true>();
    TestSQ4BitGemmKernel<M, 1, 31, 16, false>();
    TestSQ4BitGemmKernel<M, 1, 31, 16, true>();
    TestSQ4BitGemmKernel<M, 8, 1, 16, false>();
    TestSQ4BitGemmKernel<M, 8, 1, 16, true>();
    TestSQ4BitGemmKernel<M, 8, 16, 16, false>();
    TestSQ4BitGemmKernel<M, 8, 16, 16, true>();
    TestSQ4BitGemmKernel<M, 9, 31, 16, false>();
    TestSQ4BitGemmKernel<M, 9, 31, 16, true>();
    TestSQ4BitGemmKernel<M, 9, 33, 32, false>();
    TestSQ4BitGemmKernel<M, 9, 33, 32, true>();
    TestSQ4BitGemmKernel<M, 15, 33, 16, false>();
    TestSQ4BitGemmKernel<M, 15, 33, 16, true>();
    TestSQ4BitGemmKernel<M, 23, 71, 16, false>();
    TestSQ4BitGemmKernel<M, 23, 71, 16, true>();
    TestSQ4BitGemmKernel<M, 17, 96, 128, false>();
    TestSQ4BitGemmKernel<M, 17, 96, 128, true>();
    TestSQ4BitGemmKernel<M, 31, 519, 16, false>();
    TestSQ4BitGemmKernel<M, 31, 519, 16, true>();
  }

  void ExecuteShort(void) override {
    ExecuteShort_T<1>();
    ExecuteShort_T<7>();
    ExecuteShort_T<23>();
    ExecuteShort_T<63>();
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasNeonFp16CastTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNeonFp16PrepackTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNeonFp16DequantBTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNeonFp16SQ4BitGemmKernelTest>::RegisterShortExecute();
  }
  return count;
});

#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
