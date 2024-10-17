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
#include "sqnbitgemm.h"

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

template <size_t N, size_t K, size_t BlkLen>
class MlasNeonFp16PrepackTest : public MlasTestBase {
 private:
  std::random_device rd;  // a seed source for the random number engine
  std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(0, 255);

  template<size_t BufferSize>
  MLAS_FORCEINLINE
  void InitializeBuffer(vector<uint8_t>& buffer) {
    for (size_t i = 0; i < BufferSize; i++) {
      buffer[i] = distrib(gen);
    }
  }

  template<size_t Ldb>
  MLAS_FORCEINLINE
  void Transpose8x8(std::vector<uint8_t>& src, size_t n, size_t k, std::vector<uint8_t>& dst) {
    for (size_t c = 0; c < 8; c++) {
      for (size_t r = 0; r < 8; r++) {
        size_t i = (n + c) * Ldb + r + k;
        size_t j = (r + n) * Ldb + c + k;
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

  template<size_t Ldb>
  MLAS_FORCEINLINE
  void Prepack(std::vector<uint8_t>& src, std::vector<uint8_t>& dst) {
    int n = 0;
    for (; n + 8 <= N; n += 8) {
      for (int k = 0; k < K; k += 8) {
        Transpose8x8<Ldb>(src, n, k, dst, Ldb);
      }
    }

    for (; n < N; ++n) {
      for (int k = 0; k < K; k += 8) {
        PrepackSlice(src, n * Ldb + k, dst);
      }
    }
  }

  template<size_t Ldb>
  MLAS_FORCEINLINE
  void Check(std::vector<uint8_t>& packed, std::vector<uint8_t>& ref) {
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < K; j++) {
        ASSERT_EQ(packed[i * Ldb + j], ref[i * Ldb + j]);
      }
    }
  }

  template<size_t N, size_t K, size_t BlkLen>
  void TestPrepack() {
    constexpr size_t Bits = 4;
    constexpr size_t Ldb = (((K + BlkLen - 1) & (~(BlkLen - 1))) * Bits + 7) / 8;
    constexpr size_t BufferSize = N * Ldb;

    std::vector<uint8_t> input(BufferSize), packed(BufferSize), ref(BufferSize);
    InitializeBuffer<BufferSize>(input);
    MlasSQNBitGemmPackQuantBData(
        N, K, Bits, BlkLen, MlasComputeType::CompFp16, input.data(), packed.data(), nullptr
    );
    Prepack<Ldb>(input, ref);
    Check<Ldb>(packed, ref);
  }

 public:
  static const char* GetTestSuiteName() {
    return "NeonFp16Prepack";
  }

  void ExecuteShort(void) override {
    TestPrepack<1, 1, 16>();
    TestPrepack<1, 15, 16>();
    TestPrepack<1, 31, 16>();
    TestPrepack<8, 16, 16>();
    TestPrepack<9, 31, 16>();
    TestPrepack<9, 33, 32>();
    TestPrepack<15, 33, 16>();
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasNeonFp16CastTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNeonFp16PrepackTest>::RegisterShortExecute();
  }
  return count;
});

#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
