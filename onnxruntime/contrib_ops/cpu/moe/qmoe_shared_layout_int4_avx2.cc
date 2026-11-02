// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AVX2 + FMA + F16C CPU INT4 expert kernels and FP16/BF16 row conversion helpers for QMoE.

#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS 1
#pragma warning(disable : 4996 4556 4068 4100)
#endif

#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if !defined(QMOE_COMPILER_SUPPORTS_AVX2)
#if defined(__AVX2__) || (defined(__GNUC__) && defined(__x86_64__)) || \
    (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define QMOE_COMPILER_SUPPORTS_AVX2 1
#else
#define QMOE_COMPILER_SUPPORTS_AVX2 0
#endif
#endif

#if QMOE_COMPILER_SUPPORTS_AVX2

#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")
#define QMOE_SHARED_POP_AVX2_GCC_TARGET 1
#endif

#include <immintrin.h>

#if defined(__clang__) && defined(__x86_64__)
#pragma clang attribute push(__attribute__((target("avx2,fma,f16c"))), apply_to = function)
#define QMOE_SHARED_POP_AVX2_CLANG_TARGET 1
#endif

#if defined(_MSC_VER)
#define QMOE_AVX2_LAMBDA_INLINE [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
#define QMOE_AVX2_LAMBDA_INLINE __attribute__((always_inline))
#else
#define QMOE_AVX2_LAMBDA_INLINE
#endif

namespace onnxruntime {
namespace contrib {
namespace qmoe_shared {
namespace detail {

inline float Avx2HalfBitsToFloat(uint16_t h) noexcept {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

inline float Avx2BFloat16BitsToFloat(uint16_t b) noexcept {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float Avx2RoundFp16Scalar(float v) noexcept {
  const __m128i half_bits = _mm_cvtps_ph(_mm_set_ss(v), _MM_FROUND_TO_NEAREST_INT);
  return _mm_cvtss_f32(_mm_cvtph_ps(half_bits));
}

void Avx2RoundFp16InPlace(float* buf, int64_t count) noexcept {
  int64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m128i half_bits = _mm256_cvtps_ph(_mm256_loadu_ps(buf + i), _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_ps(buf + i, _mm256_cvtph_ps(half_bits));
  }
  for (; i < count; ++i) {
    buf[i] = Avx2RoundFp16Scalar(buf[i]);
  }
}

inline __m256 Avx2RoundBf16Vec(__m256 v) noexcept {
  const __m256i fbits = _mm256_castps_si256(v);
  const __m256i bias = _mm256_set1_epi32(0x7FFFu);
  const __m256i one = _mm256_set1_epi32(1);
  const __m256i exp_mask = _mm256_set1_epi32(0x7F800000);
  const __m256i magnitude = _mm256_and_si256(fbits, _mm256_set1_epi32(0x7FFFFFFF));
  const __m256i is_nan = _mm256_cmpgt_epi32(magnitude, exp_mask);
  const __m256i is_finite = _mm256_cmpgt_epi32(exp_mask, magnitude);
  const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(fbits, 16), one);
  const __m256i rounding = _mm256_and_si256(is_finite, _mm256_add_epi32(bias, lsb));
  const __m256i rounded = _mm256_add_epi32(fbits, rounding);
  const __m256i canonical = _mm256_blendv_epi8(rounded, _mm256_set1_epi32(0x7FC10000), is_nan);
  return _mm256_castsi256_ps(_mm256_and_si256(canonical, _mm256_set1_epi32(static_cast<int>(0xFFFF0000u))));
}

inline float Avx2ReduceAddPs(__m256 v) noexcept {
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 sum128 = _mm_add_ps(lo, hi);
  const __m128 shuf = _mm_movehdup_ps(sum128);
  const __m128 sums = _mm_add_ps(sum128, shuf);
  return _mm_cvtss_f32(_mm_add_ss(sums, _mm_movehl_ps(shuf, sums)));
}

void Avx2HalfBitsToFloatRow(const uint16_t* src, float* dst, int64_t count) {
  int64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i))));
  }
  for (; i < count; ++i) dst[i] = Avx2HalfBitsToFloat(src[i]);
}

void Avx2FloatToHalfBitsRow(const float* src, uint16_t* dst, int64_t count) {
  int64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                     _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
  }
  for (; i < count; ++i) dst[i] = FloatToHalf(src[i]);
}

void Avx2BFloat16BitsToFloatRow(const uint16_t* src, float* dst, int64_t count) {
  int64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m128i bf = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    const __m256i bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf), 16);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(bits));
  }
  for (; i < count; ++i) dst[i] = BFloat16ToFloat(src[i]);
}

void Avx2FloatToBFloat16BitsRow(const float* src, uint16_t* dst, int64_t count) {
  int64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256i rounded = _mm256_castps_si256(Avx2RoundBf16Vec(_mm256_loadu_ps(src + i)));
    const __m256i shifted = _mm256_srli_epi32(rounded, 16);
    const __m128i lo128 = _mm256_castsi256_si128(shifted);
    const __m128i hi128 = _mm256_extracti128_si256(shifted, 1);
    const __m128i packed = _mm_packus_epi32(lo128, hi128);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), packed);
  }
  for (; i < count; ++i) dst[i] = FloatToBFloat16(src[i]);
}

template <bool ScaleIsBf16>
inline __m256 RoundInt4WeightsAvx2(__m256 values, float scale) {
  values = _mm256_mul_ps(values, _mm256_set1_ps(scale));
  // Use if constexpr/else so the BF16 path is discarded (not left unreachable) for FP16 scales; MSVC reports C4702.
  if constexpr (!ScaleIsBf16) {
    return _mm256_cvtph_ps(_mm256_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  } else {
    const __m256i bits = _mm256_castps_si256(values);
    const __m256i nonfinite = _mm256_cmpeq_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7f800000)),
                                                 _mm256_set1_epi32(0x7f800000));
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    const __m256i rounded = _mm256_and_si256(_mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb)),
                                             _mm256_set1_epi32(static_cast<int>(0xffff0000u)));
    const __m256 nan = _mm256_cmp_ps(values, values, _CMP_UNORD_Q);
    values = _mm256_castsi256_ps(_mm256_blendv_epi8(rounded, bits, nonfinite));
    return _mm256_blendv_ps(values, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000)), nan);
  }
}

template <int RT, bool WeightsPrepacked, bool ScalesBlockMajor, bool ScaleIsBf16, bool DeferredScale>
void Int4Group8Avx2(const uint8_t* w_base, const uint8_t* next_w_base, const uint16_t* scales,
                    int64_t n_total, int64_t n0, int64_t k,
                    const float* a_reordered, float (&result)[8][RT]) {
  const int64_t blocks = k / kBlockSize;
  const int64_t group_stride = kColumnsInterleaved * k / 2;
  const int64_t raw_w_stride = k / 2;
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi32(8);
  __m256 acc0_0 = _mm256_setzero_ps(), acc0_1 = _mm256_setzero_ps();
  __m256 acc1_0 = _mm256_setzero_ps(), acc1_1 = _mm256_setzero_ps();
  __m256 acc2_0 = _mm256_setzero_ps(), acc2_1 = _mm256_setzero_ps();
  __m256 acc3_0 = _mm256_setzero_ps(), acc3_1 = _mm256_setzero_ps();
  __m256 acc4_0 = _mm256_setzero_ps(), acc4_1 = _mm256_setzero_ps();
  __m256 acc5_0 = _mm256_setzero_ps(), acc5_1 = _mm256_setzero_ps();
  __m256 acc6_0 = _mm256_setzero_ps(), acc6_1 = _mm256_setzero_ps();
  __m256 acc7_0 = _mm256_setzero_ps(), acc7_1 = _mm256_setzero_ps();

  const uint8_t* w_base1 = WeightsPrepacked ? (w_base + group_stride) : (w_base + 4 * raw_w_stride);
  for (int64_t kb = 0; kb < blocks; ++kb) {
    const uint8_t* tile0 = WeightsPrepacked ? (w_base + (kb >> 1) * 128 + (kb & 1) * 16) : nullptr;
    const uint8_t* tile1 = WeightsPrepacked ? (w_base1 + (kb >> 1) * 128 + (kb & 1) * 16) : nullptr;
    if (next_w_base != nullptr && (kb & 1) == 0) {
      if constexpr (WeightsPrepacked) {
        _mm_prefetch(reinterpret_cast<const char*>(next_w_base + (kb >> 1) * 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(next_w_base + group_stride + (kb >> 1) * 128), _MM_HINT_T0);
      }
    }
    const __m256 al0_0 = _mm256_loadu_ps(a_reordered + 0 * k + kb * kBlockSize);
    const __m256 al0_1 = _mm256_loadu_ps(a_reordered + 0 * k + kb * kBlockSize + 8);
    const __m256 ah0_0 = _mm256_loadu_ps(a_reordered + 0 * k + kb * kBlockSize + 16);
    const __m256 ah0_1 = _mm256_loadu_ps(a_reordered + 0 * k + kb * kBlockSize + 24);
    __m256 al1_0, al1_1, ah1_0, ah1_1;
    if constexpr (RT >= 2) {
      al1_0 = _mm256_loadu_ps(a_reordered + 1 * k + kb * kBlockSize);
      al1_1 = _mm256_loadu_ps(a_reordered + 1 * k + kb * kBlockSize + 8);
      ah1_0 = _mm256_loadu_ps(a_reordered + 1 * k + kb * kBlockSize + 16);
      ah1_1 = _mm256_loadu_ps(a_reordered + 1 * k + kb * kBlockSize + 24);
    }
    auto step_col = [&](int c, const uint8_t* byte_ptr, __m256& a0, __m256& a1) QMOE_AVX2_LAMBDA_INLINE {
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_ptr));
      const __m128i lo = _mm_and_si128(x, mask);
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), mask);
      __m256 wl0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(lo), eight));
      __m256 wl1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_bsrli_si128(lo, 8)), eight));
      __m256 wh0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(hi), eight));
      __m256 wh1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_bsrli_si128(hi, 8)), eight));
      const uint16_t s_bits = ScalesBlockMajor ? scales[kb * n_total + n0 + c] : scales[(n0 + c) * blocks + kb];
      const float s_val = ScaleIsBf16 ? Avx2BFloat16BitsToFloat(s_bits) : Avx2HalfBitsToFloat(s_bits);
      if constexpr (!DeferredScale) {
        wl0 = RoundInt4WeightsAvx2<ScaleIsBf16>(wl0, s_val);
        wl1 = RoundInt4WeightsAvx2<ScaleIsBf16>(wl1, s_val);
        wh0 = RoundInt4WeightsAvx2<ScaleIsBf16>(wh0, s_val);
        wh1 = RoundInt4WeightsAvx2<ScaleIsBf16>(wh1, s_val);
      }
      const __m256 s = _mm256_set1_ps(DeferredScale ? s_val : 1.0f);
      const __m256 prod0_lo = _mm256_fmadd_ps(wl0, al0_0, _mm256_mul_ps(wh0, ah0_0));
      const __m256 prod0_hi = _mm256_fmadd_ps(wl1, al0_1, _mm256_mul_ps(wh1, ah0_1));
      a0 = _mm256_fmadd_ps(_mm256_add_ps(prod0_lo, prod0_hi), s, a0);
      if constexpr (RT >= 2) {
        const __m256 prod1_lo = _mm256_fmadd_ps(wl0, al1_0, _mm256_mul_ps(wh0, ah1_0));
        const __m256 prod1_hi = _mm256_fmadd_ps(wl1, al1_1, _mm256_mul_ps(wh1, ah1_1));
        a1 = _mm256_fmadd_ps(_mm256_add_ps(prod1_lo, prod1_hi), s, a1);
      }
    };
    step_col(0, WeightsPrepacked ? (tile0 + 0 * 32) : (w_base + 0 * raw_w_stride + kb * 16), acc0_0, acc0_1);
    step_col(1, WeightsPrepacked ? (tile0 + 1 * 32) : (w_base + 1 * raw_w_stride + kb * 16), acc1_0, acc1_1);
    step_col(2, WeightsPrepacked ? (tile0 + 2 * 32) : (w_base + 2 * raw_w_stride + kb * 16), acc2_0, acc2_1);
    step_col(3, WeightsPrepacked ? (tile0 + 3 * 32) : (w_base + 3 * raw_w_stride + kb * 16), acc3_0, acc3_1);
    step_col(4, WeightsPrepacked ? (tile1 + 0 * 32) : (w_base + 4 * raw_w_stride + kb * 16), acc4_0, acc4_1);
    step_col(5, WeightsPrepacked ? (tile1 + 1 * 32) : (w_base + 5 * raw_w_stride + kb * 16), acc5_0, acc5_1);
    step_col(6, WeightsPrepacked ? (tile1 + 2 * 32) : (w_base + 6 * raw_w_stride + kb * 16), acc6_0, acc6_1);
    step_col(7, WeightsPrepacked ? (tile1 + 3 * 32) : (w_base + 7 * raw_w_stride + kb * 16), acc7_0, acc7_1);
  }
  auto store_col = [&](int c, __m256 a0, __m256 a1) QMOE_AVX2_LAMBDA_INLINE {
    result[c][0] = Avx2ReduceAddPs(a0);
    if constexpr (RT >= 2) result[c][1] = Avx2ReduceAddPs(a1);
  };
  store_col(0, acc0_0, acc0_1);
  store_col(1, acc1_0, acc1_1);
  store_col(2, acc2_0, acc2_1);
  store_col(3, acc3_0, acc3_1);
  store_col(4, acc4_0, acc4_1);
  store_col(5, acc5_0, acc5_1);
  store_col(6, acc6_0, acc6_1);
  store_col(7, acc7_0, acc7_1);
}

template <int RT, bool WeightsPrepacked, bool ScalesBlockMajor, bool ScaleIsBf16, bool DeferredScale>
void Int4Group4Avx2(const uint8_t* w_base, const uint8_t* next_w_base, const uint16_t* scales,
                    int64_t n_total, int64_t n0, int64_t k,
                    const float* a_reordered, float (&result)[4][RT]) {
  const int64_t blocks = k / kBlockSize;
  const int64_t raw_w_stride = k / 2;
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi32(8);
  __m256 acc0_0 = _mm256_setzero_ps(), acc0_1 = _mm256_setzero_ps(), acc0_2 = _mm256_setzero_ps(), acc0_3 = _mm256_setzero_ps();
  __m256 acc1_0 = _mm256_setzero_ps(), acc1_1 = _mm256_setzero_ps(), acc1_2 = _mm256_setzero_ps(), acc1_3 = _mm256_setzero_ps();
  __m256 acc2_0 = _mm256_setzero_ps(), acc2_1 = _mm256_setzero_ps(), acc2_2 = _mm256_setzero_ps(), acc2_3 = _mm256_setzero_ps();
  __m256 acc3_0 = _mm256_setzero_ps(), acc3_1 = _mm256_setzero_ps(), acc3_2 = _mm256_setzero_ps(), acc3_3 = _mm256_setzero_ps();

  for (int64_t kb = 0; kb < blocks; ++kb) {
    const uint8_t* tile = WeightsPrepacked ? (w_base + (kb >> 1) * 128 + (kb & 1) * 16) : nullptr;
    if (next_w_base != nullptr) {
      if constexpr (WeightsPrepacked) {
        if ((kb & 1) == 0) _mm_prefetch(reinterpret_cast<const char*>(next_w_base + (kb >> 1) * 128), _MM_HINT_T0);
      } else if ((kb & 3) == 0) {
        const int64_t byte_off = kb * 16;
        for (int c = 0; c < 4; ++c) {
          _mm_prefetch(reinterpret_cast<const char*>(next_w_base + c * raw_w_stride + byte_off), _MM_HINT_T0);
        }
      }
    }
    const __m256 al0_0 = _mm256_loadu_ps(a_reordered + 0 * k + kb * kBlockSize);
    const __m256 al0_1 = _mm256_loadu_ps(a_reordered + 0 * k + kb * kBlockSize + 8);
    const __m256 ah0_0 = _mm256_loadu_ps(a_reordered + 0 * k + kb * kBlockSize + 16);
    const __m256 ah0_1 = _mm256_loadu_ps(a_reordered + 0 * k + kb * kBlockSize + 24);
    __m256 al1_0, al1_1, ah1_0, ah1_1;
    __m256 al2_0, al2_1, ah2_0, ah2_1;
    __m256 al3_0, al3_1, ah3_0, ah3_1;
    if constexpr (RT >= 2) {
      al1_0 = _mm256_loadu_ps(a_reordered + 1 * k + kb * kBlockSize);
      al1_1 = _mm256_loadu_ps(a_reordered + 1 * k + kb * kBlockSize + 8);
      ah1_0 = _mm256_loadu_ps(a_reordered + 1 * k + kb * kBlockSize + 16);
      ah1_1 = _mm256_loadu_ps(a_reordered + 1 * k + kb * kBlockSize + 24);
    }
    if constexpr (RT >= 3) {
      al2_0 = _mm256_loadu_ps(a_reordered + 2 * k + kb * kBlockSize);
      al2_1 = _mm256_loadu_ps(a_reordered + 2 * k + kb * kBlockSize + 8);
      ah2_0 = _mm256_loadu_ps(a_reordered + 2 * k + kb * kBlockSize + 16);
      ah2_1 = _mm256_loadu_ps(a_reordered + 2 * k + kb * kBlockSize + 24);
    }
    if constexpr (RT >= 4) {
      al3_0 = _mm256_loadu_ps(a_reordered + 3 * k + kb * kBlockSize);
      al3_1 = _mm256_loadu_ps(a_reordered + 3 * k + kb * kBlockSize + 8);
      ah3_0 = _mm256_loadu_ps(a_reordered + 3 * k + kb * kBlockSize + 16);
      ah3_1 = _mm256_loadu_ps(a_reordered + 3 * k + kb * kBlockSize + 24);
    }
    auto step_col = [&](int c, __m256& a0, __m256& a1, __m256& a2, __m256& a3) QMOE_AVX2_LAMBDA_INLINE {
      const uint8_t* byte_ptr = WeightsPrepacked ? (tile + c * 32) : (w_base + c * raw_w_stride + kb * 16);
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_ptr));
      const __m128i lo = _mm_and_si128(x, mask);
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), mask);
      __m256 wl0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(lo), eight));
      __m256 wl1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_bsrli_si128(lo, 8)), eight));
      __m256 wh0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(hi), eight));
      __m256 wh1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_bsrli_si128(hi, 8)), eight));
      const uint16_t s_bits = ScalesBlockMajor ? scales[kb * n_total + n0 + c] : scales[(n0 + c) * blocks + kb];
      const float s_val = ScaleIsBf16 ? Avx2BFloat16BitsToFloat(s_bits) : Avx2HalfBitsToFloat(s_bits);
      if constexpr (!DeferredScale) {
        wl0 = RoundInt4WeightsAvx2<ScaleIsBf16>(wl0, s_val);
        wl1 = RoundInt4WeightsAvx2<ScaleIsBf16>(wl1, s_val);
        wh0 = RoundInt4WeightsAvx2<ScaleIsBf16>(wh0, s_val);
        wh1 = RoundInt4WeightsAvx2<ScaleIsBf16>(wh1, s_val);
      }
      const __m256 s = _mm256_set1_ps(DeferredScale ? s_val : 1.0f);
      const __m256 prod0_lo = _mm256_fmadd_ps(wl0, al0_0, _mm256_mul_ps(wh0, ah0_0));
      const __m256 prod0_hi = _mm256_fmadd_ps(wl1, al0_1, _mm256_mul_ps(wh1, ah0_1));
      a0 = _mm256_fmadd_ps(_mm256_add_ps(prod0_lo, prod0_hi), s, a0);
      if constexpr (RT >= 2) {
        const __m256 prod1_lo = _mm256_fmadd_ps(wl0, al1_0, _mm256_mul_ps(wh0, ah1_0));
        const __m256 prod1_hi = _mm256_fmadd_ps(wl1, al1_1, _mm256_mul_ps(wh1, ah1_1));
        a1 = _mm256_fmadd_ps(_mm256_add_ps(prod1_lo, prod1_hi), s, a1);
      }
      if constexpr (RT >= 3) {
        const __m256 prod2_lo = _mm256_fmadd_ps(wl0, al2_0, _mm256_mul_ps(wh0, ah2_0));
        const __m256 prod2_hi = _mm256_fmadd_ps(wl1, al2_1, _mm256_mul_ps(wh1, ah2_1));
        a2 = _mm256_fmadd_ps(_mm256_add_ps(prod2_lo, prod2_hi), s, a2);
      }
      if constexpr (RT >= 4) {
        const __m256 prod3_lo = _mm256_fmadd_ps(wl0, al3_0, _mm256_mul_ps(wh0, ah3_0));
        const __m256 prod3_hi = _mm256_fmadd_ps(wl1, al3_1, _mm256_mul_ps(wh1, ah3_1));
        a3 = _mm256_fmadd_ps(_mm256_add_ps(prod3_lo, prod3_hi), s, a3);
      }
    };
    step_col(0, acc0_0, acc0_1, acc0_2, acc0_3);
    step_col(1, acc1_0, acc1_1, acc1_2, acc1_3);
    step_col(2, acc2_0, acc2_1, acc2_2, acc2_3);
    step_col(3, acc3_0, acc3_1, acc3_2, acc3_3);
  }
  auto store_col = [&](int c, __m256 a0, __m256 a1, __m256 a2, __m256 a3) QMOE_AVX2_LAMBDA_INLINE {
    result[c][0] = Avx2ReduceAddPs(a0);
    if constexpr (RT >= 2) result[c][1] = Avx2ReduceAddPs(a1);
    if constexpr (RT >= 3) result[c][2] = Avx2ReduceAddPs(a2);
    if constexpr (RT >= 4) result[c][3] = Avx2ReduceAddPs(a3);
  };
  store_col(0, acc0_0, acc0_1, acc0_2, acc0_3);
  store_col(1, acc1_0, acc1_1, acc1_2, acc1_3);
  store_col(2, acc2_0, acc2_1, acc2_2, acc2_3);
  store_col(3, acc3_0, acc3_1, acc3_2, acc3_3);
}

template <int RT, bool DeferredScale>
void Int4RowTileAvx2(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                     bool scales_are_block_major, bool weights_are_prepacked,
                     bool scale_is_bf16, bool bias_is_bf16,
                     int64_t n_total, int64_t k, const float* a_reordered,
                     float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  if (n_begin >= n_end) return;
  if (weights_are_prepacked) {
    const int64_t aligned_begin = (n_begin + kColumnsInterleaved - 1) & ~(kColumnsInterleaved - 1);
    const int64_t aligned_end = n_end & ~(kColumnsInterleaved - 1);
    if (aligned_begin >= aligned_end) {
      Int4GemmRange(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                    scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, RT,
                    out, out_stride, n_begin, n_end, /*force_scalar=*/true, DeferredScale);
      return;
    }
    if (n_begin < aligned_begin) {
      Int4GemmRange(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                    scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, RT,
                    out, out_stride, n_begin, aligned_begin, /*force_scalar=*/true, DeferredScale);
    }
    const int64_t group_stride = kColumnsInterleaved * k / 2;
    int64_t n0 = aligned_begin;
    if constexpr (RT == 1) {
      for (; n0 + 8 <= aligned_end; n0 += 8) {
        const uint8_t* group = weights + (n0 / kColumnsInterleaved) * group_stride;
        const uint8_t* next_group = (n0 + 16 <= aligned_end) ? (group + 2 * group_stride) : nullptr;
        float result8[8][RT];
        if (scales_are_block_major) {
          if (scale_is_bf16) {
            Int4Group8Avx2<RT, true, true, true, DeferredScale>(group, next_group, scales, n_total, n0, k, a_reordered,
                                                                result8);
          } else {
            Int4Group8Avx2<RT, true, true, false, DeferredScale>(group, next_group, scales, n_total, n0, k, a_reordered,
                                                                 result8);
          }
        } else {
          if (scale_is_bf16) {
            Int4Group8Avx2<RT, true, false, true, DeferredScale>(group, next_group, scales, n_total, n0, k, a_reordered,
                                                                 result8);
          } else {
            Int4Group8Avx2<RT, true, false, false, DeferredScale>(group, next_group, scales, n_total, n0, k,
                                                                  a_reordered, result8);
          }
        }
        for (int c = 0; c < 8; ++c) {
          const float b = (bias != nullptr)
                              ? (bias_is_bf16 ? Avx2BFloat16BitsToFloat(bias[n0 + c]) : Avx2HalfBitsToFloat(bias[n0 + c]))
                              : 0.0f;
          for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result8[c][r] + b;
        }
      }
    }
    for (; n0 < aligned_end; n0 += kColumnsInterleaved) {
      const uint8_t* group = weights + (n0 / kColumnsInterleaved) * group_stride;
      const uint8_t* next_group = (n0 + kColumnsInterleaved < aligned_end) ? (group + group_stride) : nullptr;
      float result[4][RT];
      if (scales_are_block_major) {
        if (scale_is_bf16) {
          Int4Group4Avx2<RT, true, true, true, DeferredScale>(group, next_group, scales, n_total, n0, k, a_reordered,
                                                              result);
        } else {
          Int4Group4Avx2<RT, true, true, false, DeferredScale>(group, next_group, scales, n_total, n0, k, a_reordered,
                                                               result);
        }
      } else {
        if (scale_is_bf16) {
          Int4Group4Avx2<RT, true, false, true, DeferredScale>(group, next_group, scales, n_total, n0, k, a_reordered,
                                                               result);
        } else {
          Int4Group4Avx2<RT, true, false, false, DeferredScale>(group, next_group, scales, n_total, n0, k, a_reordered,
                                                                result);
        }
      }
      for (int c = 0; c < 4; ++c) {
        const float b = (bias != nullptr)
                            ? (bias_is_bf16 ? Avx2BFloat16BitsToFloat(bias[n0 + c]) : Avx2HalfBitsToFloat(bias[n0 + c]))
                            : 0.0f;
        for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result[c][r] + b;
      }
    }
    if (aligned_end < n_end) {
      Int4GemmRange(weights, scales, bias, scales_are_block_major, weights_are_prepacked, scale_is_bf16, bias_is_bf16,
                    n_total, k, a_reordered, RT, out + (aligned_end - n_begin), out_stride, aligned_end, n_end,
                    /*force_scalar=*/true, DeferredScale);
    }
  } else {
    const int64_t raw_w_stride = k / 2;
    int64_t n0 = n_begin;
    for (; n0 + 4 <= n_end; n0 += 4) {
      const uint8_t* w_base = weights + n0 * raw_w_stride;
      const uint8_t* next_w = (n0 + 8 <= n_end) ? (weights + (n0 + 4) * raw_w_stride) : nullptr;
      float result[4][RT];
      if (scales_are_block_major) {
        if (scale_is_bf16) {
          Int4Group4Avx2<RT, false, true, true, DeferredScale>(w_base, next_w, scales, n_total, n0, k, a_reordered,
                                                               result);
        } else {
          Int4Group4Avx2<RT, false, true, false, DeferredScale>(w_base, next_w, scales, n_total, n0, k, a_reordered,
                                                                result);
        }
      } else {
        if (scale_is_bf16) {
          Int4Group4Avx2<RT, false, false, true, DeferredScale>(w_base, next_w, scales, n_total, n0, k, a_reordered,
                                                                result);
        } else {
          Int4Group4Avx2<RT, false, false, false, DeferredScale>(w_base, next_w, scales, n_total, n0, k, a_reordered,
                                                                 result);
        }
      }
      for (int c = 0; c < 4; ++c) {
        const float b = (bias != nullptr)
                            ? (bias_is_bf16 ? Avx2BFloat16BitsToFloat(bias[n0 + c]) : Avx2HalfBitsToFloat(bias[n0 + c]))
                            : 0.0f;
        for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result[c][r] + b;
      }
    }
    if (n0 < n_end) {
      Int4GemmRange(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                    scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, RT,
                    out + (n0 - n_begin), out_stride, n0, n_end, /*force_scalar=*/true, DeferredScale);
    }
  }
}

template <bool DeferredScale>
void Avx2Int4GemmRangeImpl(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                           bool scales_are_block_major, bool weights_are_prepacked,
                           bool scale_is_bf16, bool bias_is_bf16,
                           int64_t n_total, int64_t k, const float* a_reordered, int64_t m,
                           float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  int64_t r = 0;
  while (r < m) {
    const int64_t left = m - r;
    const float* ar = a_reordered + r * k;
    float* orow = out + r * out_stride;
    if (left >= 4) {
      Int4RowTileAvx2<4, DeferredScale>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                        scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 4;
    } else if (left == 3) {
      Int4RowTileAvx2<3, DeferredScale>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                        scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 3;
    } else if (left == 2) {
      Int4RowTileAvx2<2, DeferredScale>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                        scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 2;
    } else {
      Int4RowTileAvx2<1, DeferredScale>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                        scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 1;
    }
  }
}

void Avx2Int4GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                       bool scales_are_block_major, bool weights_are_prepacked,
                       bool scale_is_bf16, bool bias_is_bf16,
                       int64_t n_total, int64_t k, const float* a_reordered, int64_t m,
                       float* out, int64_t out_stride, int64_t n_begin, int64_t n_end, bool deferred_scale) {
  if (deferred_scale) {
    Avx2Int4GemmRangeImpl<true>(weights, scales, bias, scales_are_block_major, weights_are_prepacked, scale_is_bf16,
                                bias_is_bf16, n_total, k, a_reordered, m, out, out_stride, n_begin, n_end);
  } else {
    Avx2Int4GemmRangeImpl<false>(weights, scales, bias, scales_are_block_major, weights_are_prepacked, scale_is_bf16,
                                 bias_is_bf16, n_total, k, a_reordered, m, out, out_stride, n_begin, n_end);
  }
}

void Avx2GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias, int64_t n_total, int64_t k,
                   const float* a, int64_t m, float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  Avx2Int4GemmRange(weights, scales, bias, /*scales_are_block_major=*/true, /*weights_are_prepacked=*/true,
                    /*scale_is_bf16=*/false, /*bias_is_bf16=*/false, n_total, k, a, m,
                    out, out_stride, n_begin, n_end, /*deferred_scale=*/true);
}

}  // namespace detail
}  // namespace qmoe_shared
}  // namespace contrib
}  // namespace onnxruntime

#if defined(QMOE_SHARED_POP_AVX2_GCC_TARGET)
#pragma GCC pop_options
#undef QMOE_SHARED_POP_AVX2_GCC_TARGET
#endif

#if defined(QMOE_SHARED_POP_AVX2_CLANG_TARGET)
#pragma clang attribute pop
#undef QMOE_SHARED_POP_AVX2_CLANG_TARGET
#endif

#endif  // QMOE_COMPILER_SUPPORTS_AVX2
