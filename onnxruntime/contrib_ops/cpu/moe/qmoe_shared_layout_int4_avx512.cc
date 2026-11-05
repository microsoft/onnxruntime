// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AVX-512 CPU expert kernels for QMoEOffload (INT4 SharedExpertBlockV1, NVFP4 block-16 E4M3,
// MXFP4 block-32 E8M0, and FP16/BF16 activation row conversions).
// Compiled only when the host compiler supports AVX-512 (QMOE_COMPILER_SUPPORTS_AVX512).
// Runtime CPU support is verified via CpuHasAvx512() / MlasPlatformAvx512Supported() before
// any function in this translation unit is invoked.

#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS 1
#pragma warning(disable : 4996 4556 4068 4100)
#define QMOE_LAMBDA_INLINE [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
#define QMOE_LAMBDA_INLINE __attribute__((always_inline))
#else
#define QMOE_LAMBDA_INLINE
#endif

#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

#if !defined(QMOE_COMPILER_SUPPORTS_AVX512)
#if defined(__AVX512F__) || (defined(__GNUC__) && defined(__x86_64__))
#define QMOE_COMPILER_SUPPORTS_AVX512 1
#else
#define QMOE_COMPILER_SUPPORTS_AVX512 0
#endif
#endif

#if QMOE_COMPILER_SUPPORTS_AVX512

#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && !defined(__AVX512F__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,fma,f16c")
#define QMOE_SHARED_POP_GCC_TARGET 1
#endif

#include <immintrin.h>

#if defined(__clang__) && defined(__x86_64__) && !defined(__AVX512F__)
#pragma clang attribute push(__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma,f16c"))), apply_to = function)
#define QMOE_SHARED_POP_CLANG_TARGET 1
#endif

namespace onnxruntime {
namespace contrib {
namespace qmoe_shared {
namespace detail {

inline float Avx512HalfBitsToFloat(uint16_t h) noexcept {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

inline float Avx512BFloat16BitsToFloat(uint16_t b) noexcept {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline __m512 Avx512RoundBf16Vec(__m512 v) noexcept {
  const __m512i fbits = _mm512_castps_si512(v);
  const __m512i bias = _mm512_set1_epi32(0x7FFFu);
  const __m512i one = _mm512_set1_epi32(1);
  const __m512i exp_mask = _mm512_set1_epi32(0x7F800000);
  const __m512i magnitude = _mm512_and_epi32(fbits, _mm512_set1_epi32(0x7FFFFFFF));
  const __mmask16 finite = _mm512_cmplt_epi32_mask(magnitude, exp_mask);
  const __mmask16 nan = _mm512_cmpgt_epi32_mask(magnitude, exp_mask);
  const __m512i lsb = _mm512_and_epi32(_mm512_srli_epi32(fbits, 16), one);
  const __m512i rounded = _mm512_mask_add_epi32(fbits, finite, fbits, _mm512_add_epi32(bias, lsb));
  const __m512i canonical = _mm512_mask_mov_epi32(rounded, nan, _mm512_set1_epi32(0x7FC10000));
  return _mm512_castsi512_ps(_mm512_and_epi32(canonical, _mm512_set1_epi32(static_cast<int>(0xFFFF0000u))));
}

template <int RT>
void Group4Avx512(const uint8_t* group, const uint8_t* next_group, const uint16_t* scales, int64_t n_total, int64_t n0,
                  int64_t k, const float* a, float (&result)[4][RT]) {
  const int64_t blocks = k / kBlockSize;
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m512i eight = _mm512_set1_epi32(8);
  __m512 acc[4][RT];
  for (int c = 0; c < 4; ++c)
    for (int r = 0; r < RT; ++r) acc[c][r] = _mm512_setzero_ps();

  for (int64_t kb = 0; kb < blocks; ++kb) {
    const uint8_t* tile = group + (kb >> 1) * 128 + (kb & 1) * 16;
    const uint16_t* sc = scales + kb * n_total + n0;
    if (next_group != nullptr) _mm_prefetch(reinterpret_cast<const char*>(next_group + kb * 64), _MM_HINT_T0);
    __m512 al[RT];
    __m512 ah[RT];
    for (int r = 0; r < RT; ++r) {
      al[r] = _mm512_loadu_ps(a + r * k + kb * kBlockSize);
      ah[r] = _mm512_loadu_ps(a + r * k + kb * kBlockSize + 16);
    }
    for (int c = 0; c < 4; ++c) {
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tile + c * 32));
      const __m128i lo = _mm_and_si128(x, mask);
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), mask);
      const __m512 wl = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(lo), eight));
      const __m512 wh = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(hi), eight));
      const __m512 s = _mm512_set1_ps(Avx512HalfBitsToFloat(sc[c]));
      for (int r = 0; r < RT; ++r) {
        const __m512 prod = _mm512_fmadd_ps(wl, al[r], _mm512_mul_ps(wh, ah[r]));
        acc[c][r] = _mm512_fmadd_ps(prod, s, acc[c][r]);
      }
    }
  }
  for (int c = 0; c < 4; ++c)
    for (int r = 0; r < RT; ++r) result[c][r] = _mm512_reduce_add_ps(acc[c][r]);
}

template <int RT>
void RowTileAvx512(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias, int64_t n_total, int64_t k,
                   const float* a, float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  for (int64_t n0 = n_begin; n0 < n_end; n0 += kColumnsInterleaved) {
    const uint8_t* group = weights + (n0 / kColumnsInterleaved) * (kColumnsInterleaved * k / 2);
    float result[4][RT];
    const uint8_t* next_group = (n0 + kColumnsInterleaved < n_end) ? group + kColumnsInterleaved * k / 2 : nullptr;
    Group4Avx512<RT>(group, next_group, scales, n_total, n0, k, a, result);
    for (int c = 0; c < 4; ++c) {
      const float b = bias != nullptr ? Avx512HalfBitsToFloat(bias[n0 + c]) : 0.0f;
      for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result[c][r] + b;
    }
  }
}

void Avx512GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias, int64_t n_total, int64_t k,
                     const float* a, int64_t m, float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  int64_t r = 0;
  while (r < m) {
    const int64_t left = m - r;
    const float* ar = a + r * k;
    float* orow = out + r * out_stride;
    if (left >= 4) {
      RowTileAvx512<4>(weights, scales, bias, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 4;
    } else if (left == 3) {
      RowTileAvx512<3>(weights, scales, bias, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 3;
    } else if (left == 2) {
      RowTileAvx512<2>(weights, scales, bias, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 2;
    } else {
      RowTileAvx512<1>(weights, scales, bias, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 1;
    }
  }
}

template <bool ScaleIsBf16>
inline __m512 RoundInt4WeightsAvx512(__m512 values, float scale) {
  values = _mm512_mul_ps(values, _mm512_set1_ps(scale));
  // Use if constexpr/else so the BF16 path is discarded (not left unreachable) for FP16 scales; MSVC reports C4702.
  if constexpr (!ScaleIsBf16) {
    return _mm512_cvtph_ps(_mm512_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  } else {
    const __m512i bits = _mm512_castps_si512(values);
    const __mmask16 finite = _mm512_cmpneq_epi32_mask(_mm512_and_si512(bits, _mm512_set1_epi32(0x7f800000)),
                                                      _mm512_set1_epi32(0x7f800000));
    const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    const __m512i rounded = _mm512_and_si512(_mm512_add_epi32(bits, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), lsb)),
                                             _mm512_set1_epi32(static_cast<int>(0xffff0000u)));
    const __mmask16 nan = _mm512_cmp_ps_mask(values, values, _CMP_UNORD_Q);
    values = _mm512_castsi512_ps(_mm512_mask_mov_epi32(bits, finite, rounded));
    return _mm512_mask_mov_ps(values, nan, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000)));
  }
}

template <int RT, bool WeightsPrepacked, bool ScalesBlockMajor, bool ScaleIsBf16, bool DeferredScale>
void Int4Group8Avx512(const uint8_t* w_base, const uint8_t* next_w_base, const uint16_t* scales,
                      int64_t n_total, int64_t n0, int64_t k,
                      const float* a_reordered, float (&result)[8][RT]) {
  const int64_t blocks = k / kBlockSize;
  const int64_t group_stride = kColumnsInterleaved * k / 2;
  const int64_t raw_w_stride = k / 2;
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m512i eight = _mm512_set1_epi32(8);
  __m512 acc0_0 = _mm512_setzero_ps(), acc0_1 = _mm512_setzero_ps();
  __m512 acc1_0 = _mm512_setzero_ps(), acc1_1 = _mm512_setzero_ps();
  __m512 acc2_0 = _mm512_setzero_ps(), acc2_1 = _mm512_setzero_ps();
  __m512 acc3_0 = _mm512_setzero_ps(), acc3_1 = _mm512_setzero_ps();
  __m512 acc4_0 = _mm512_setzero_ps(), acc4_1 = _mm512_setzero_ps();
  __m512 acc5_0 = _mm512_setzero_ps(), acc5_1 = _mm512_setzero_ps();
  __m512 acc6_0 = _mm512_setzero_ps(), acc6_1 = _mm512_setzero_ps();
  __m512 acc7_0 = _mm512_setzero_ps(), acc7_1 = _mm512_setzero_ps();

  const uint8_t* w_base1 = WeightsPrepacked ? (w_base + group_stride) : (w_base + 4 * raw_w_stride);
  for (int64_t kb = 0; kb < blocks; ++kb) {
    const uint8_t* tile0 = WeightsPrepacked ? (w_base + (kb >> 1) * 128 + (kb & 1) * 16) : nullptr;
    const uint8_t* tile1 = WeightsPrepacked ? (w_base1 + (kb >> 1) * 128 + (kb & 1) * 16) : nullptr;
    if (next_w_base != nullptr) {
      if constexpr (WeightsPrepacked) {
        const int64_t pf_off = (kb >> 1) * 128 + (kb & 1) * 64;
        _mm_prefetch(reinterpret_cast<const char*>(next_w_base + pf_off), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(next_w_base + group_stride + pf_off), _MM_HINT_T0);
      }
    }
    const __m512 al0 = _mm512_loadu_ps(a_reordered + 0 * k + kb * kBlockSize);
    const __m512 ah0 = _mm512_loadu_ps(a_reordered + 0 * k + kb * kBlockSize + 16);
    __m512 al1, ah1;
    if constexpr (RT >= 2) {
      al1 = _mm512_loadu_ps(a_reordered + 1 * k + kb * kBlockSize);
      ah1 = _mm512_loadu_ps(a_reordered + 1 * k + kb * kBlockSize + 16);
    }
    auto step_col = [&](int c, const uint8_t* byte_ptr, __m512& a0, __m512& a1) QMOE_LAMBDA_INLINE {
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_ptr));
      const __m128i lo = _mm_and_si128(x, mask);
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), mask);
      __m512 wl = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(lo), eight));
      __m512 wh = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(hi), eight));
      const uint16_t s_bits = ScalesBlockMajor ? scales[kb * n_total + n0 + c] : scales[(n0 + c) * blocks + kb];
      const float s_val = ScaleIsBf16 ? Avx512BFloat16BitsToFloat(s_bits) : Avx512HalfBitsToFloat(s_bits);
      if constexpr (!DeferredScale) {
        wl = RoundInt4WeightsAvx512<ScaleIsBf16>(wl, s_val);
        wh = RoundInt4WeightsAvx512<ScaleIsBf16>(wh, s_val);
      }
      const __m512 s = _mm512_set1_ps(DeferredScale ? s_val : 1.0f);
      const __m512 prod0 = _mm512_fmadd_ps(wl, al0, _mm512_mul_ps(wh, ah0));
      a0 = _mm512_fmadd_ps(prod0, s, a0);
      if constexpr (RT >= 2) {
        const __m512 prod1 = _mm512_fmadd_ps(wl, al1, _mm512_mul_ps(wh, ah1));
        a1 = _mm512_fmadd_ps(prod1, s, a1);
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
  auto store_col = [&](int c, __m512 a0, __m512 a1) QMOE_LAMBDA_INLINE {
    result[c][0] = _mm512_reduce_add_ps(a0);
    if constexpr (RT >= 2) result[c][1] = _mm512_reduce_add_ps(a1);
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
void Int4Group4Avx512(const uint8_t* w_base, const uint8_t* next_w_base, const uint16_t* scales,
                      int64_t n_total, int64_t n0, int64_t k,
                      const float* a_reordered, float (&result)[4][RT]) {
  const int64_t blocks = k / kBlockSize;
  const int64_t raw_w_stride = k / 2;
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m512i eight = _mm512_set1_epi32(8);
  __m512 acc[4][RT];
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < RT; ++r) {
      acc[c][r] = _mm512_setzero_ps();
    }
  }

  for (int64_t kb = 0; kb < blocks; ++kb) {
    const uint8_t* tile = WeightsPrepacked ? (w_base + (kb >> 1) * 128 + (kb & 1) * 16) : nullptr;
    if (next_w_base != nullptr) {
      if constexpr (WeightsPrepacked) {
        _mm_prefetch(reinterpret_cast<const char*>(next_w_base + (kb >> 1) * 128 + (kb & 1) * 64), _MM_HINT_T0);
      } else if ((kb & 3) == 0) {
        const int64_t byte_off = kb * 16;
        for (int c = 0; c < 4; ++c) {
          _mm_prefetch(reinterpret_cast<const char*>(next_w_base + c * raw_w_stride + byte_off), _MM_HINT_T0);
        }
      }
    }
    __m512 al[RT], ah[RT];
    for (int r = 0; r < RT; ++r) {
      al[r] = _mm512_loadu_ps(a_reordered + r * k + kb * kBlockSize);
      ah[r] = _mm512_loadu_ps(a_reordered + r * k + kb * kBlockSize + 16);
    }
    auto step_col = [&](int c) QMOE_LAMBDA_INLINE {
      const uint8_t* byte_ptr = WeightsPrepacked ? (tile + c * 32) : (w_base + c * raw_w_stride + kb * 16);
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_ptr));
      const __m128i lo = _mm_and_si128(x, mask);
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), mask);
      __m512 wl = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(lo), eight));
      __m512 wh = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(hi), eight));
      const uint16_t s_bits = ScalesBlockMajor ? scales[kb * n_total + n0 + c] : scales[(n0 + c) * blocks + kb];
      const float s_val = ScaleIsBf16 ? Avx512BFloat16BitsToFloat(s_bits) : Avx512HalfBitsToFloat(s_bits);
      if constexpr (!DeferredScale) {
        wl = RoundInt4WeightsAvx512<ScaleIsBf16>(wl, s_val);
        wh = RoundInt4WeightsAvx512<ScaleIsBf16>(wh, s_val);
      }
      const __m512 s = _mm512_set1_ps(DeferredScale ? s_val : 1.0f);
      for (int r = 0; r < RT; ++r) {
        const __m512 prod = _mm512_fmadd_ps(wl, al[r], _mm512_mul_ps(wh, ah[r]));
        acc[c][r] = _mm512_fmadd_ps(prod, s, acc[c][r]);
      }
    };
    step_col(0);
    step_col(1);
    step_col(2);
    step_col(3);
  }
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < RT; ++r) {
      result[c][r] = _mm512_reduce_add_ps(acc[c][r]);
    }
  }
}

template <int RT, bool DeferredScale>
void Int4RowTileAvx512(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
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
    if constexpr (RT <= 2) {
      for (; n0 + 8 <= aligned_end; n0 += 8) {
        const uint8_t* group = weights + (n0 / kColumnsInterleaved) * group_stride;
        const uint8_t* next_group = (n0 + 16 <= aligned_end) ? (group + 2 * group_stride) : nullptr;
        float result8[8][RT];
        if (scales_are_block_major) {
          if (scale_is_bf16) {
            Int4Group8Avx512<RT, true, true, true, DeferredScale>(group, next_group, scales, n_total, n0, k,
                                                                  a_reordered, result8);
          } else {
            Int4Group8Avx512<RT, true, true, false, DeferredScale>(group, next_group, scales, n_total, n0, k,
                                                                   a_reordered, result8);
          }
        } else {
          if (scale_is_bf16) {
            Int4Group8Avx512<RT, true, false, true, DeferredScale>(group, next_group, scales, n_total, n0, k,
                                                                   a_reordered, result8);
          } else {
            Int4Group8Avx512<RT, true, false, false, DeferredScale>(group, next_group, scales, n_total, n0, k,
                                                                    a_reordered, result8);
          }
        }
        for (int c = 0; c < 8; ++c) {
          const float b = (bias != nullptr)
                              ? (bias_is_bf16 ? Avx512BFloat16BitsToFloat(bias[n0 + c]) : Avx512HalfBitsToFloat(bias[n0 + c]))
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
          Int4Group4Avx512<RT, true, true, true, DeferredScale>(group, next_group, scales, n_total, n0, k, a_reordered,
                                                                result);
        } else {
          Int4Group4Avx512<RT, true, true, false, DeferredScale>(group, next_group, scales, n_total, n0, k, a_reordered,
                                                                 result);
        }
      } else {
        if (scale_is_bf16) {
          Int4Group4Avx512<RT, true, false, true, DeferredScale>(group, next_group, scales, n_total, n0, k, a_reordered,
                                                                 result);
        } else {
          Int4Group4Avx512<RT, true, false, false, DeferredScale>(group, next_group, scales, n_total, n0, k,
                                                                  a_reordered, result);
        }
      }
      for (int c = 0; c < 4; ++c) {
        const float b = (bias != nullptr)
                            ? (bias_is_bf16 ? Avx512BFloat16BitsToFloat(bias[n0 + c]) : Avx512HalfBitsToFloat(bias[n0 + c]))
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
          Int4Group4Avx512<RT, false, true, true, DeferredScale>(w_base, next_w, scales, n_total, n0, k, a_reordered,
                                                                 result);
        } else {
          Int4Group4Avx512<RT, false, true, false, DeferredScale>(w_base, next_w, scales, n_total, n0, k, a_reordered,
                                                                  result);
        }
      } else {
        if (scale_is_bf16) {
          Int4Group4Avx512<RT, false, false, true, DeferredScale>(w_base, next_w, scales, n_total, n0, k, a_reordered,
                                                                  result);
        } else {
          Int4Group4Avx512<RT, false, false, false, DeferredScale>(w_base, next_w, scales, n_total, n0, k, a_reordered,
                                                                   result);
        }
      }
      for (int c = 0; c < 4; ++c) {
        const float b = (bias != nullptr)
                            ? (bias_is_bf16 ? Avx512BFloat16BitsToFloat(bias[n0 + c]) : Avx512HalfBitsToFloat(bias[n0 + c]))
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
void Avx512Int4GemmRangeImpl(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                             bool scales_are_block_major, bool weights_are_prepacked,
                             bool scale_is_bf16, bool bias_is_bf16,
                             int64_t n_total, int64_t k, const float* a_reordered, int64_t m,
                             float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  int64_t r = 0;
  while (r < m) {
    const int64_t left = m - r;
    const float* ar = a_reordered + r * k;
    float* orow = out + r * out_stride;
    if (left >= 8) {
      Int4RowTileAvx512<8, DeferredScale>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                          scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin,
                                          n_end);
      r += 8;
    } else if (left == 7) {
      Int4RowTileAvx512<7, DeferredScale>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                          scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin,
                                          n_end);
      r += 7;
    } else if (left == 6) {
      Int4RowTileAvx512<6, DeferredScale>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                          scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin,
                                          n_end);
      r += 6;
    } else if (left == 5) {
      Int4RowTileAvx512<5, DeferredScale>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                          scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin,
                                          n_end);
      r += 5;
    } else if (left == 4) {
      Int4RowTileAvx512<4, DeferredScale>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                          scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin,
                                          n_end);
      r += 4;
    } else if (left == 3) {
      Int4RowTileAvx512<3, DeferredScale>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                          scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin,
                                          n_end);
      r += 3;
    } else if (left == 2) {
      Int4RowTileAvx512<2, DeferredScale>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                          scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin,
                                          n_end);
      r += 2;
    } else {
      Int4RowTileAvx512<1, DeferredScale>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                          scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin,
                                          n_end);
      r += 1;
    }
  }
}

void Avx512Int4GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                         bool scales_are_block_major, bool weights_are_prepacked,
                         bool scale_is_bf16, bool bias_is_bf16,
                         int64_t n_total, int64_t k, const float* a_reordered, int64_t m,
                         float* out, int64_t out_stride, int64_t n_begin, int64_t n_end, bool deferred_scale) {
  if (deferred_scale) {
    Avx512Int4GemmRangeImpl<true>(weights, scales, bias, scales_are_block_major, weights_are_prepacked, scale_is_bf16,
                                  bias_is_bf16, n_total, k, a_reordered, m, out, out_stride, n_begin, n_end);
  } else {
    Avx512Int4GemmRangeImpl<false>(weights, scales, bias, scales_are_block_major, weights_are_prepacked, scale_is_bf16,
                                   bias_is_bf16, n_total, k, a_reordered, m, out, out_stride, n_begin, n_end);
  }
}

void Avx512HalfBitsToFloatRow(const uint16_t* src, float* dst, int64_t count) {
  int64_t i = 0;
  for (; i + 16 <= count; i += 16) {
    _mm512_storeu_ps(dst + i, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i))));
  }
  for (; i < count; ++i) dst[i] = Avx512HalfBitsToFloat(src[i]);
}

void Avx512FloatToHalfBitsRow(const float* src, uint16_t* dst, int64_t count) {
  int64_t i = 0;
  for (; i + 16 <= count; i += 16) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                        _mm512_cvtps_ph(_mm512_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
  }
  for (; i < count; ++i) dst[i] = FloatToHalf(src[i]);
}

void Avx512BFloat16BitsToFloatRow(const uint16_t* src, float* dst, int64_t count) {
  int64_t i = 0;
  for (; i + 16 <= count; i += 16) {
    const __m256i bf = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    const __m512i bits = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf), 16);
    _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(bits));
  }
  for (; i < count; ++i) dst[i] = BFloat16ToFloat(src[i]);
}

void Avx512FloatToBFloat16BitsRow(const float* src, uint16_t* dst, int64_t count) {
  int64_t i = 0;
  for (; i + 16 <= count; i += 16) {
    const __m512i rounded = _mm512_castps_si512(Avx512RoundBf16Vec(_mm512_loadu_ps(src + i)));
    const __m256i out16 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(rounded, 16));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), out16);
  }
  for (; i < count; ++i) dst[i] = FloatToBFloat16(src[i]);
}

}  // namespace detail
}  // namespace qmoe_shared
}  // namespace contrib
}  // namespace onnxruntime

#if defined(QMOE_SHARED_POP_GCC_TARGET)
#pragma GCC pop_options
#undef QMOE_SHARED_POP_GCC_TARGET
#endif

#if defined(QMOE_SHARED_POP_CLANG_TARGET)
#pragma clang attribute pop
#undef QMOE_SHARED_POP_CLANG_TARGET
#endif

#endif  // QMOE_COMPILER_SUPPORTS_AVX512
