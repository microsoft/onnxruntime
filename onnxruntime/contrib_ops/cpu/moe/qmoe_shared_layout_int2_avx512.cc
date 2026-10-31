// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AVX-512 CPU INT2 expert kernels for QMoE.
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
#include <cstdint>
#include <cstring>

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

inline float Int2Avx512HalfBitsToFloat(uint16_t h) noexcept {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

inline float Int2Avx512BFloat16BitsToFloat(uint16_t b) noexcept {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

template <int RT, int BlockSize, bool WeightsPrepacked, bool ScalesBlockMajor, bool ScaleIsBf16>
void Int2Group8Avx512(const uint8_t* w_base, const uint8_t* next_w_base, const uint16_t* scales,
                      int64_t n_total, int64_t n0, int64_t k,
                      const float* a_reordered, float (&result)[8][RT]) {
  static_assert(RT == 1 || RT == 2, "Int2Group8Avx512 is specialized for RT=1,2");
  static_assert(BlockSize == 32 || BlockSize == 64 || BlockSize == 128,
                "Int2Group8Avx512 requires BlockSize 32, 64, or 128");
  constexpr int64_t kTileK = 64;
  const int64_t tiles64 = k / kTileK;
  const int64_t scale_blocks = k / BlockSize;
  const int64_t raw_w_stride = k / 4;
  const __m128i mask = _mm_set1_epi8(0x03);
  const __m128i two8 = _mm_set1_epi8(2);
  __m512 acc0_0 = _mm512_setzero_ps(), acc0_1 = _mm512_setzero_ps();
  __m512 acc1_0 = _mm512_setzero_ps(), acc1_1 = _mm512_setzero_ps();
  __m512 acc2_0 = _mm512_setzero_ps(), acc2_1 = _mm512_setzero_ps();
  __m512 acc3_0 = _mm512_setzero_ps(), acc3_1 = _mm512_setzero_ps();
  __m512 acc4_0 = _mm512_setzero_ps(), acc4_1 = _mm512_setzero_ps();
  __m512 acc5_0 = _mm512_setzero_ps(), acc5_1 = _mm512_setzero_ps();
  __m512 acc6_0 = _mm512_setzero_ps(), acc6_1 = _mm512_setzero_ps();
  __m512 acc7_0 = _mm512_setzero_ps(), acc7_1 = _mm512_setzero_ps();

  auto read_scale = [&](int c, int64_t sb) QMOE_LAMBDA_INLINE -> float {
    const uint16_t s_bits =
        ScalesBlockMajor ? scales[sb * n_total + n0 + c] : scales[(n0 + c) * scale_blocks + sb];
    return ScaleIsBf16 ? Int2Avx512BFloat16BitsToFloat(s_bits) : Int2Avx512HalfBitsToFloat(s_bits);
  };

  for (int64_t kb64 = 0; kb64 < tiles64; ++kb64) {
    const uint8_t* tile = WeightsPrepacked ? (w_base + kb64 * 128) : nullptr;
    if (next_w_base != nullptr) {
      if constexpr (WeightsPrepacked) {
        _mm_prefetch(reinterpret_cast<const char*>(next_w_base + kb64 * 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(next_w_base + kb64 * 128 + 64), _MM_HINT_T0);
      }
    }
    const __m512 a0_0 = _mm512_loadu_ps(a_reordered + 0 * k + kb64 * kTileK + 0);
    const __m512 a1_0 = _mm512_loadu_ps(a_reordered + 0 * k + kb64 * kTileK + 16);
    const __m512 a2_0 = _mm512_loadu_ps(a_reordered + 0 * k + kb64 * kTileK + 32);
    const __m512 a3_0 = _mm512_loadu_ps(a_reordered + 0 * k + kb64 * kTileK + 48);
    __m512 a0_1, a1_1, a2_1, a3_1;
    if constexpr (RT >= 2) {
      a0_1 = _mm512_loadu_ps(a_reordered + 1 * k + kb64 * kTileK + 0);
      a1_1 = _mm512_loadu_ps(a_reordered + 1 * k + kb64 * kTileK + 16);
      a2_1 = _mm512_loadu_ps(a_reordered + 1 * k + kb64 * kTileK + 32);
      a3_1 = _mm512_loadu_ps(a_reordered + 1 * k + kb64 * kTileK + 48);
    }
    auto step_col = [&](int c, const uint8_t* byte_ptr, __m512& acc_r0, __m512& acc_r1) QMOE_LAMBDA_INLINE {
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_ptr));
      const __m128i b0 = _mm_sub_epi8(_mm_and_si128(x, mask), two8);
      const __m128i b1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(x, 2), mask), two8);
      const __m128i b2 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(x, 4), mask), two8);
      const __m128i b3 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(x, 6), mask), two8);
      __m512 w0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b0));
      __m512 w1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b1));
      __m512 w2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b2));
      __m512 w3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b3));
      __m512 s;
      if constexpr (BlockSize == 32) {
        const __m512 s0 = _mm512_set1_ps(read_scale(c, 2 * kb64));
        const __m512 s1 = _mm512_set1_ps(read_scale(c, 2 * kb64 + 1));
        constexpr __mmask16 kSecondHalfMask = WeightsPrepacked ? 0xAAAA : 0xFF00;
        s = _mm512_mask_blend_ps(kSecondHalfMask, s0, s1);
      } else {
        const int64_t sb = (BlockSize == 64) ? kb64 : (kb64 >> 1);
        s = _mm512_set1_ps(read_scale(c, sb));
      }
      // Scale weights before products to avoid overflowing an unscaled dot product.
      w0 = _mm512_mul_ps(w0, s);
      w1 = _mm512_mul_ps(w1, s);
      w2 = _mm512_mul_ps(w2, s);
      w3 = _mm512_mul_ps(w3, s);
      const __m512 prod0 =
          _mm512_fmadd_ps(w0, a0_0, _mm512_fmadd_ps(w1, a1_0, _mm512_fmadd_ps(w2, a2_0, _mm512_mul_ps(w3, a3_0))));
      acc_r0 = _mm512_add_ps(prod0, acc_r0);
      if constexpr (RT >= 2) {
        const __m512 prod1 =
            _mm512_fmadd_ps(w0, a0_1, _mm512_fmadd_ps(w1, a1_1, _mm512_fmadd_ps(w2, a2_1, _mm512_mul_ps(w3, a3_1))));
        acc_r1 = _mm512_add_ps(prod1, acc_r1);
      }
    };
    step_col(0, WeightsPrepacked ? (tile + 0 * 16) : (w_base + 0 * raw_w_stride + kb64 * 16), acc0_0, acc0_1);
    step_col(1, WeightsPrepacked ? (tile + 1 * 16) : (w_base + 1 * raw_w_stride + kb64 * 16), acc1_0, acc1_1);
    step_col(2, WeightsPrepacked ? (tile + 2 * 16) : (w_base + 2 * raw_w_stride + kb64 * 16), acc2_0, acc2_1);
    step_col(3, WeightsPrepacked ? (tile + 3 * 16) : (w_base + 3 * raw_w_stride + kb64 * 16), acc3_0, acc3_1);
    step_col(4, WeightsPrepacked ? (tile + 4 * 16) : (w_base + 4 * raw_w_stride + kb64 * 16), acc4_0, acc4_1);
    step_col(5, WeightsPrepacked ? (tile + 5 * 16) : (w_base + 5 * raw_w_stride + kb64 * 16), acc5_0, acc5_1);
    step_col(6, WeightsPrepacked ? (tile + 6 * 16) : (w_base + 6 * raw_w_stride + kb64 * 16), acc6_0, acc6_1);
    step_col(7, WeightsPrepacked ? (tile + 7 * 16) : (w_base + 7 * raw_w_stride + kb64 * 16), acc7_0, acc7_1);
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

template <int RT, int BlockSize, bool WeightsPrepacked, bool ScalesBlockMajor, bool ScaleIsBf16>
void Int2Group4Avx512(const uint8_t* w_base, const uint8_t* next_w_base, const uint16_t* scales,
                      int64_t n_total, int64_t n0, int64_t c0_in_group, int64_t k,
                      const float* a_reordered, float (&result)[4][RT]) {
  static_assert(BlockSize == 32 || BlockSize == 64 || BlockSize == 128,
                "Int2Group4Avx512 requires BlockSize 32, 64, or 128");
  constexpr int64_t kTileK = 64;
  const int64_t tiles64 = k / kTileK;
  const int64_t scale_blocks = k / BlockSize;
  const int64_t raw_w_stride = k / 4;
  const __m128i mask = _mm_set1_epi8(0x03);
  const __m128i two8 = _mm_set1_epi8(2);
  __m512 acc[4][RT];
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < RT; ++r) {
      acc[c][r] = _mm512_setzero_ps();
    }
  }
  auto read_scale = [&](int c, int64_t sb) QMOE_LAMBDA_INLINE -> float {
    const uint16_t s_bits =
        ScalesBlockMajor ? scales[sb * n_total + n0 + c] : scales[(n0 + c) * scale_blocks + sb];
    return ScaleIsBf16 ? Int2Avx512BFloat16BitsToFloat(s_bits) : Int2Avx512HalfBitsToFloat(s_bits);
  };

  for (int64_t kb64 = 0; kb64 < tiles64; ++kb64) {
    const uint8_t* tile = WeightsPrepacked ? (w_base + kb64 * 128 + c0_in_group * 16) : nullptr;
    if (next_w_base != nullptr && (kb64 & 1) == 0) {
      if constexpr (WeightsPrepacked) {
        _mm_prefetch(reinterpret_cast<const char*>(next_w_base + kb64 * 128), _MM_HINT_T0);
      }
    }
    __m512 a0[RT], a1[RT], a2[RT], a3[RT];
    for (int r = 0; r < RT; ++r) {
      a0[r] = _mm512_loadu_ps(a_reordered + r * k + kb64 * kTileK + 0);
      a1[r] = _mm512_loadu_ps(a_reordered + r * k + kb64 * kTileK + 16);
      a2[r] = _mm512_loadu_ps(a_reordered + r * k + kb64 * kTileK + 32);
      a3[r] = _mm512_loadu_ps(a_reordered + r * k + kb64 * kTileK + 48);
    }
    auto step_col = [&](int c) QMOE_LAMBDA_INLINE {
      const uint8_t* byte_ptr = WeightsPrepacked ? (tile + c * 16) : (w_base + c * raw_w_stride + kb64 * 16);
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_ptr));
      const __m128i b0 = _mm_sub_epi8(_mm_and_si128(x, mask), two8);
      const __m128i b1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(x, 2), mask), two8);
      const __m128i b2 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(x, 4), mask), two8);
      const __m128i b3 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(x, 6), mask), two8);
      __m512 w0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b0));
      __m512 w1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b1));
      __m512 w2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b2));
      __m512 w3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b3));
      __m512 s;
      if constexpr (BlockSize == 32) {
        const __m512 s0 = _mm512_set1_ps(read_scale(c, 2 * kb64));
        const __m512 s1 = _mm512_set1_ps(read_scale(c, 2 * kb64 + 1));
        constexpr __mmask16 kSecondHalfMask = WeightsPrepacked ? 0xAAAA : 0xFF00;
        s = _mm512_mask_blend_ps(kSecondHalfMask, s0, s1);
      } else {
        const int64_t sb = (BlockSize == 64) ? kb64 : (kb64 >> 1);
        s = _mm512_set1_ps(read_scale(c, sb));
      }
      // Scale weights before products to avoid overflowing an unscaled dot product.
      w0 = _mm512_mul_ps(w0, s);
      w1 = _mm512_mul_ps(w1, s);
      w2 = _mm512_mul_ps(w2, s);
      w3 = _mm512_mul_ps(w3, s);
      for (int r = 0; r < RT; ++r) {
        const __m512 prod =
            _mm512_fmadd_ps(w0, a0[r], _mm512_fmadd_ps(w1, a1[r], _mm512_fmadd_ps(w2, a2[r], _mm512_mul_ps(w3, a3[r]))));
        acc[c][r] = _mm512_add_ps(prod, acc[c][r]);
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

template <int RT, int BlockSize>
void Int2RowTileAvx512(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                       bool scales_are_block_major, bool weights_are_prepacked,
                       bool scale_is_bf16, bool bias_is_bf16,
                       int64_t n_total, int64_t k, const float* a_reordered,
                       float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  if (n_begin >= n_end) return;
  constexpr int64_t kInt2ColsInterleaved = 8;
  if (weights_are_prepacked) {
    const int64_t aligned_begin = (n_begin + 3) & ~int64_t{3};
    const int64_t aligned_end = n_end & ~int64_t{3};
    if (aligned_begin >= aligned_end) {
      Int2GemmRange(weights, scales, bias, BlockSize, scales_are_block_major, weights_are_prepacked,
                    scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, RT,
                    out, out_stride, n_begin, n_end, /*force_scalar=*/true);
      return;
    }
    if (n_begin < aligned_begin) {
      Int2GemmRange(weights, scales, bias, BlockSize, scales_are_block_major, weights_are_prepacked,
                    scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, RT,
                    out, out_stride, n_begin, aligned_begin, /*force_scalar=*/true);
    }
    const int64_t group_stride = kInt2ColsInterleaved * (k / 4);
    int64_t n0 = aligned_begin;
    if constexpr (RT <= 2) {
      for (; n0 < aligned_end && (n0 % kInt2ColsInterleaved) != 0; n0 += 4) {
        const uint8_t* group = weights + (n0 / kInt2ColsInterleaved) * group_stride;
        float result[4][RT];
        if (scales_are_block_major) {
          if (scale_is_bf16) {
            Int2Group4Avx512<RT, BlockSize, true, true, true>(
                group, nullptr, scales, n_total, n0, n0 % kInt2ColsInterleaved, k, a_reordered, result);
          } else {
            Int2Group4Avx512<RT, BlockSize, true, true, false>(
                group, nullptr, scales, n_total, n0, n0 % kInt2ColsInterleaved, k, a_reordered, result);
          }
        } else {
          if (scale_is_bf16) {
            Int2Group4Avx512<RT, BlockSize, true, false, true>(
                group, nullptr, scales, n_total, n0, n0 % kInt2ColsInterleaved, k, a_reordered, result);
          } else {
            Int2Group4Avx512<RT, BlockSize, true, false, false>(
                group, nullptr, scales, n_total, n0, n0 % kInt2ColsInterleaved, k, a_reordered, result);
          }
        }
        for (int c = 0; c < 4; ++c) {
          const float b = (bias != nullptr)
                              ? (bias_is_bf16 ? Int2Avx512BFloat16BitsToFloat(bias[n0 + c]) : Int2Avx512HalfBitsToFloat(bias[n0 + c]))
                              : 0.0f;
          for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result[c][r] + b;
        }
      }
      for (; n0 + 8 <= aligned_end; n0 += 8) {
        const uint8_t* group = weights + (n0 / kInt2ColsInterleaved) * group_stride;
        const uint8_t* next_group = (n0 + 16 <= aligned_end) ? (group + group_stride) : nullptr;
        float result8[8][RT];
        if (scales_are_block_major) {
          if (scale_is_bf16) {
            Int2Group8Avx512<RT, BlockSize, true, true, true>(group, next_group, scales, n_total, n0, k, a_reordered, result8);
          } else {
            Int2Group8Avx512<RT, BlockSize, true, true, false>(group, next_group, scales, n_total, n0, k, a_reordered, result8);
          }
        } else {
          if (scale_is_bf16) {
            Int2Group8Avx512<RT, BlockSize, true, false, true>(group, next_group, scales, n_total, n0, k, a_reordered, result8);
          } else {
            Int2Group8Avx512<RT, BlockSize, true, false, false>(group, next_group, scales, n_total, n0, k, a_reordered, result8);
          }
        }
        for (int c = 0; c < 8; ++c) {
          const float b = (bias != nullptr)
                              ? (bias_is_bf16 ? Int2Avx512BFloat16BitsToFloat(bias[n0 + c]) : Int2Avx512HalfBitsToFloat(bias[n0 + c]))
                              : 0.0f;
          for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result8[c][r] + b;
        }
      }
    }
    for (; n0 < aligned_end; n0 += 4) {
      const uint8_t* group = weights + (n0 / kInt2ColsInterleaved) * group_stride;
      const uint8_t* next_group = (n0 + kInt2ColsInterleaved < aligned_end) ? (group + group_stride) : nullptr;
      float result[4][RT];
      if (scales_are_block_major) {
        if (scale_is_bf16) {
          Int2Group4Avx512<RT, BlockSize, true, true, true>(
              group, next_group, scales, n_total, n0, n0 % kInt2ColsInterleaved, k, a_reordered, result);
        } else {
          Int2Group4Avx512<RT, BlockSize, true, true, false>(
              group, next_group, scales, n_total, n0, n0 % kInt2ColsInterleaved, k, a_reordered, result);
        }
      } else {
        if (scale_is_bf16) {
          Int2Group4Avx512<RT, BlockSize, true, false, true>(
              group, next_group, scales, n_total, n0, n0 % kInt2ColsInterleaved, k, a_reordered, result);
        } else {
          Int2Group4Avx512<RT, BlockSize, true, false, false>(
              group, next_group, scales, n_total, n0, n0 % kInt2ColsInterleaved, k, a_reordered, result);
        }
      }
      for (int c = 0; c < 4; ++c) {
        const float b = (bias != nullptr)
                            ? (bias_is_bf16 ? Int2Avx512BFloat16BitsToFloat(bias[n0 + c]) : Int2Avx512HalfBitsToFloat(bias[n0 + c]))
                            : 0.0f;
        for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result[c][r] + b;
      }
    }
    if (aligned_end < n_end) {
      Int2GemmRange(weights, scales, bias, BlockSize, scales_are_block_major, weights_are_prepacked,
                    scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, RT,
                    out + (aligned_end - n_begin), out_stride, aligned_end, n_end, /*force_scalar=*/true);
    }
  } else {
    const int64_t raw_w_stride = k / 4;
    int64_t n0 = n_begin;
    for (; n0 + 4 <= n_end; n0 += 4) {
      const uint8_t* w_base = weights + n0 * raw_w_stride;
      const uint8_t* next_w = (n0 + 8 <= n_end) ? (weights + (n0 + 4) * raw_w_stride) : nullptr;
      float result[4][RT];
      if (scales_are_block_major) {
        if (scale_is_bf16) {
          Int2Group4Avx512<RT, BlockSize, false, true, true>(w_base, next_w, scales, n_total, n0, 0, k, a_reordered, result);
        } else {
          Int2Group4Avx512<RT, BlockSize, false, true, false>(w_base, next_w, scales, n_total, n0, 0, k, a_reordered, result);
        }
      } else {
        if (scale_is_bf16) {
          Int2Group4Avx512<RT, BlockSize, false, false, true>(w_base, next_w, scales, n_total, n0, 0, k, a_reordered, result);
        } else {
          Int2Group4Avx512<RT, BlockSize, false, false, false>(w_base, next_w, scales, n_total, n0, 0, k, a_reordered, result);
        }
      }
      for (int c = 0; c < 4; ++c) {
        const float b = (bias != nullptr)
                            ? (bias_is_bf16 ? Int2Avx512BFloat16BitsToFloat(bias[n0 + c]) : Int2Avx512HalfBitsToFloat(bias[n0 + c]))
                            : 0.0f;
        for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result[c][r] + b;
      }
    }
    if (n0 < n_end) {
      Int2GemmRange(weights, scales, bias, BlockSize, scales_are_block_major, weights_are_prepacked,
                    scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, RT,
                    out + (n0 - n_begin), out_stride, n0, n_end, /*force_scalar=*/true);
    }
  }
}

template <int BlockSize>
void Avx512Int2GemmRangeByBlock(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
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
      Int2RowTileAvx512<4, BlockSize>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                      scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 4;
    } else if (left == 3) {
      Int2RowTileAvx512<3, BlockSize>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                      scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 3;
    } else if (left == 2) {
      Int2RowTileAvx512<2, BlockSize>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                      scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 2;
    } else {
      Int2RowTileAvx512<1, BlockSize>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                      scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 1;
    }
  }
}

void Avx512Int2GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                         int64_t block_size, bool scales_are_block_major, bool weights_are_prepacked,
                         bool scale_is_bf16, bool bias_is_bf16,
                         int64_t n_total, int64_t k, const float* a_reordered, int64_t m,
                         float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  if (block_size == 64) {
    Avx512Int2GemmRangeByBlock<64>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                   scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, m,
                                   out, out_stride, n_begin, n_end);
  } else if (block_size == 32) {
    Avx512Int2GemmRangeByBlock<32>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                   scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, m,
                                   out, out_stride, n_begin, n_end);
  } else {
    Avx512Int2GemmRangeByBlock<128>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                    scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, m,
                                    out, out_stride, n_begin, n_end);
  }
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
