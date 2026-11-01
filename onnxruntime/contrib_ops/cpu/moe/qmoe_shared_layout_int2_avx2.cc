// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AVX2 + FMA + F16C CPU INT2 expert kernels for QMoE.

#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS 1
#pragma warning(disable : 4996 4556 4068 4100)
#endif

#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"

#include <algorithm>
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

inline float Int2Avx2HalfBitsToFloat(uint16_t h) noexcept {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

inline float Int2Avx2BFloat16BitsToFloat(uint16_t b) noexcept {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline float Int2Avx2ReduceAddPs(__m256 v) noexcept {
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 sum128 = _mm_add_ps(lo, hi);
  const __m128 shuf = _mm_movehdup_ps(sum128);
  const __m128 sums = _mm_add_ps(sum128, shuf);
  return _mm_cvtss_f32(_mm_add_ss(sums, _mm_movehl_ps(shuf, sums)));
}

template <int RT, int BlockSize, bool WeightsPrepacked, bool ScalesBlockMajor, bool ScaleIsBf16>
void Int2Group4Avx2(const uint8_t* w_base, const uint8_t* next_w_base, const uint16_t* scales,
                    int64_t n_total, int64_t n0, int64_t c0_in_group, int64_t k,
                    const float* a_reordered, float (&result)[4][RT]) {
  static_assert(BlockSize == 32 || BlockSize == 64 || BlockSize == 128,
                "Int2Group4Avx2 requires BlockSize 32, 64, or 128");
  constexpr int64_t kTileK = 64;
  const int64_t tiles64 = k / kTileK;
  const int64_t scale_blocks = k / BlockSize;
  const int64_t raw_w_stride = k / 4;
  const __m128i mask = _mm_set1_epi8(0x03);
  const __m128i two8 = _mm_set1_epi8(2);
  __m256 acc[4][RT];
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < RT; ++r) {
      acc[c][r] = _mm256_setzero_ps();
    }
  }
  auto read_scale = [&](int c, int64_t sb) QMOE_AVX2_LAMBDA_INLINE -> float {
    const uint16_t s_bits =
        ScalesBlockMajor ? scales[sb * n_total + n0 + c] : scales[(n0 + c) * scale_blocks + sb];
    return ScaleIsBf16 ? Int2Avx2BFloat16BitsToFloat(s_bits) : Int2Avx2HalfBitsToFloat(s_bits);
  };

  for (int64_t kb64 = 0; kb64 < tiles64; ++kb64) {
    const uint8_t* tile = WeightsPrepacked ? (w_base + kb64 * 128 + c0_in_group * 16) : nullptr;
    if (next_w_base != nullptr && (kb64 & 1) == 0) {
      if constexpr (WeightsPrepacked) {
        _mm_prefetch(reinterpret_cast<const char*>(next_w_base + kb64 * 128), _MM_HINT_T0);
      }
    }
    __m256 a_lo[4][RT], a_hi[4][RT];
    for (int s = 0; s < 4; ++s) {
      for (int r = 0; r < RT; ++r) {
        a_lo[s][r] = _mm256_loadu_ps(a_reordered + r * k + kb64 * kTileK + s * 16 + 0);
        a_hi[s][r] = _mm256_loadu_ps(a_reordered + r * k + kb64 * kTileK + s * 16 + 8);
      }
    }
    auto step_col = [&](int c) QMOE_AVX2_LAMBDA_INLINE {
      const uint8_t* byte_ptr = WeightsPrepacked ? (tile + c * 16) : (w_base + c * raw_w_stride + kb64 * 16);
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_ptr));
      const __m128i b0 = _mm_sub_epi8(_mm_and_si128(x, mask), two8);
      const __m128i b1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(x, 2), mask), two8);
      const __m128i b2 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(x, 4), mask), two8);
      const __m128i b3 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(x, 6), mask), two8);
      __m256 w0_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0));
      __m256 w0_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(b0, 8)));
      __m256 w1_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1));
      __m256 w1_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(b1, 8)));
      __m256 w2_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b2));
      __m256 w2_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(b2, 8)));
      __m256 w3_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b3));
      __m256 w3_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(b3, 8)));
      __m256 s_lo, s_hi;
      if constexpr (BlockSize == 32) {
        const __m256 s0 = _mm256_set1_ps(read_scale(c, 2 * kb64));
        const __m256 s1 = _mm256_set1_ps(read_scale(c, 2 * kb64 + 1));
        if constexpr (WeightsPrepacked) {
          s_lo = _mm256_blend_ps(s0, s1, 0xAA);
          s_hi = s_lo;
        } else {
          s_lo = s0;
          s_hi = s1;
        }
      } else {
        const int64_t sb = (BlockSize == 64) ? kb64 : (kb64 >> 1);
        s_lo = _mm256_set1_ps(read_scale(c, sb));
        s_hi = s_lo;
      }
      // Scale weights before products to avoid overflowing an unscaled dot product.
      w0_lo = _mm256_mul_ps(w0_lo, s_lo);
      w0_hi = _mm256_mul_ps(w0_hi, s_hi);
      w1_lo = _mm256_mul_ps(w1_lo, s_lo);
      w1_hi = _mm256_mul_ps(w1_hi, s_hi);
      w2_lo = _mm256_mul_ps(w2_lo, s_lo);
      w2_hi = _mm256_mul_ps(w2_hi, s_hi);
      w3_lo = _mm256_mul_ps(w3_lo, s_lo);
      w3_hi = _mm256_mul_ps(w3_hi, s_hi);
      for (int r = 0; r < RT; ++r) {
        const __m256 prod_lo =
            _mm256_fmadd_ps(w0_lo, a_lo[0][r],
                            _mm256_fmadd_ps(w1_lo, a_lo[1][r],
                                            _mm256_fmadd_ps(w2_lo, a_lo[2][r], _mm256_mul_ps(w3_lo, a_lo[3][r]))));
        const __m256 prod_hi =
            _mm256_fmadd_ps(w0_hi, a_hi[0][r],
                            _mm256_fmadd_ps(w1_hi, a_hi[1][r],
                                            _mm256_fmadd_ps(w2_hi, a_hi[2][r], _mm256_mul_ps(w3_hi, a_hi[3][r]))));
        acc[c][r] = _mm256_add_ps(prod_lo, _mm256_add_ps(prod_hi, acc[c][r]));
      }
    };
    step_col(0);
    step_col(1);
    step_col(2);
    step_col(3);
  }
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < RT; ++r) {
      result[c][r] = Int2Avx2ReduceAddPs(acc[c][r]);
    }
  }
}

template <int RT, int BlockSize>
void Int2RowTileAvx2(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
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
    for (int64_t n0 = aligned_begin; n0 < aligned_end; n0 += 4) {
      const uint8_t* group = weights + (n0 / kInt2ColsInterleaved) * group_stride;
      const uint8_t* next_group = (n0 + kInt2ColsInterleaved < aligned_end) ? (group + group_stride) : nullptr;
      float result[4][RT];
      if (scales_are_block_major) {
        if (scale_is_bf16) {
          Int2Group4Avx2<RT, BlockSize, true, true, true>(
              group, next_group, scales, n_total, n0, n0 % kInt2ColsInterleaved, k, a_reordered, result);
        } else {
          Int2Group4Avx2<RT, BlockSize, true, true, false>(
              group, next_group, scales, n_total, n0, n0 % kInt2ColsInterleaved, k, a_reordered, result);
        }
      } else {
        if (scale_is_bf16) {
          Int2Group4Avx2<RT, BlockSize, true, false, true>(
              group, next_group, scales, n_total, n0, n0 % kInt2ColsInterleaved, k, a_reordered, result);
        } else {
          Int2Group4Avx2<RT, BlockSize, true, false, false>(
              group, next_group, scales, n_total, n0, n0 % kInt2ColsInterleaved, k, a_reordered, result);
        }
      }
      for (int c = 0; c < 4; ++c) {
        const float b = (bias != nullptr)
                            ? (bias_is_bf16 ? Int2Avx2BFloat16BitsToFloat(bias[n0 + c]) : Int2Avx2HalfBitsToFloat(bias[n0 + c]))
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
          Int2Group4Avx2<RT, BlockSize, false, true, true>(w_base, next_w, scales, n_total, n0, 0, k, a_reordered, result);
        } else {
          Int2Group4Avx2<RT, BlockSize, false, true, false>(w_base, next_w, scales, n_total, n0, 0, k, a_reordered, result);
        }
      } else {
        if (scale_is_bf16) {
          Int2Group4Avx2<RT, BlockSize, false, false, true>(w_base, next_w, scales, n_total, n0, 0, k, a_reordered, result);
        } else {
          Int2Group4Avx2<RT, BlockSize, false, false, false>(w_base, next_w, scales, n_total, n0, 0, k, a_reordered, result);
        }
      }
      for (int c = 0; c < 4; ++c) {
        const float b = (bias != nullptr)
                            ? (bias_is_bf16 ? Int2Avx2BFloat16BitsToFloat(bias[n0 + c]) : Int2Avx2HalfBitsToFloat(bias[n0 + c]))
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
void Avx2Int2GemmRangeByBlock(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
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
      Int2RowTileAvx2<4, BlockSize>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                    scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 4;
    } else if (left == 3) {
      Int2RowTileAvx2<3, BlockSize>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                    scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 3;
    } else if (left == 2) {
      Int2RowTileAvx2<2, BlockSize>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                    scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 2;
    } else {
      Int2RowTileAvx2<1, BlockSize>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                    scale_is_bf16, bias_is_bf16, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 1;
    }
  }
}

void Avx2Int2GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                       int64_t block_size, bool scales_are_block_major, bool weights_are_prepacked,
                       bool scale_is_bf16, bool bias_is_bf16,
                       int64_t n_total, int64_t k, const float* a_reordered, int64_t m,
                       float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  if (block_size == 64) {
    Avx2Int2GemmRangeByBlock<64>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                 scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, m,
                                 out, out_stride, n_begin, n_end);
  } else if (block_size == 32) {
    Avx2Int2GemmRangeByBlock<32>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                 scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, m,
                                 out, out_stride, n_begin, n_end);
  } else {
    Avx2Int2GemmRangeByBlock<128>(weights, scales, bias, scales_are_block_major, weights_are_prepacked,
                                  scale_is_bf16, bias_is_bf16, n_total, k, a_reordered, m,
                                  out, out_stride, n_begin, n_end);
  }
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
