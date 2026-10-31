// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AVX2 + FMA + F16C CPU INT3 expert kernels for QMoE (row-local LSB-first 3-bit codes).

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

namespace {

inline float Avx2Int3HalfBitsToFloat(uint16_t h) noexcept {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

inline float Avx2Int3BFloat16BitsToFloat(uint16_t b) noexcept {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline float Avx2Int3ReduceAddPs(__m256 v) noexcept {
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 sum128 = _mm_add_ps(lo, hi);
  const __m128 shuf = _mm_movehdup_ps(sum128);
  const __m128 sums = _mm_add_ps(sum128, shuf);
  return _mm_cvtss_f32(_mm_add_ss(sums, _mm_movehl_ps(shuf, sums)));
}

inline __m256 Unpack8Int3WeightsAvx2(
    uint32_t u24, const __m256i& shifts, const __m256i& mask7, const __m256i& four) noexcept {
  return _mm256_cvtepi32_ps(
      _mm256_sub_epi32(_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(u24)), shifts), mask7), four));
}

inline __m256 Int3LookupAvx2(float scale, bool bf16) {
  __m256 values = _mm256_mul_ps(_mm256_setr_ps(-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f),
                                _mm256_set1_ps(scale));
  if (bf16) {
    const __m256i bits = _mm256_castps_si256(values);
    const __m256i exponent = _mm256_and_si256(bits, _mm256_set1_epi32(0x7f800000));
    const __m256i nonfinite = _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7f800000));
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    const __m256i rounded = _mm256_and_si256(_mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb)),
                                             _mm256_set1_epi32(static_cast<int>(0xffff0000u)));
    const __m256 nan_mask = _mm256_cmp_ps(values, values, _CMP_UNORD_Q);
    values = _mm256_castsi256_ps(_mm256_blendv_epi8(rounded, bits, nonfinite));
    values = _mm256_blendv_ps(values, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000)), nan_mask);
  } else {
    values = _mm256_cvtph_ps(_mm256_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  return values;
}

template <bool DeferredScale>
inline __m256 DequantInt3Avx2(__m256 codes, __m256 table, __m256i four) {
  if constexpr (DeferredScale) return codes;
  return _mm256_permutevar8x32_ps(table, _mm256_add_epi32(_mm256_cvtps_epi32(codes), four));
}

template <bool DeferredScale>
inline __m256 Dot32Int3Avx2(
    const uint8_t* p, const __m256& a0, const __m256& a1, const __m256& a2, const __m256& a3,
    const __m256i& shifts, const __m256i& mask7, const __m256i& four, const __m256& table) noexcept {
  uint32_t u0 = 0, u1 = 0, u2 = 0, u3 = 0;
  std::memcpy(&u0, p + 0, sizeof(uint32_t));
  std::memcpy(&u1, p + 3, sizeof(uint32_t));
  std::memcpy(&u2, p + 6, sizeof(uint32_t));
  std::memcpy(&u3, p + 8, sizeof(uint32_t));
  u3 >>= 8;
  const __m256 w0 = DequantInt3Avx2<DeferredScale>(Unpack8Int3WeightsAvx2(u0, shifts, mask7, four), table, four);
  const __m256 w1 = DequantInt3Avx2<DeferredScale>(Unpack8Int3WeightsAvx2(u1, shifts, mask7, four), table, four);
  const __m256 p01 = _mm256_fmadd_ps(w0, a0, _mm256_mul_ps(w1, a1));
  const __m256 w2 = DequantInt3Avx2<DeferredScale>(Unpack8Int3WeightsAvx2(u2, shifts, mask7, four), table, four);
  const __m256 w3 = DequantInt3Avx2<DeferredScale>(Unpack8Int3WeightsAvx2(u3, shifts, mask7, four), table, four);
  const __m256 p23 = _mm256_fmadd_ps(w2, a2, _mm256_mul_ps(w3, a3));
  return _mm256_add_ps(p01, p23);
}

template <int RT, int BlockSize, bool ScaleIsBf16, bool DeferredScale>
void Int3Group4Avx2(const uint8_t* w_base, const uint8_t* next_w_base, const uint16_t* scales,
                    int64_t n0, int64_t k, const float* a_natural, float (&result)[4][RT]) noexcept;

template <int BlockSize, bool ScaleIsBf16, bool DeferredScale>
void Int3Group8Avx2(const uint8_t* w_base, const uint8_t* next_w_base, const uint16_t* scales,
                    int64_t n0, int64_t k, const float* a_natural, float (&result)[8][1]) noexcept {
  float first[4][1], second[4][1];
  Int3Group4Avx2<1, BlockSize, ScaleIsBf16, DeferredScale>(w_base, next_w_base, scales, n0, k, a_natural, first);
  Int3Group4Avx2<1, BlockSize, ScaleIsBf16, DeferredScale>(w_base + 4 * k * 3 / 8,
                                                           next_w_base ? next_w_base + 4 * k * 3 / 8 : nullptr,
                                                           scales, n0 + 4, k, a_natural, second);
  for (int c = 0; c < 4; ++c) {
    result[c][0] = first[c][0];
    result[c + 4][0] = second[c][0];
  }
}

template <int RT, int BlockSize, bool ScaleIsBf16, bool DeferredScale>
void Int3Group4Avx2(const uint8_t* w_base, const uint8_t* next_w_base, const uint16_t* scales,
                    int64_t n0, int64_t k, const float* a_natural, float (&result)[4][RT]) noexcept {
  static_assert(BlockSize == 32 || BlockSize == 64 || BlockSize == 128,
                "Int3Group4Avx2 requires BlockSize 32, 64, or 128");
  constexpr int kChunks32 = BlockSize / 32;
  const int64_t blocks = k / BlockSize;
  const int64_t raw_w_stride = (k * 3) / 8;
  constexpr int64_t kBlockBytes = (BlockSize * 3) / 8;
  const __m256i shifts = _mm256_set_epi32(21, 18, 15, 12, 9, 6, 3, 0);
  const __m256i mask7 = _mm256_set1_epi32(7);
  const __m256i four = _mm256_set1_epi32(4);
  __m256 acc0_0 = _mm256_setzero_ps(), acc0_1 = _mm256_setzero_ps();
  __m256 acc1_0 = _mm256_setzero_ps(), acc1_1 = _mm256_setzero_ps();
  __m256 acc2_0 = _mm256_setzero_ps(), acc2_1 = _mm256_setzero_ps();
  __m256 acc3_0 = _mm256_setzero_ps(), acc3_1 = _mm256_setzero_ps();

  for (int64_t kb = 0; kb < blocks; ++kb) {
    const int64_t byte_off = kb * kBlockBytes;
    if (next_w_base != nullptr && (kb & 3) == 0) {
      for (int c = 0; c < 4; ++c) {
        _mm_prefetch(reinterpret_cast<const char*>(next_w_base + c * raw_w_stride + byte_off), _MM_HINT_T0);
      }
    }
    auto step_col = [&](int c, __m256& acc_r0, __m256& acc_r1) QMOE_AVX2_LAMBDA_INLINE {
      const uint8_t* p = w_base + c * raw_w_stride + byte_off;
      const float s_val = ScaleIsBf16 ? Avx2Int3BFloat16BitsToFloat(scales[(n0 + c) * blocks + kb])
                                      : Avx2Int3HalfBitsToFloat(scales[(n0 + c) * blocks + kb]);
      const __m256 table = DeferredScale ? _mm256_set1_ps(s_val) : Int3LookupAvx2(s_val, ScaleIsBf16);
      __m256 prod0 = _mm256_setzero_ps();
      __m256 prod1 = _mm256_setzero_ps();
      for (int ch = 0; ch < kChunks32; ++ch) {
        const __m256 a0_0 = _mm256_loadu_ps(a_natural + 0 * k + kb * BlockSize + ch * 32 + 0);
        const __m256 a1_0 = _mm256_loadu_ps(a_natural + 0 * k + kb * BlockSize + ch * 32 + 8);
        const __m256 a2_0 = _mm256_loadu_ps(a_natural + 0 * k + kb * BlockSize + ch * 32 + 16);
        const __m256 a3_0 = _mm256_loadu_ps(a_natural + 0 * k + kb * BlockSize + ch * 32 + 24);
        const __m256 d0 = Dot32Int3Avx2<DeferredScale>(p + ch * 12, a0_0, a1_0, a2_0, a3_0, shifts, mask7, four, table);
        prod0 = (ch == 0) ? d0 : _mm256_add_ps(prod0, d0);
        if constexpr (RT >= 2) {
          const __m256 a0_1 = _mm256_loadu_ps(a_natural + 1 * k + kb * BlockSize + ch * 32 + 0);
          const __m256 a1_1 = _mm256_loadu_ps(a_natural + 1 * k + kb * BlockSize + ch * 32 + 8);
          const __m256 a2_1 = _mm256_loadu_ps(a_natural + 1 * k + kb * BlockSize + ch * 32 + 16);
          const __m256 a3_1 = _mm256_loadu_ps(a_natural + 1 * k + kb * BlockSize + ch * 32 + 24);
          const __m256 d1 =
              Dot32Int3Avx2<DeferredScale>(p + ch * 12, a0_1, a1_1, a2_1, a3_1, shifts, mask7, four, table);
          prod1 = (ch == 0) ? d1 : _mm256_add_ps(prod1, d1);
        }
      }
      acc_r0 = DeferredScale ? _mm256_fmadd_ps(prod0, table, acc_r0) : _mm256_add_ps(prod0, acc_r0);
      if constexpr (RT >= 2) {
        acc_r1 = DeferredScale ? _mm256_fmadd_ps(prod1, table, acc_r1) : _mm256_add_ps(prod1, acc_r1);
      } else {
        (void)acc_r1;
      }
    };
    step_col(0, acc0_0, acc0_1);
    step_col(1, acc1_0, acc1_1);
    step_col(2, acc2_0, acc2_1);
    step_col(3, acc3_0, acc3_1);
  }
  result[0][0] = Avx2Int3ReduceAddPs(acc0_0);
  result[1][0] = Avx2Int3ReduceAddPs(acc1_0);
  result[2][0] = Avx2Int3ReduceAddPs(acc2_0);
  result[3][0] = Avx2Int3ReduceAddPs(acc3_0);
  if constexpr (RT >= 2) {
    result[0][1] = Avx2Int3ReduceAddPs(acc0_1);
    result[1][1] = Avx2Int3ReduceAddPs(acc1_1);
    result[2][1] = Avx2Int3ReduceAddPs(acc2_1);
    result[3][1] = Avx2Int3ReduceAddPs(acc3_1);
  }
}

template <int RT, bool DeferredScale>
void Int3RowTileAvx2(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                     int64_t block_size, bool scale_is_bf16, bool bias_is_bf16,
                     int64_t n_total, int64_t k, const float* a_natural,
                     float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  if (n_begin >= n_end) return;
  if ((block_size != 32 && block_size != 64 && block_size != 128) || (k % block_size) != 0) {
    Int3GemmRange(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16,
                  n_total, k, a_natural, RT, out, out_stride, n_begin, n_end, /*force_scalar=*/true, DeferredScale);
    return;
  }
  const int64_t raw_w_stride = (k * 3) / 8;
  int64_t n0 = n_begin;
  if constexpr (RT == 1) {
    if (block_size == 32 || block_size == 64) {
      for (; n0 + 8 <= n_end; n0 += 8) {
        const uint8_t* w_base = weights + n0 * raw_w_stride;
        const uint8_t* next_w = (n0 + 16 <= n_end) ? (weights + (n0 + 8) * raw_w_stride) : nullptr;
        float result8[8][1];
        if (block_size == 32) {
          if (scale_is_bf16) {
            Int3Group8Avx2<32, true, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result8);
          } else {
            Int3Group8Avx2<32, false, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result8);
          }
        } else {
          if (scale_is_bf16) {
            Int3Group8Avx2<64, true, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result8);
          } else {
            Int3Group8Avx2<64, false, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result8);
          }
        }
        for (int c = 0; c < 8; ++c) {
          const float b = (bias != nullptr)
                              ? (bias_is_bf16 ? Avx2Int3BFloat16BitsToFloat(bias[n0 + c]) : Avx2Int3HalfBitsToFloat(bias[n0 + c]))
                              : 0.0f;
          out[n0 + c - n_begin] = result8[c][0] + b;
        }
      }
    }
  }
  for (; n0 + 4 <= n_end; n0 += 4) {
    const uint8_t* w_base = weights + n0 * raw_w_stride;
    const uint8_t* next_w = (n0 + 8 <= n_end) ? (weights + (n0 + 4) * raw_w_stride) : nullptr;
    float result[4][RT];
    if (block_size == 32) {
      if (scale_is_bf16) {
        Int3Group4Avx2<RT, 32, true, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result);
      } else {
        Int3Group4Avx2<RT, 32, false, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result);
      }
    } else if (block_size == 64) {
      if (scale_is_bf16) {
        Int3Group4Avx2<RT, 64, true, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result);
      } else {
        Int3Group4Avx2<RT, 64, false, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result);
      }
    } else {
      if (scale_is_bf16) {
        Int3Group4Avx2<RT, 128, true, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result);
      } else {
        Int3Group4Avx2<RT, 128, false, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result);
      }
    }
    for (int c = 0; c < 4; ++c) {
      const float b = (bias != nullptr)
                          ? (bias_is_bf16 ? Avx2Int3BFloat16BitsToFloat(bias[n0 + c]) : Avx2Int3HalfBitsToFloat(bias[n0 + c]))
                          : 0.0f;
      for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result[c][r] + b;
    }
  }
  if (n0 < n_end) {
    Int3GemmRange(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16, n_total, k, a_natural, RT,
                  out + (n0 - n_begin), out_stride, n0, n_end, /*force_scalar=*/true, DeferredScale);
  }
}

}  // namespace

template <bool DeferredScale>
void Avx2Int3GemmRangeImpl(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                           int64_t block_size, bool scale_is_bf16, bool bias_is_bf16,
                           int64_t n_total, int64_t k, const float* a_natural, int64_t m,
                           float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  int64_t r = 0;
  while (r + 2 <= m) {
    Int3RowTileAvx2<2, DeferredScale>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16,
                                      n_total, k, a_natural + r * k, out + r * out_stride, out_stride, n_begin, n_end);
    r += 2;
  }
  if (r < m) {
    Int3RowTileAvx2<1, DeferredScale>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16,
                                      n_total, k, a_natural + r * k, out + r * out_stride, out_stride, n_begin, n_end);
  }
}

void Avx2Int3GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                       int64_t block_size, bool scale_is_bf16, bool bias_is_bf16,
                       int64_t n_total, int64_t k, const float* a_natural, int64_t m,
                       float* out, int64_t out_stride, int64_t n_begin, int64_t n_end, bool deferred_scale) {
  if (deferred_scale) {
    Avx2Int3GemmRangeImpl<true>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16, n_total, k, a_natural,
                                m, out, out_stride, n_begin, n_end);
  } else {
    Avx2Int3GemmRangeImpl<false>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16, n_total, k, a_natural,
                                 m, out, out_stride, n_begin, n_end);
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
