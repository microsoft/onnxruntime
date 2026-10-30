// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AVX-512 CPU INT3 expert kernels for QMoE (row-local LSB-first 3-bit codes, block_size 32, 64, 128).

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

namespace {

inline float Avx512Int3HalfBitsToFloat(uint16_t h) noexcept {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

inline float Avx512Int3BFloat16BitsToFloat(uint16_t b) noexcept {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline __m512 Unpack16Int3Avx512(
    const uint8_t* byte_ptr, const __m512i& shifts, const __m512i& mask7, const __m512i& four) noexcept {
  uint32_t lo24 = 0;
  uint32_t hi24 = 0;
  std::memcpy(&lo24, byte_ptr, sizeof(uint32_t));
  std::memcpy(&hi24, byte_ptr + 2, sizeof(uint32_t));
  hi24 >>= 8;
  const __m512i packed = _mm512_inserti64x4(
      _mm512_castsi256_si512(_mm256_set1_epi32(static_cast<int>(lo24))),
      _mm256_set1_epi32(static_cast<int>(hi24)), 1);
  const __m512i codes = _mm512_sub_epi32(_mm512_and_si512(_mm512_srlv_epi32(packed, shifts), mask7), four);
  return _mm512_cvtepi32_ps(codes);
}

inline __m512 Int3LookupAvx512(float scale, bool bf16) {
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
  return _mm512_broadcast_f32x8(values);
}

template <bool DeferredScale>
inline __m512 DequantInt3Avx512(__m512 codes, __m512 table) {
  if constexpr (DeferredScale) return codes;
  return _mm512_permutexvar_ps(_mm512_add_epi32(_mm512_cvtps_epi32(codes), _mm512_set1_epi32(4)), table);
}

template <int RT, int BlockSize, bool ScaleIsBf16, bool DeferredScale>
void Int3Group8Avx512(const uint8_t* w_base, const uint8_t* next_w_base, const uint16_t* scales,
                      int64_t n0, int64_t k, const float* a_natural, float (&result)[8][RT]) {
  static_assert(RT == 1 || RT == 2, "Int3Group8Avx512 is specialized for RT=1,2");
  static_assert(BlockSize == 32 || BlockSize == 64, "Int3Group8Avx512 requires BlockSize 32 or 64");
  const int64_t blocks = k / BlockSize;
  const int64_t raw_w_stride = (k * 3) / 8;
  constexpr int64_t kBlockBytes = (BlockSize * 3) / 8;
  const __m512i shifts = _mm512_set_epi32(21, 18, 15, 12, 9, 6, 3, 0, 21, 18, 15, 12, 9, 6, 3, 0);
  const __m512i mask7 = _mm512_set1_epi32(7);
  const __m512i four = _mm512_set1_epi32(4);
  __m512 acc0_0 = _mm512_setzero_ps(), acc0_1 = _mm512_setzero_ps();
  __m512 acc1_0 = _mm512_setzero_ps(), acc1_1 = _mm512_setzero_ps();
  __m512 acc2_0 = _mm512_setzero_ps(), acc2_1 = _mm512_setzero_ps();
  __m512 acc3_0 = _mm512_setzero_ps(), acc3_1 = _mm512_setzero_ps();
  __m512 acc4_0 = _mm512_setzero_ps(), acc4_1 = _mm512_setzero_ps();
  __m512 acc5_0 = _mm512_setzero_ps(), acc5_1 = _mm512_setzero_ps();
  __m512 acc6_0 = _mm512_setzero_ps(), acc6_1 = _mm512_setzero_ps();
  __m512 acc7_0 = _mm512_setzero_ps(), acc7_1 = _mm512_setzero_ps();

  for (int64_t kb = 0; kb < blocks; ++kb) {
    const int64_t byte_off = kb * kBlockBytes;
    if (next_w_base != nullptr) {
      const int c_base = static_cast<int>((kb & 1) * 4);
      for (int c = c_base; c < c_base + 4; ++c) {
        _mm_prefetch(reinterpret_cast<const char*>(next_w_base + c * raw_w_stride + byte_off), _MM_HINT_T0);
      }
    }
    const __m512 a0_0 = _mm512_loadu_ps(a_natural + 0 * k + kb * BlockSize);
    const __m512 a1_0 = _mm512_loadu_ps(a_natural + 0 * k + kb * BlockSize + 16);
    __m512 a2_0, a3_0;
    if constexpr (BlockSize == 64) {
      a2_0 = _mm512_loadu_ps(a_natural + 0 * k + kb * BlockSize + 32);
      a3_0 = _mm512_loadu_ps(a_natural + 0 * k + kb * BlockSize + 48);
    }
    __m512 a0_1, a1_1, a2_1, a3_1;
    if constexpr (RT >= 2) {
      a0_1 = _mm512_loadu_ps(a_natural + 1 * k + kb * BlockSize);
      a1_1 = _mm512_loadu_ps(a_natural + 1 * k + kb * BlockSize + 16);
      if constexpr (BlockSize == 64) {
        a2_1 = _mm512_loadu_ps(a_natural + 1 * k + kb * BlockSize + 32);
        a3_1 = _mm512_loadu_ps(a_natural + 1 * k + kb * BlockSize + 48);
      }
    }
    auto step_col = [&](int c, __m512& acc_r0, __m512& acc_r1) QMOE_LAMBDA_INLINE {
      const uint8_t* p = w_base + c * raw_w_stride + byte_off;
      const uint16_t s_bits = scales[(n0 + c) * blocks + kb];
      const float s_val = ScaleIsBf16 ? Avx512Int3BFloat16BitsToFloat(s_bits) : Avx512Int3HalfBitsToFloat(s_bits);
      const __m512 table = DeferredScale ? _mm512_set1_ps(s_val) : Int3LookupAvx512(s_val, ScaleIsBf16);
      const __m512 w0 = DequantInt3Avx512<DeferredScale>(Unpack16Int3Avx512(p + 0, shifts, mask7, four), table);
      const __m512 w1 = DequantInt3Avx512<DeferredScale>(Unpack16Int3Avx512(p + 6, shifts, mask7, four), table);
      __m512 prod0 = _mm512_fmadd_ps(w0, a0_0, _mm512_mul_ps(w1, a1_0));
      __m512 prod1;
      if constexpr (RT >= 2) {
        prod1 = _mm512_fmadd_ps(w0, a0_1, _mm512_mul_ps(w1, a1_1));
      }
      if constexpr (BlockSize == 64) {
        const __m512 w2 = DequantInt3Avx512<DeferredScale>(Unpack16Int3Avx512(p + 12, shifts, mask7, four), table);
        const __m512 w3 = DequantInt3Avx512<DeferredScale>(Unpack16Int3Avx512(p + 18, shifts, mask7, four), table);
        prod0 = _mm512_fmadd_ps(w2, a2_0, _mm512_fmadd_ps(w3, a3_0, prod0));
        if constexpr (RT >= 2) {
          prod1 = _mm512_fmadd_ps(w2, a2_1, _mm512_fmadd_ps(w3, a3_1, prod1));
        }
      }
      acc_r0 = DeferredScale ? _mm512_fmadd_ps(prod0, table, acc_r0) : _mm512_add_ps(prod0, acc_r0);
      if constexpr (RT >= 2) {
        acc_r1 = DeferredScale ? _mm512_fmadd_ps(prod1, table, acc_r1) : _mm512_add_ps(prod1, acc_r1);
      }
    };
    step_col(0, acc0_0, acc0_1);
    step_col(1, acc1_0, acc1_1);
    step_col(2, acc2_0, acc2_1);
    step_col(3, acc3_0, acc3_1);
    step_col(4, acc4_0, acc4_1);
    step_col(5, acc5_0, acc5_1);
    step_col(6, acc6_0, acc6_1);
    step_col(7, acc7_0, acc7_1);
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

template <int RT, int BlockSize, bool ScaleIsBf16, bool DeferredScale>
void Int3Group4Avx512(const uint8_t* w_base, const uint8_t* next_w_base, const uint16_t* scales,
                      int64_t n0, int64_t k, const float* a_natural, float (&result)[4][RT]) {
  static_assert(BlockSize == 32 || BlockSize == 64 || BlockSize == 128,
                "Int3Group4Avx512 requires BlockSize 32, 64, or 128");
  constexpr int kChunks32 = BlockSize / 32;
  const int64_t blocks = k / BlockSize;
  const int64_t raw_w_stride = (k * 3) / 8;
  constexpr int64_t kBlockBytes = (BlockSize * 3) / 8;
  const __m512i shifts = _mm512_set_epi32(21, 18, 15, 12, 9, 6, 3, 0, 21, 18, 15, 12, 9, 6, 3, 0);
  const __m512i mask7 = _mm512_set1_epi32(7);
  const __m512i four = _mm512_set1_epi32(4);
  __m512 acc[4][RT];
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < RT; ++r) {
      acc[c][r] = _mm512_setzero_ps();
    }
  }

  for (int64_t kb = 0; kb < blocks; ++kb) {
    const int64_t byte_off = kb * kBlockBytes;
    if (next_w_base != nullptr && (kb & 3) == 0) {
      for (int c = 0; c < 4; ++c) {
        _mm_prefetch(reinterpret_cast<const char*>(next_w_base + c * raw_w_stride + byte_off), _MM_HINT_T0);
      }
    }
    __m512 a0[kChunks32][RT], a1[kChunks32][RT];
    for (int ch = 0; ch < kChunks32; ++ch) {
      for (int r = 0; r < RT; ++r) {
        a0[ch][r] = _mm512_loadu_ps(a_natural + r * k + kb * BlockSize + ch * 32);
        a1[ch][r] = _mm512_loadu_ps(a_natural + r * k + kb * BlockSize + ch * 32 + 16);
      }
    }
    auto step_col = [&](int c) QMOE_LAMBDA_INLINE {
      const uint8_t* p = w_base + c * raw_w_stride + byte_off;
      const uint16_t s_bits = scales[(n0 + c) * blocks + kb];
      const float s_val = ScaleIsBf16 ? Avx512Int3BFloat16BitsToFloat(s_bits) : Avx512Int3HalfBitsToFloat(s_bits);
      const __m512 table = DeferredScale ? _mm512_set1_ps(s_val) : Int3LookupAvx512(s_val, ScaleIsBf16);
      __m512 w0[kChunks32], w1[kChunks32];
      for (int ch = 0; ch < kChunks32; ++ch) {
        w0[ch] = DequantInt3Avx512<DeferredScale>(Unpack16Int3Avx512(p + ch * 12 + 0, shifts, mask7, four), table);
        w1[ch] = DequantInt3Avx512<DeferredScale>(Unpack16Int3Avx512(p + ch * 12 + 6, shifts, mask7, four), table);
      }
      for (int r = 0; r < RT; ++r) {
        __m512 prod = _mm512_fmadd_ps(w0[0], a0[0][r], _mm512_mul_ps(w1[0], a1[0][r]));
        for (int ch = 1; ch < kChunks32; ++ch) {
          prod = _mm512_fmadd_ps(w0[ch], a0[ch][r], _mm512_fmadd_ps(w1[ch], a1[ch][r], prod));
        }
        acc[c][r] = DeferredScale ? _mm512_fmadd_ps(prod, table, acc[c][r]) : _mm512_add_ps(prod, acc[c][r]);
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
void Int3RowTileAvx512(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
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
  if constexpr (RT <= 2) {
    if (block_size == 32 || block_size == 64) {
      for (; n0 + 8 <= n_end; n0 += 8) {
        const uint8_t* w_base = weights + n0 * raw_w_stride;
        const uint8_t* next_w = (n0 + 16 <= n_end) ? (weights + (n0 + 8) * raw_w_stride) : nullptr;
        float result8[8][RT];
        if (block_size == 32) {
          if (scale_is_bf16) {
            Int3Group8Avx512<RT, 32, true, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result8);
          } else {
            Int3Group8Avx512<RT, 32, false, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result8);
          }
        } else {
          if (scale_is_bf16) {
            Int3Group8Avx512<RT, 64, true, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result8);
          } else {
            Int3Group8Avx512<RT, 64, false, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result8);
          }
        }
        for (int c = 0; c < 8; ++c) {
          const float b = (bias != nullptr)
                              ? (bias_is_bf16 ? Avx512Int3BFloat16BitsToFloat(bias[n0 + c]) : Avx512Int3HalfBitsToFloat(bias[n0 + c]))
                              : 0.0f;
          for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result8[c][r] + b;
        }
      }
    }
  }
  for (; n0 + 4 <= n_end; n0 += 4) {
    const uint8_t* w_base = weights + n0 * raw_w_stride;
    const uint8_t* next_w = (n0 + 8 <= n_end) ? (weights + (n0 + 4) * raw_w_stride) : nullptr;
    float result4[4][RT];
    if (block_size == 32) {
      if (scale_is_bf16) {
        Int3Group4Avx512<RT, 32, true, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result4);
      } else {
        Int3Group4Avx512<RT, 32, false, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result4);
      }
    } else if (block_size == 64) {
      if (scale_is_bf16) {
        Int3Group4Avx512<RT, 64, true, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result4);
      } else {
        Int3Group4Avx512<RT, 64, false, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result4);
      }
    } else {
      if (scale_is_bf16) {
        Int3Group4Avx512<RT, 128, true, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result4);
      } else {
        Int3Group4Avx512<RT, 128, false, DeferredScale>(w_base, next_w, scales, n0, k, a_natural, result4);
      }
    }
    for (int c = 0; c < 4; ++c) {
      const float b = (bias != nullptr)
                          ? (bias_is_bf16 ? Avx512Int3BFloat16BitsToFloat(bias[n0 + c]) : Avx512Int3HalfBitsToFloat(bias[n0 + c]))
                          : 0.0f;
      for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result4[c][r] + b;
    }
  }
  if (n0 < n_end) {
    Int3GemmRange(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16, n_total, k, a_natural, RT,
                  out + (n0 - n_begin), out_stride, n0, n_end, /*force_scalar=*/true, DeferredScale);
  }
}

}  // namespace

template <bool DeferredScale>
void Avx512Int3GemmRangeImpl(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                             int64_t block_size, bool scale_is_bf16, bool bias_is_bf16,
                             int64_t n_total, int64_t k, const float* a_natural, int64_t m,
                             float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  int64_t r = 0;
  while (r < m) {
    const int64_t left = m - r;
    const float* ar = a_natural + r * k;
    float* orow = out + r * out_stride;
    if (left >= 8) {
      Int3RowTileAvx512<8, DeferredScale>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16,
                                          n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 8;
    } else if (left == 7) {
      Int3RowTileAvx512<7, DeferredScale>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16,
                                          n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 7;
    } else if (left == 6) {
      Int3RowTileAvx512<6, DeferredScale>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16,
                                          n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 6;
    } else if (left == 5) {
      Int3RowTileAvx512<5, DeferredScale>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16,
                                          n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 5;
    } else if (left == 4) {
      Int3RowTileAvx512<4, DeferredScale>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16,
                                          n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 4;
    } else if (left == 3) {
      Int3RowTileAvx512<3, DeferredScale>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16,
                                          n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 3;
    } else if (left == 2) {
      Int3RowTileAvx512<2, DeferredScale>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16,
                                          n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 2;
    } else {
      Int3RowTileAvx512<1, DeferredScale>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16,
                                          n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 1;
    }
  }
}

void Avx512Int3GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                         int64_t block_size, bool scale_is_bf16, bool bias_is_bf16,
                         int64_t n_total, int64_t k, const float* a_natural, int64_t m,
                         float* out, int64_t out_stride, int64_t n_begin, int64_t n_end, bool deferred_scale) {
  if (deferred_scale) {
    Avx512Int3GemmRangeImpl<true>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16, n_total, k, a_natural,
                                  m, out, out_stride, n_begin, n_end);
  } else {
    Avx512Int3GemmRangeImpl<false>(weights, scales, bias, block_size, scale_is_bf16, bias_is_bf16, n_total, k,
                                   a_natural, m, out, out_stride, n_begin, n_end);
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
