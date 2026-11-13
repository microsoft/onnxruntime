// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__CUDA_ARCH__)
#include <immintrin.h>
#endif

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#if !defined(__CUDA_ARCH__)
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "core/common/safeint.h"
#endif

namespace onnxruntime {
namespace contrib {
namespace gguf_quant {

#if !defined(__CUDA_ARCH__)

struct SignXorMaskTable {
  alignas(32) uint32_t mask[256][8];
  constexpr SignXorMaskTable() : mask{} {
    for (int s = 0; s < 256; ++s) {
      for (int j = 0; j < 8; ++j) {
        mask[s][j] = ((s >> j) & 1) ? 0x80000000u : 0u;
      }
    }
  }
};

inline constexpr SignXorMaskTable kSignXorMask256{};

struct SignBytes7BitTable {
  alignas(64) uint64_t mask[128];
  constexpr SignBytes7BitTable() : mask{} {
    for (int idx = 0; idx < 128; ++idx) {
      const uint8_t s = ksigns_iq2xs[idx];
      uint64_t v = 0;
      for (int j = 0; j < 8; ++j) {
        const uint64_t b = ((s >> j) & 1) ? 0xFFULL : 0x01ULL;
        v |= (b << (j * 8));
      }
      mask[idx] = v;
    }
  }
};

struct SignBytes8BitTable {
  alignas(64) uint64_t mask[256];
  constexpr SignBytes8BitTable() : mask{} {
    for (int s = 0; s < 256; ++s) {
      uint64_t v = 0;
      for (int j = 0; j < 8; ++j) {
        const uint64_t b = ((s >> j) & 1) ? 0xFFULL : 0x01ULL;
        v |= (b << (j * 8));
      }
      mask[s] = v;
    }
  }
};

struct HighBit16LutTable {
  alignas(64) uint64_t mask[256];
  constexpr HighBit16LutTable() : mask{} {
    for (int s = 0; s < 256; ++s) {
      uint64_t v = 0;
      for (int j = 0; j < 8; ++j) {
        const uint64_t b = ((s >> j) & 1) ? 0x10ULL : 0x00ULL;
        v |= (b << (j * 8));
      }
      mask[s] = v;
    }
  }
};

inline constexpr SignBytes7BitTable kSignBytes7Bit{};
inline constexpr SignBytes8BitTable kSignBytes8Bit{};
inline constexpr HighBit16LutTable kHighBit16Lut{};

struct alignas(16) GgufQ8Block {
  float d;
  int32_t bsum;
  int8_t qs[32];
};

inline void QuantizeBlockQ8Scalar(const float* x, GgufQ8Block* blk) noexcept {
  float max_abs = 0.0f;
  for (int i = 0; i < 32; ++i) {
    const float a = std::fabs(x[i]);
    if (a > max_abs) max_abs = a;
  }
  if (!(max_abs > 0.0f)) {
    blk->d = 0.0f;
    blk->bsum = 0;
    std::memset(blk->qs, 0, sizeof(blk->qs));
    return;
  }
  const float d = max_abs / 127.0f;
  const float id = 127.0f / max_abs;
  const bool use_mul = std::isfinite(id);
  int32_t bsum = 0;
  for (int i = 0; i < 32; ++i) {
    const float scaled = use_mul ? (x[i] * id) : ((x[i] / max_abs) * 127.0f);
    int q = static_cast<int>(std::nearbyint(scaled));
    q = std::clamp(q, -127, 127);
    blk->qs[i] = static_cast<int8_t>(q);
    bsum += q;
  }
  blk->d = d;
  blk->bsum = bsum;
}

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#define ORT_GGUF_CPU_HAS_AVX2_TARGET 1
#else
#define ORT_GGUF_CPU_HAS_AVX2_TARGET 0
#endif

#if ORT_GGUF_CPU_HAS_AVX2_TARGET

__attribute__((target("avx2,fma,f16c"), always_inline)) inline float Fp16ToFloatF16c(uint16_t bits) noexcept {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(bits))));
}

__attribute__((target("avx2,fma,f16c"))) inline float HsumAvx2(__m256 v) {
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 sum128 = _mm_add_ps(lo, hi);
  const __m128 shuf = _mm_movehdup_ps(sum128);
  const __m128 sums = _mm_add_ps(sum128, shuf);
  const __m128 shuf2 = _mm_movehl_ps(shuf, sums);
  return _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
}

__attribute__((target("avx2,fma,f16c"))) inline void Hsum4Avx2(
    __m256 a0, __m256 a1, __m256 a2, __m256 a3, float* out4) {
  const __m256 t0 = _mm256_hadd_ps(a0, a1);
  const __m256 t1 = _mm256_hadd_ps(a2, a3);
  const __m256 t2 = _mm256_hadd_ps(t0, t1);
  const __m128 lo = _mm256_castps256_ps128(t2);
  const __m128 hi = _mm256_extractf128_ps(t2, 1);
  _mm_storeu_ps(out4, _mm_add_ps(lo, hi));
}

__attribute__((target("avx2,fma,f16c"))) inline __m256 DecodeGrid8SignedAvx2(uint64_t grid8, uint8_t signs) {
  const __m256i v_i32 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<int64_t>(grid8)));
  const __m256 v_f32 = _mm256_cvtepi32_ps(v_i32);
  const __m256 v_sign = _mm256_load_ps(reinterpret_cast<const float*>(kSignXorMask256.mask[signs]));
  return _mm256_xor_ps(v_f32, v_sign);
}

__attribute__((target("avx2,fma,f16c"))) inline void DecodeIq4Nl32Avx2(
    const uint8_t* qs, __m256& w0, __m256& w1, __m256& w2, __m256& w3) {
  const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
  const __m128i mask4 = _mm_set1_epi8(0x0F);
  const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
  const __m128i v_lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw16, mask4));
  const __m128i v_hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw16, 4), mask4));
  w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo));
  w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(v_lo, 8)));
  w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi));
  w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(v_hi, 8)));
}

__attribute__((target("avx2,fma,f16c"))) inline float VecDotGgufRowAvx2(
    int quant_type, const uint8_t* weight_row, const float* x, int64_t K) {
  __m256 total_acc0 = _mm256_setzero_ps();
  __m256 total_acc1 = _mm256_setzero_ps();

  switch (quant_type) {
    case kGgmlTypeIQ2_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 66;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk)) * 0.25f;
        const uint8_t* qs = blk + 2;
        const float* xp = x + i * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const uint8_t* aux8 = qs + 8 * ib32;
          const uint32_t aux32 = ReadU32Le(aux8 + 4);
          const float db = d * (0.5f + static_cast<float>(aux32 >> 28));
          const __m256 w0 = DecodeGrid8SignedAvx2(iq2xxs_grid[aux8[0]], ksigns_iq2xs[(aux32 >> 0) & 127u]);
          const __m256 w1 = DecodeGrid8SignedAvx2(iq2xxs_grid[aux8[1]], ksigns_iq2xs[(aux32 >> 7) & 127u]);
          const __m256 w2 = DecodeGrid8SignedAvx2(iq2xxs_grid[aux8[2]], ksigns_iq2xs[(aux32 >> 14) & 127u]);
          const __m256 w3 = DecodeGrid8SignedAvx2(iq2xxs_grid[aux8[3]], ksigns_iq2xs[(aux32 >> 21) & 127u]);
          __m256 sub0 = _mm256_mul_ps(w0, _mm256_loadu_ps(xp + 0));
          __m256 sub1 = _mm256_mul_ps(w1, _mm256_loadu_ps(xp + 8));
          sub0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 16), sub0);
          sub1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), sub1);
          total_acc0 = _mm256_fmadd_ps(_mm256_add_ps(sub0, sub1), _mm256_set1_ps(db), total_acc0);
          xp += 32;
        }
      }
      return HsumAvx2(total_acc0);
    }
    case kGgmlTypeIQ2_XS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 74;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk)) * 0.25f;
        const uint8_t* qs = blk + 2;
        const uint8_t* scales = blk + 66;
        const float* xp = x + i * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float db0 = d * (0.5f + static_cast<float>(scales[ib32] & 0xF));
          const float db1 = d * (0.5f + static_cast<float>(scales[ib32] >> 4));
          const uint16_t q0 = ReadU16Le(qs + (4 * ib32 + 0) * 2);
          const uint16_t q1 = ReadU16Le(qs + (4 * ib32 + 1) * 2);
          const uint16_t q2 = ReadU16Le(qs + (4 * ib32 + 2) * 2);
          const uint16_t q3 = ReadU16Le(qs + (4 * ib32 + 3) * 2);
          const __m256 w0 = DecodeGrid8SignedAvx2(iq2xs_grid[q0 & 511u], ksigns_iq2xs[q0 >> 9]);
          const __m256 w1 = DecodeGrid8SignedAvx2(iq2xs_grid[q1 & 511u], ksigns_iq2xs[q1 >> 9]);
          const __m256 w2 = DecodeGrid8SignedAvx2(iq2xs_grid[q2 & 511u], ksigns_iq2xs[q2 >> 9]);
          const __m256 w3 = DecodeGrid8SignedAvx2(iq2xs_grid[q3 & 511u], ksigns_iq2xs[q3 >> 9]);
          const __m256 sub0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 8), _mm256_mul_ps(w0, _mm256_loadu_ps(xp + 0)));
          const __m256 sub1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), _mm256_mul_ps(w2, _mm256_loadu_ps(xp + 16)));
          total_acc0 = _mm256_fmadd_ps(sub0, _mm256_set1_ps(db0), total_acc0);
          total_acc1 = _mm256_fmadd_ps(sub1, _mm256_set1_ps(db1), total_acc1);
          xp += 32;
        }
      }
      return HsumAvx2(_mm256_add_ps(total_acc0, total_acc1));
    }
    case kGgmlTypeIQ2_S: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 82;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk)) * 0.25f;
        const uint8_t* qs = blk + 2;
        const uint8_t* signs = qs + 32;
        const uint8_t* qh = blk + 66;
        const uint8_t* scales = blk + 74;
        const float* xp = x + i * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float db0 = d * (0.5f + static_cast<float>(scales[ib32] & 0xF));
          const float db1 = d * (0.5f + static_cast<float>(scales[ib32] >> 4));
          const uint32_t qh_b = qh[ib32];
          const __m256 w0 = DecodeGrid8SignedAvx2(iq2s_grid[qs[0] | ((qh_b << 8) & 0x300u)], signs[0]);
          const __m256 w1 = DecodeGrid8SignedAvx2(iq2s_grid[qs[1] | ((qh_b << 6) & 0x300u)], signs[1]);
          const __m256 w2 = DecodeGrid8SignedAvx2(iq2s_grid[qs[2] | ((qh_b << 4) & 0x300u)], signs[2]);
          const __m256 w3 = DecodeGrid8SignedAvx2(iq2s_grid[qs[3] | ((qh_b << 2) & 0x300u)], signs[3]);
          const __m256 sub0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 8), _mm256_mul_ps(w0, _mm256_loadu_ps(xp + 0)));
          const __m256 sub1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), _mm256_mul_ps(w2, _mm256_loadu_ps(xp + 16)));
          total_acc0 = _mm256_fmadd_ps(sub0, _mm256_set1_ps(db0), total_acc0);
          total_acc1 = _mm256_fmadd_ps(sub1, _mm256_set1_ps(db1), total_acc1);
          qs += 4;
          signs += 4;
          xp += 32;
        }
      }
      return HsumAvx2(_mm256_add_ps(total_acc0, total_acc1));
    }
    case kGgmlTypeIQ3_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 98;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk)) * 0.5f;
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64;
        const float* xp = x + i * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const uint32_t aux32 = ReadU32Le(scales_and_signs + 4 * ib32);
          const float db = d * (0.5f + static_cast<float>(aux32 >> 28));
          __m256 sub0 = _mm256_setzero_ps();
          __m256 sub1 = _mm256_setzero_ps();
          for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
            const uint64_t grid8 = static_cast<uint64_t>(iq3xxs_grid[qs[2 * l + 0]]) |
                                   (static_cast<uint64_t>(iq3xxs_grid[qs[2 * l + 1]]) << 32);
            const __m256 w = DecodeGrid8SignedAvx2(grid8, signs);
            if ((l & 1) == 0) {
              sub0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(xp + l * 8), sub0);
            } else {
              sub1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(xp + l * 8), sub1);
            }
          }
          total_acc0 = _mm256_fmadd_ps(_mm256_add_ps(sub0, sub1), _mm256_set1_ps(db), total_acc0);
          qs += 8;
          xp += 32;
        }
      }
      return HsumAvx2(total_acc0);
    }
    case kGgmlTypeIQ4_NL: {
      const int64_t nb = K / 32;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 18;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk));
        const float* xp = x + i * 32;
        __m256 w0, w1, w2, w3;
        DecodeIq4Nl32Avx2(blk + 2, w0, w1, w2, w3);
        __m256 sub0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 16), _mm256_mul_ps(w0, _mm256_loadu_ps(xp + 0)));
        __m256 sub1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), _mm256_mul_ps(w1, _mm256_loadu_ps(xp + 8)));
        total_acc0 = _mm256_fmadd_ps(_mm256_add_ps(sub0, sub1), _mm256_set1_ps(d), total_acc0);
      }
      return HsumAvx2(total_acc0);
    }
    case kGgmlTypeIQ4_XS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 136;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk));
        const uint16_t scales_h = ReadU16Le(blk + 2);
        const uint8_t* scales_l = blk + 4;
        const uint8_t* qs = blk + 8;
        const float* xp = x + i * 256;
        for (int ib = 0; ib < 8; ++ib) {
          const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
          const float dl = d * static_cast<float>(ls - 32);
          __m256 w0, w1, w2, w3;
          DecodeIq4Nl32Avx2(qs + ib * 16, w0, w1, w2, w3);
          __m256 sub0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 16), _mm256_mul_ps(w0, _mm256_loadu_ps(xp + 0)));
          __m256 sub1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), _mm256_mul_ps(w1, _mm256_loadu_ps(xp + 8)));
          total_acc0 = _mm256_fmadd_ps(_mm256_add_ps(sub0, sub1), _mm256_set1_ps(dl), total_acc0);
          xp += 32;
        }
      }
      return HsumAvx2(total_acc0);
    }
    case kGgmlTypeQ2_0: {
      const int64_t nb = K / 64;
      const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
      const __m256i mask2 = _mm256_set1_epi32(3);
      const __m256i one = _mm256_set1_epi32(1);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 18;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk));
        const uint8_t* qs = blk + 2;
        const float* xp = x + i * 64;
        __m256 sub0 = _mm256_setzero_ps();
        __m256 sub1 = _mm256_setzero_ps();
        for (int l = 0; l < 8; l += 2) {
          const __m256i q0 = _mm256_sub_epi32(
              _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(ReadU16Le(qs + l * 2)), shifts), mask2), one);
          const __m256i q1 = _mm256_sub_epi32(
              _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(ReadU16Le(qs + (l + 1) * 2)), shifts), mask2), one);
          sub0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q0), _mm256_loadu_ps(xp + l * 8), sub0);
          sub1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q1), _mm256_loadu_ps(xp + (l + 1) * 8), sub1);
        }
        total_acc0 = _mm256_fmadd_ps(_mm256_add_ps(sub0, sub1), _mm256_set1_ps(d), total_acc0);
      }
      return HsumAvx2(total_acc0);
    }
    case kGgmlTypeQ5_0: {
      const int64_t nb = K / 32;
      const __m256i shifts0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
      const __m256i shifts1 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
      const __m256i shifts2 = _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23);
      const __m256i shifts3 = _mm256_setr_epi32(24, 25, 26, 27, 28, 29, 30, 31);
      const __m256i mask1 = _mm256_set1_epi32(1);
      const __m256i off16 = _mm256_set1_epi32(16);
      const __m128i mask4 = _mm_set1_epi8(0x0F);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 22;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk));
        const __m256i qh_v = _mm256_set1_epi32(static_cast<int>(ReadU32Le(blk + 2)));
        const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 6));
        const __m128i lo_b = _mm_and_si128(raw16, mask4);
        const __m128i hi_b = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask4);
        const __m256i q0 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_cvtepu8_epi32(lo_b),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh_v, shifts0), mask1), 4)),
            off16);
        const __m256i q1 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_bsrli_si128(lo_b, 8)),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh_v, shifts1), mask1), 4)),
            off16);
        const __m256i q2 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_cvtepu8_epi32(hi_b),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh_v, shifts2), mask1), 4)),
            off16);
        const __m256i q3 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_bsrli_si128(hi_b, 8)),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh_v, shifts3), mask1), 4)),
            off16);
        const float* xp = x + i * 32;
        __m256 sub0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q2), _mm256_loadu_ps(xp + 16),
                                      _mm256_mul_ps(_mm256_cvtepi32_ps(q0), _mm256_loadu_ps(xp + 0)));
        __m256 sub1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q3), _mm256_loadu_ps(xp + 24),
                                      _mm256_mul_ps(_mm256_cvtepi32_ps(q1), _mm256_loadu_ps(xp + 8)));
        total_acc0 = _mm256_fmadd_ps(_mm256_add_ps(sub0, sub1), _mm256_set1_ps(d), total_acc0);
      }
      return HsumAvx2(total_acc0);
    }
    case kGgmlTypeQ4_0: {
      const int64_t nb = K / 32;
      const __m128i mask4 = _mm_set1_epi8(0x0F);
      const __m256i off8 = _mm256_set1_epi32(8);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 18;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk));
        const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
        const __m128i lo_b = _mm_and_si128(raw16, mask4);
        const __m128i hi_b = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask4);
        const __m256 w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(lo_b), off8));
        const __m256 w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_bsrli_si128(lo_b, 8)), off8));
        const __m256 w2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(hi_b), off8));
        const __m256 w3 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_bsrli_si128(hi_b, 8)), off8));
        const float* xp = x + i * 32;
        __m256 sub0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 16), _mm256_mul_ps(w0, _mm256_loadu_ps(xp + 0)));
        __m256 sub1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), _mm256_mul_ps(w1, _mm256_loadu_ps(xp + 8)));
        total_acc0 = _mm256_fmadd_ps(_mm256_add_ps(sub0, sub1), _mm256_set1_ps(d), total_acc0);
      }
      return HsumAvx2(total_acc0);
    }
    case kGgmlTypeQ8_0: {
      const int64_t nb = K / 32;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 34;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk));
        const __m128i raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
        const __m128i raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 18));
        const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw0));
        const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(raw0, 8)));
        const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw1));
        const __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(raw1, 8)));
        const float* xp = x + i * 32;
        __m256 sub0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 16), _mm256_mul_ps(w0, _mm256_loadu_ps(xp + 0)));
        __m256 sub1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), _mm256_mul_ps(w1, _mm256_loadu_ps(xp + 8)));
        total_acc0 = _mm256_fmadd_ps(_mm256_add_ps(sub0, sub1), _mm256_set1_ps(d), total_acc0);
      }
      return HsumAvx2(total_acc0);
    }
    default: {
      int64_t block_elems = 0;
      int64_t block_bytes = 0;
      if (!GgmlTypeBlockInfo(quant_type, block_elems, block_bytes) || block_elems <= 0) {
        return 0.0f;
      }
      const int64_t chunk_elems = (block_elems < 32) ? 32 : block_elems;
      const int64_t chunk_bytes = (chunk_elems / block_elems) * block_bytes;
      alignas(32) float buf[256];
      const int64_t n_chunks = K / chunk_elems;
      for (int64_t c = 0; c < n_chunks; ++c) {
        DequantizeGgufRowCpu(weight_row + c * chunk_bytes, quant_type, chunk_elems, buf);
        const float* xp = x + c * chunk_elems;
        for (int64_t j = 0; j < chunk_elems; j += 16) {
          total_acc0 = _mm256_fmadd_ps(_mm256_load_ps(buf + j), _mm256_loadu_ps(xp + j), total_acc0);
          total_acc1 = _mm256_fmadd_ps(_mm256_load_ps(buf + j + 8), _mm256_loadu_ps(xp + j + 8), total_acc1);
        }
      }
      return HsumAvx2(_mm256_add_ps(total_acc0, total_acc1));
    }
  }
}

__attribute__((target("avx2,fma,f16c"))) inline void VecDotGgufGateUpRowAvx2(
    int gate_type,
    const uint8_t* gate_row,
    int up_type,
    const uint8_t* up_row,
    const float* x,
    int64_t K,
    float* out_gate,
    float* out_up) {
  if (gate_type != up_type) {
    *out_gate = VecDotGgufRowAvx2(gate_type, gate_row, x, K);
    *out_up = VecDotGgufRowAvx2(up_type, up_row, x, K);
    return;
  }

  __m256 g_acc0 = _mm256_setzero_ps();
  __m256 g_acc1 = _mm256_setzero_ps();
  __m256 u_acc0 = _mm256_setzero_ps();
  __m256 u_acc1 = _mm256_setzero_ps();

  switch (gate_type) {
    case kGgmlTypeIQ2_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* g_blk = gate_row + i * 66;
        const uint8_t* u_blk = up_row + i * 66;
        const float gd = Fp16ToFloatF16c(ReadU16Le(g_blk)) * 0.25f;
        const float ud = Fp16ToFloatF16c(ReadU16Le(u_blk)) * 0.25f;
        const uint8_t* g_qs = g_blk + 2;
        const uint8_t* u_qs = u_blk + 2;
        const float* xp = x + i * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const uint8_t* g_aux8 = g_qs + 8 * ib32;
          const uint8_t* u_aux8 = u_qs + 8 * ib32;
          const uint32_t g_aux32 = ReadU32Le(g_aux8 + 4);
          const uint32_t u_aux32 = ReadU32Le(u_aux8 + 4);
          const float g_db = gd * (0.5f + static_cast<float>(g_aux32 >> 28));
          const float u_db = ud * (0.5f + static_cast<float>(u_aux32 >> 28));
          __m256 g_sub = _mm256_setzero_ps();
          __m256 u_sub = _mm256_setzero_ps();
          for (int l = 0; l < 4; ++l) {
            const __m256 vx = _mm256_loadu_ps(xp + l * 8);
            const __m256 gw = DecodeGrid8SignedAvx2(iq2xxs_grid[g_aux8[l]], ksigns_iq2xs[(g_aux32 >> (7 * l)) & 127u]);
            const __m256 uw = DecodeGrid8SignedAvx2(iq2xxs_grid[u_aux8[l]], ksigns_iq2xs[(u_aux32 >> (7 * l)) & 127u]);
            g_sub = _mm256_fmadd_ps(gw, vx, g_sub);
            u_sub = _mm256_fmadd_ps(uw, vx, u_sub);
          }
          g_acc0 = _mm256_fmadd_ps(g_sub, _mm256_set1_ps(g_db), g_acc0);
          u_acc0 = _mm256_fmadd_ps(u_sub, _mm256_set1_ps(u_db), u_acc0);
          xp += 32;
        }
      }
      *out_gate = HsumAvx2(g_acc0);
      *out_up = HsumAvx2(u_acc0);
      return;
    }
    case kGgmlTypeIQ2_XS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* g_blk = gate_row + i * 74;
        const uint8_t* u_blk = up_row + i * 74;
        const float gd = Fp16ToFloatF16c(ReadU16Le(g_blk)) * 0.25f;
        const float ud = Fp16ToFloatF16c(ReadU16Le(u_blk)) * 0.25f;
        const uint8_t* g_qs = g_blk + 2;
        const uint8_t* u_qs = u_blk + 2;
        const uint8_t* g_sc = g_blk + 66;
        const uint8_t* u_sc = u_blk + 66;
        const float* xp = x + i * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float g_db0 = gd * (0.5f + static_cast<float>(g_sc[ib32] & 0xF));
          const float g_db1 = gd * (0.5f + static_cast<float>(g_sc[ib32] >> 4));
          const float u_db0 = ud * (0.5f + static_cast<float>(u_sc[ib32] & 0xF));
          const float u_db1 = ud * (0.5f + static_cast<float>(u_sc[ib32] >> 4));
          const __m256 vx0 = _mm256_loadu_ps(xp + 0);
          const __m256 vx1 = _mm256_loadu_ps(xp + 8);
          const __m256 vx2 = _mm256_loadu_ps(xp + 16);
          const __m256 vx3 = _mm256_loadu_ps(xp + 24);
          const uint16_t gq0 = ReadU16Le(g_qs + (4 * ib32 + 0) * 2);
          const uint16_t gq1 = ReadU16Le(g_qs + (4 * ib32 + 1) * 2);
          const uint16_t gq2 = ReadU16Le(g_qs + (4 * ib32 + 2) * 2);
          const uint16_t gq3 = ReadU16Le(g_qs + (4 * ib32 + 3) * 2);
          const uint16_t uq0 = ReadU16Le(u_qs + (4 * ib32 + 0) * 2);
          const uint16_t uq1 = ReadU16Le(u_qs + (4 * ib32 + 1) * 2);
          const uint16_t uq2 = ReadU16Le(u_qs + (4 * ib32 + 2) * 2);
          const uint16_t uq3 = ReadU16Le(u_qs + (4 * ib32 + 3) * 2);
          const __m256 g_sub0 = _mm256_fmadd_ps(DecodeGrid8SignedAvx2(iq2xs_grid[gq1 & 511u], ksigns_iq2xs[gq1 >> 9]), vx1,
                                                _mm256_mul_ps(DecodeGrid8SignedAvx2(iq2xs_grid[gq0 & 511u], ksigns_iq2xs[gq0 >> 9]), vx0));
          const __m256 g_sub1 = _mm256_fmadd_ps(DecodeGrid8SignedAvx2(iq2xs_grid[gq3 & 511u], ksigns_iq2xs[gq3 >> 9]), vx3,
                                                _mm256_mul_ps(DecodeGrid8SignedAvx2(iq2xs_grid[gq2 & 511u], ksigns_iq2xs[gq2 >> 9]), vx2));
          const __m256 u_sub0 = _mm256_fmadd_ps(DecodeGrid8SignedAvx2(iq2xs_grid[uq1 & 511u], ksigns_iq2xs[uq1 >> 9]), vx1,
                                                _mm256_mul_ps(DecodeGrid8SignedAvx2(iq2xs_grid[uq0 & 511u], ksigns_iq2xs[uq0 >> 9]), vx0));
          const __m256 u_sub1 = _mm256_fmadd_ps(DecodeGrid8SignedAvx2(iq2xs_grid[uq3 & 511u], ksigns_iq2xs[uq3 >> 9]), vx3,
                                                _mm256_mul_ps(DecodeGrid8SignedAvx2(iq2xs_grid[uq2 & 511u], ksigns_iq2xs[uq2 >> 9]), vx2));
          g_acc0 = _mm256_fmadd_ps(g_sub0, _mm256_set1_ps(g_db0), g_acc0);
          g_acc1 = _mm256_fmadd_ps(g_sub1, _mm256_set1_ps(g_db1), g_acc1);
          u_acc0 = _mm256_fmadd_ps(u_sub0, _mm256_set1_ps(u_db0), u_acc0);
          u_acc1 = _mm256_fmadd_ps(u_sub1, _mm256_set1_ps(u_db1), u_acc1);
          xp += 32;
        }
      }
      *out_gate = HsumAvx2(_mm256_add_ps(g_acc0, g_acc1));
      *out_up = HsumAvx2(_mm256_add_ps(u_acc0, u_acc1));
      return;
    }
    case kGgmlTypeIQ2_S: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* g_blk = gate_row + i * 82;
        const uint8_t* u_blk = up_row + i * 82;
        const float gd = Fp16ToFloatF16c(ReadU16Le(g_blk)) * 0.25f;
        const float ud = Fp16ToFloatF16c(ReadU16Le(u_blk)) * 0.25f;
        const uint8_t* g_qs = g_blk + 2;
        const uint8_t* g_signs = g_qs + 32;
        const uint8_t* g_qh = g_blk + 66;
        const uint8_t* g_sc = g_blk + 74;
        const uint8_t* u_qs = u_blk + 2;
        const uint8_t* u_signs = u_qs + 32;
        const uint8_t* u_qh = u_blk + 66;
        const uint8_t* u_sc = u_blk + 74;
        const float* xp = x + i * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float g_db0 = gd * (0.5f + static_cast<float>(g_sc[ib32] & 0xF));
          const float g_db1 = gd * (0.5f + static_cast<float>(g_sc[ib32] >> 4));
          const float u_db0 = ud * (0.5f + static_cast<float>(u_sc[ib32] & 0xF));
          const float u_db1 = ud * (0.5f + static_cast<float>(u_sc[ib32] >> 4));
          const uint32_t g_qhb = g_qh[ib32];
          const uint32_t u_qhb = u_qh[ib32];
          const __m256 vx0 = _mm256_loadu_ps(xp + 0);
          const __m256 vx1 = _mm256_loadu_ps(xp + 8);
          const __m256 vx2 = _mm256_loadu_ps(xp + 16);
          const __m256 vx3 = _mm256_loadu_ps(xp + 24);
          const __m256 g_sub0 = _mm256_fmadd_ps(DecodeGrid8SignedAvx2(iq2s_grid[g_qs[1] | ((g_qhb << 6) & 0x300u)], g_signs[1]), vx1,
                                                _mm256_mul_ps(DecodeGrid8SignedAvx2(iq2s_grid[g_qs[0] | ((g_qhb << 8) & 0x300u)], g_signs[0]), vx0));
          const __m256 g_sub1 = _mm256_fmadd_ps(DecodeGrid8SignedAvx2(iq2s_grid[g_qs[3] | ((g_qhb << 2) & 0x300u)], g_signs[3]), vx3,
                                                _mm256_mul_ps(DecodeGrid8SignedAvx2(iq2s_grid[g_qs[2] | ((g_qhb << 4) & 0x300u)], g_signs[2]), vx2));
          const __m256 u_sub0 = _mm256_fmadd_ps(DecodeGrid8SignedAvx2(iq2s_grid[u_qs[1] | ((u_qhb << 6) & 0x300u)], u_signs[1]), vx1,
                                                _mm256_mul_ps(DecodeGrid8SignedAvx2(iq2s_grid[u_qs[0] | ((u_qhb << 8) & 0x300u)], u_signs[0]), vx0));
          const __m256 u_sub1 = _mm256_fmadd_ps(DecodeGrid8SignedAvx2(iq2s_grid[u_qs[3] | ((u_qhb << 2) & 0x300u)], u_signs[3]), vx3,
                                                _mm256_mul_ps(DecodeGrid8SignedAvx2(iq2s_grid[u_qs[2] | ((u_qhb << 4) & 0x300u)], u_signs[2]), vx2));
          g_acc0 = _mm256_fmadd_ps(g_sub0, _mm256_set1_ps(g_db0), g_acc0);
          g_acc1 = _mm256_fmadd_ps(g_sub1, _mm256_set1_ps(g_db1), g_acc1);
          u_acc0 = _mm256_fmadd_ps(u_sub0, _mm256_set1_ps(u_db0), u_acc0);
          u_acc1 = _mm256_fmadd_ps(u_sub1, _mm256_set1_ps(u_db1), u_acc1);
          g_qs += 4;
          g_signs += 4;
          u_qs += 4;
          u_signs += 4;
          xp += 32;
        }
      }
      *out_gate = HsumAvx2(_mm256_add_ps(g_acc0, g_acc1));
      *out_up = HsumAvx2(_mm256_add_ps(u_acc0, u_acc1));
      return;
    }
    case kGgmlTypeIQ3_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* g_blk = gate_row + i * 98;
        const uint8_t* u_blk = up_row + i * 98;
        const float gd = Fp16ToFloatF16c(ReadU16Le(g_blk)) * 0.5f;
        const float ud = Fp16ToFloatF16c(ReadU16Le(u_blk)) * 0.5f;
        const uint8_t* g_qs = g_blk + 2;
        const uint8_t* g_ss = g_qs + 64;
        const uint8_t* u_qs = u_blk + 2;
        const uint8_t* u_ss = u_qs + 64;
        const float* xp = x + i * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const uint32_t g_aux32 = ReadU32Le(g_ss + 4 * ib32);
          const uint32_t u_aux32 = ReadU32Le(u_ss + 4 * ib32);
          const float g_db = gd * (0.5f + static_cast<float>(g_aux32 >> 28));
          const float u_db = ud * (0.5f + static_cast<float>(u_aux32 >> 28));
          __m256 g_sub = _mm256_setzero_ps();
          __m256 u_sub = _mm256_setzero_ps();
          for (int l = 0; l < 4; ++l) {
            const __m256 vx = _mm256_loadu_ps(xp + l * 8);
            const uint8_t g_signs = ksigns_iq2xs[(g_aux32 >> (7 * l)) & 127u];
            const uint8_t u_signs = ksigns_iq2xs[(u_aux32 >> (7 * l)) & 127u];
            const uint64_t g_grid8 = static_cast<uint64_t>(iq3xxs_grid[g_qs[2 * l + 0]]) |
                                     (static_cast<uint64_t>(iq3xxs_grid[g_qs[2 * l + 1]]) << 32);
            const uint64_t u_grid8 = static_cast<uint64_t>(iq3xxs_grid[u_qs[2 * l + 0]]) |
                                     (static_cast<uint64_t>(iq3xxs_grid[u_qs[2 * l + 1]]) << 32);
            g_sub = _mm256_fmadd_ps(DecodeGrid8SignedAvx2(g_grid8, g_signs), vx, g_sub);
            u_sub = _mm256_fmadd_ps(DecodeGrid8SignedAvx2(u_grid8, u_signs), vx, u_sub);
          }
          g_acc0 = _mm256_fmadd_ps(g_sub, _mm256_set1_ps(g_db), g_acc0);
          u_acc0 = _mm256_fmadd_ps(u_sub, _mm256_set1_ps(u_db), u_acc0);
          g_qs += 8;
          u_qs += 8;
          xp += 32;
        }
      }
      *out_gate = HsumAvx2(g_acc0);
      *out_up = HsumAvx2(u_acc0);
      return;
    }
    case kGgmlTypeIQ4_XS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* g_blk = gate_row + i * 136;
        const uint8_t* u_blk = up_row + i * 136;
        const float gd = Fp16ToFloatF16c(ReadU16Le(g_blk));
        const float ud = Fp16ToFloatF16c(ReadU16Le(u_blk));
        const uint16_t g_sh = ReadU16Le(g_blk + 2);
        const uint16_t u_sh = ReadU16Le(u_blk + 2);
        const uint8_t* g_sl = g_blk + 4;
        const uint8_t* u_sl = u_blk + 4;
        const uint8_t* g_qs = g_blk + 8;
        const uint8_t* u_qs = u_blk + 8;
        const float* xp = x + i * 256;
        for (int ib = 0; ib < 8; ++ib) {
          const int g_ls = ((g_sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((g_sh >> (2 * ib)) & 3) << 4);
          const int u_ls = ((u_sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((u_sh >> (2 * ib)) & 3) << 4);
          const float g_dl = gd * static_cast<float>(g_ls - 32);
          const float u_dl = ud * static_cast<float>(u_ls - 32);
          __m256 gw0, gw1, gw2, gw3;
          __m256 uw0, uw1, uw2, uw3;
          DecodeIq4Nl32Avx2(g_qs + ib * 16, gw0, gw1, gw2, gw3);
          DecodeIq4Nl32Avx2(u_qs + ib * 16, uw0, uw1, uw2, uw3);
          const __m256 vx0 = _mm256_loadu_ps(xp + 0);
          const __m256 vx1 = _mm256_loadu_ps(xp + 8);
          const __m256 vx2 = _mm256_loadu_ps(xp + 16);
          const __m256 vx3 = _mm256_loadu_ps(xp + 24);
          const __m256 g_sub = _mm256_add_ps(_mm256_fmadd_ps(gw2, vx2, _mm256_mul_ps(gw0, vx0)),
                                             _mm256_fmadd_ps(gw3, vx3, _mm256_mul_ps(gw1, vx1)));
          const __m256 u_sub = _mm256_add_ps(_mm256_fmadd_ps(uw2, vx2, _mm256_mul_ps(uw0, vx0)),
                                             _mm256_fmadd_ps(uw3, vx3, _mm256_mul_ps(uw1, vx1)));
          g_acc0 = _mm256_fmadd_ps(g_sub, _mm256_set1_ps(g_dl), g_acc0);
          u_acc0 = _mm256_fmadd_ps(u_sub, _mm256_set1_ps(u_dl), u_acc0);
          xp += 32;
        }
      }
      *out_gate = HsumAvx2(g_acc0);
      *out_up = HsumAvx2(u_acc0);
      return;
    }
    default: {
      *out_gate = VecDotGgufRowAvx2(gate_type, gate_row, x, K);
      *out_up = VecDotGgufRowAvx2(up_type, up_row, x, K);
      return;
    }
  }
}

__attribute__((target("avx2,fma,f16c"))) inline void DecodeQ5_0_32Avx2(
    const uint8_t* blk, __m256& w0, __m256& w1, __m256& w2, __m256& w3) {
  const __m256i shifts0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  const __m256i shifts1 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
  const __m256i shifts2 = _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23);
  const __m256i shifts3 = _mm256_setr_epi32(24, 25, 26, 27, 28, 29, 30, 31);
  const __m256i mask1 = _mm256_set1_epi32(1);
  const __m256i off16 = _mm256_set1_epi32(16);
  const __m128i mask4 = _mm_set1_epi8(0x0F);
  const __m256i qh_v = _mm256_set1_epi32(static_cast<int>(ReadU32Le(blk + 2)));
  const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 6));
  const __m128i lo_b = _mm_and_si128(raw16, mask4);
  const __m128i hi_b = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask4);
  w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
      _mm256_or_si256(_mm256_cvtepu8_epi32(lo_b),
                      _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh_v, shifts0), mask1), 4)),
      off16));
  w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
      _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_bsrli_si128(lo_b, 8)),
                      _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh_v, shifts1), mask1), 4)),
      off16));
  w2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
      _mm256_or_si256(_mm256_cvtepu8_epi32(hi_b),
                      _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh_v, shifts2), mask1), 4)),
      off16));
  w3 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
      _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_bsrli_si128(hi_b, 8)),
                      _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh_v, shifts3), mask1), 4)),
      off16));
}

__attribute__((target("avx2,fma,f16c"))) inline void VecDotGgufRow4xAvx2(
    int quant_type,
    const uint8_t* row0,
    const uint8_t* row1,
    const uint8_t* row2,
    const uint8_t* row3,
    const float* x,
    int64_t K,
    float* out4) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  switch (quant_type) {
    case kGgmlTypeQ2_0: {
      const int64_t nb = K / 64;
      const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
      const __m256i mask2 = _mm256_set1_epi32(3);
      const __m256i one = _mm256_set1_epi32(1);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 18;
        const uint8_t* b1 = row1 + i * 18;
        const uint8_t* b2 = row2 + i * 18;
        const uint8_t* b3 = row3 + i * 18;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0));
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1));
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2));
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3));
        const float* xp = x + i * 64;
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        for (int l = 0; l < 8; ++l) {
          const __m256 vx = _mm256_loadu_ps(xp + l * 8);
          const __m256 w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
              _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(ReadU16Le(b0 + 2 + l * 2)), shifts), mask2), one));
          const __m256 w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
              _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(ReadU16Le(b1 + 2 + l * 2)), shifts), mask2), one));
          const __m256 w2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
              _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(ReadU16Le(b2 + 2 + l * 2)), shifts), mask2), one));
          const __m256 w3 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
              _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(ReadU16Le(b3 + 2 + l * 2)), shifts), mask2), one));
          s0 = _mm256_fmadd_ps(w0, vx, s0);
          s1 = _mm256_fmadd_ps(w1, vx, s1);
          s2 = _mm256_fmadd_ps(w2, vx, s2);
          s3 = _mm256_fmadd_ps(w3, vx, s3);
        }
        acc0 = _mm256_fmadd_ps(s0, _mm256_set1_ps(d0), acc0);
        acc1 = _mm256_fmadd_ps(s1, _mm256_set1_ps(d1), acc1);
        acc2 = _mm256_fmadd_ps(s2, _mm256_set1_ps(d2), acc2);
        acc3 = _mm256_fmadd_ps(s3, _mm256_set1_ps(d3), acc3);
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      return;
    }
    case kGgmlTypeIQ4_NL: {
      const int64_t nb = K / 32;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 18;
        const uint8_t* b1 = row1 + i * 18;
        const uint8_t* b2 = row2 + i * 18;
        const uint8_t* b3 = row3 + i * 18;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0));
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1));
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2));
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3));
        const float* xp = x + i * 32;
        const __m256 vx0 = _mm256_loadu_ps(xp + 0);
        const __m256 vx1 = _mm256_loadu_ps(xp + 8);
        const __m256 vx2 = _mm256_loadu_ps(xp + 16);
        const __m256 vx3 = _mm256_loadu_ps(xp + 24);
        __m256 w0, w1, w2, w3;
        DecodeIq4Nl32Avx2(b0 + 2, w0, w1, w2, w3);
        acc0 = _mm256_fmadd_ps(
            _mm256_add_ps(_mm256_fmadd_ps(w2, vx2, _mm256_mul_ps(w0, vx0)),
                          _mm256_fmadd_ps(w3, vx3, _mm256_mul_ps(w1, vx1))),
            _mm256_set1_ps(d0), acc0);
        DecodeIq4Nl32Avx2(b1 + 2, w0, w1, w2, w3);
        acc1 = _mm256_fmadd_ps(
            _mm256_add_ps(_mm256_fmadd_ps(w2, vx2, _mm256_mul_ps(w0, vx0)),
                          _mm256_fmadd_ps(w3, vx3, _mm256_mul_ps(w1, vx1))),
            _mm256_set1_ps(d1), acc1);
        DecodeIq4Nl32Avx2(b2 + 2, w0, w1, w2, w3);
        acc2 = _mm256_fmadd_ps(
            _mm256_add_ps(_mm256_fmadd_ps(w2, vx2, _mm256_mul_ps(w0, vx0)),
                          _mm256_fmadd_ps(w3, vx3, _mm256_mul_ps(w1, vx1))),
            _mm256_set1_ps(d2), acc2);
        DecodeIq4Nl32Avx2(b3 + 2, w0, w1, w2, w3);
        acc3 = _mm256_fmadd_ps(
            _mm256_add_ps(_mm256_fmadd_ps(w2, vx2, _mm256_mul_ps(w0, vx0)),
                          _mm256_fmadd_ps(w3, vx3, _mm256_mul_ps(w1, vx1))),
            _mm256_set1_ps(d3), acc3);
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      return;
    }
    case kGgmlTypeQ5_0: {
      const int64_t nb = K / 32;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 22;
        const uint8_t* b1 = row1 + i * 22;
        const uint8_t* b2 = row2 + i * 22;
        const uint8_t* b3 = row3 + i * 22;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0));
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1));
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2));
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3));
        const float* xp = x + i * 32;
        const __m256 vx0 = _mm256_loadu_ps(xp + 0);
        const __m256 vx1 = _mm256_loadu_ps(xp + 8);
        const __m256 vx2 = _mm256_loadu_ps(xp + 16);
        const __m256 vx3 = _mm256_loadu_ps(xp + 24);
        __m256 w0, w1, w2, w3;
        DecodeQ5_0_32Avx2(b0, w0, w1, w2, w3);
        acc0 = _mm256_fmadd_ps(
            _mm256_add_ps(_mm256_fmadd_ps(w2, vx2, _mm256_mul_ps(w0, vx0)),
                          _mm256_fmadd_ps(w3, vx3, _mm256_mul_ps(w1, vx1))),
            _mm256_set1_ps(d0), acc0);
        DecodeQ5_0_32Avx2(b1, w0, w1, w2, w3);
        acc1 = _mm256_fmadd_ps(
            _mm256_add_ps(_mm256_fmadd_ps(w2, vx2, _mm256_mul_ps(w0, vx0)),
                          _mm256_fmadd_ps(w3, vx3, _mm256_mul_ps(w1, vx1))),
            _mm256_set1_ps(d1), acc1);
        DecodeQ5_0_32Avx2(b2, w0, w1, w2, w3);
        acc2 = _mm256_fmadd_ps(
            _mm256_add_ps(_mm256_fmadd_ps(w2, vx2, _mm256_mul_ps(w0, vx0)),
                          _mm256_fmadd_ps(w3, vx3, _mm256_mul_ps(w1, vx1))),
            _mm256_set1_ps(d2), acc2);
        DecodeQ5_0_32Avx2(b3, w0, w1, w2, w3);
        acc3 = _mm256_fmadd_ps(
            _mm256_add_ps(_mm256_fmadd_ps(w2, vx2, _mm256_mul_ps(w0, vx0)),
                          _mm256_fmadd_ps(w3, vx3, _mm256_mul_ps(w1, vx1))),
            _mm256_set1_ps(d3), acc3);
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      return;
    }
    case kGgmlTypeIQ4_XS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 136;
        const uint8_t* b1 = row1 + i * 136;
        const uint8_t* b2 = row2 + i * 136;
        const uint8_t* b3 = row3 + i * 136;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0));
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1));
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2));
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3));
        const uint16_t sh0 = ReadU16Le(b0 + 2);
        const uint16_t sh1 = ReadU16Le(b1 + 2);
        const uint16_t sh2 = ReadU16Le(b2 + 2);
        const uint16_t sh3 = ReadU16Le(b3 + 2);
        const float* xp = x + i * 256;
        for (int ib = 0; ib < 8; ++ib) {
          const int ls0 = ((b0[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh0 >> (2 * ib)) & 3) << 4);
          const int ls1 = ((b1[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh1 >> (2 * ib)) & 3) << 4);
          const int ls2 = ((b2[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh2 >> (2 * ib)) & 3) << 4);
          const int ls3 = ((b3[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh3 >> (2 * ib)) & 3) << 4);
          const __m256 vx0 = _mm256_loadu_ps(xp + 0);
          const __m256 vx1 = _mm256_loadu_ps(xp + 8);
          const __m256 vx2 = _mm256_loadu_ps(xp + 16);
          const __m256 vx3 = _mm256_loadu_ps(xp + 24);
          __m256 w0, w1, w2, w3;
          DecodeIq4Nl32Avx2(b0 + 8 + ib * 16, w0, w1, w2, w3);
          acc0 = _mm256_fmadd_ps(
              _mm256_add_ps(_mm256_fmadd_ps(w2, vx2, _mm256_mul_ps(w0, vx0)),
                            _mm256_fmadd_ps(w3, vx3, _mm256_mul_ps(w1, vx1))),
              _mm256_set1_ps(d0 * static_cast<float>(ls0 - 32)), acc0);
          DecodeIq4Nl32Avx2(b1 + 8 + ib * 16, w0, w1, w2, w3);
          acc1 = _mm256_fmadd_ps(
              _mm256_add_ps(_mm256_fmadd_ps(w2, vx2, _mm256_mul_ps(w0, vx0)),
                            _mm256_fmadd_ps(w3, vx3, _mm256_mul_ps(w1, vx1))),
              _mm256_set1_ps(d1 * static_cast<float>(ls1 - 32)), acc1);
          DecodeIq4Nl32Avx2(b2 + 8 + ib * 16, w0, w1, w2, w3);
          acc2 = _mm256_fmadd_ps(
              _mm256_add_ps(_mm256_fmadd_ps(w2, vx2, _mm256_mul_ps(w0, vx0)),
                            _mm256_fmadd_ps(w3, vx3, _mm256_mul_ps(w1, vx1))),
              _mm256_set1_ps(d2 * static_cast<float>(ls2 - 32)), acc2);
          DecodeIq4Nl32Avx2(b3 + 8 + ib * 16, w0, w1, w2, w3);
          acc3 = _mm256_fmadd_ps(
              _mm256_add_ps(_mm256_fmadd_ps(w2, vx2, _mm256_mul_ps(w0, vx0)),
                            _mm256_fmadd_ps(w3, vx3, _mm256_mul_ps(w1, vx1))),
              _mm256_set1_ps(d3 * static_cast<float>(ls3 - 32)), acc3);
          xp += 32;
        }
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      return;
    }
    default: {
      out4[0] = VecDotGgufRowAvx2(quant_type, row0, x, K);
      out4[1] = VecDotGgufRowAvx2(quant_type, row1, x, K);
      out4[2] = VecDotGgufRowAvx2(quant_type, row2, x, K);
      out4[3] = VecDotGgufRowAvx2(quant_type, row3, x, K);
      return;
    }
  }
}

__attribute__((target("avx2,fma,f16c"))) inline void QuantizeBlockQ8Avx2(
    const float* x, GgufQ8Block* blk) noexcept {
  const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  const __m256 x0 = _mm256_loadu_ps(x + 0);
  const __m256 x1 = _mm256_loadu_ps(x + 8);
  const __m256 x2 = _mm256_loadu_ps(x + 16);
  const __m256 x3 = _mm256_loadu_ps(x + 24);
  const __m256 m01 = _mm256_max_ps(_mm256_and_ps(x0, sign_mask), _mm256_and_ps(x1, sign_mask));
  const __m256 m23 = _mm256_max_ps(_mm256_and_ps(x2, sign_mask), _mm256_and_ps(x3, sign_mask));
  const __m256 mv = _mm256_max_ps(m01, m23);
  __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(mv), _mm256_extractf128_ps(mv, 1));
  m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
  m128 = _mm_max_ss(m128, _mm_movehdup_ps(m128));
  const float max_abs = _mm_cvtss_f32(m128);
  if (!(max_abs > 0.0f)) {
    blk->d = 0.0f;
    blk->bsum = 0;
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(blk->qs), _mm256_setzero_si256());
    return;
  }

  const float d = max_abs / 127.0f;
  const float id = 127.0f / max_abs;
  __m256 s0, s1, s2, s3;
  if (std::isfinite(id)) {
    const __m256 id_v = _mm256_set1_ps(id);
    s0 = _mm256_mul_ps(x0, id_v);
    s1 = _mm256_mul_ps(x1, id_v);
    s2 = _mm256_mul_ps(x2, id_v);
    s3 = _mm256_mul_ps(x3, id_v);
  } else {
    const __m256 max_v = _mm256_set1_ps(max_abs);
    const __m256 c127_v = _mm256_set1_ps(127.0f);
    s0 = _mm256_mul_ps(_mm256_div_ps(x0, max_v), c127_v);
    s1 = _mm256_mul_ps(_mm256_div_ps(x1, max_v), c127_v);
    s2 = _mm256_mul_ps(_mm256_div_ps(x2, max_v), c127_v);
    s3 = _mm256_mul_ps(_mm256_div_ps(x3, max_v), c127_v);
  }
  const __m256i q0 = _mm256_cvtps_epi32(s0);
  const __m256i q1 = _mm256_cvtps_epi32(s1);
  const __m256i q2 = _mm256_cvtps_epi32(s2);
  const __m256i q3 = _mm256_cvtps_epi32(s3);
  const __m256i p01 = _mm256_packs_epi32(q0, q1);
  const __m256i p23 = _mm256_packs_epi32(q2, q3);
  const __m256i p8_raw = _mm256_max_epi8(_mm256_packs_epi16(p01, p23), _mm256_set1_epi8(-127));
  const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  const __m256i qs_i8 = _mm256_permutevar8x32_epi32(p8_raw, perm);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(blk->qs), qs_i8);

  const __m256i sad = _mm256_sad_epu8(
      _mm256_xor_si256(qs_i8, _mm256_set1_epi8(-128)), _mm256_setzero_si256());
  const __m128i sad_sum128 = _mm_add_epi64(_mm256_castsi256_si128(sad), _mm256_extracti128_si256(sad, 1));
  const int64_t u_sum = _mm_cvtsi128_si64(sad_sum128) + _mm_extract_epi64(sad_sum128, 1);
  blk->d = d;
  blk->bsum = static_cast<int32_t>(u_sum - 4096);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq2Xxs32Q8Avx2(
    const uint8_t* aux8, float d_0125_q8d, __m256i x_q_32, __m256& acc) noexcept {
  const uint32_t aux32 = ReadU32Le(aux8 + 4);
  const __m256i grid_u8 = _mm256_set_epi64x(
      static_cast<int64_t>(iq2xxs_grid[aux8[3]]),
      static_cast<int64_t>(iq2xxs_grid[aux8[2]]),
      static_cast<int64_t>(iq2xxs_grid[aux8[1]]),
      static_cast<int64_t>(iq2xxs_grid[aux8[0]]));
  const __m256i signs = _mm256_set_epi64x(
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 21) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 14) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 7) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 0) & 127u]));
  const __m256i p16 = _mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_q_32, signs));
  const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(static_cast<int16_t>(2 * (aux32 >> 28) + 1)));
  acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(d_0125_q8d), acc);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq2Xs32Q8Avx2(
    const uint8_t* qs_ib, uint8_t sc, float d_0125_q8d, __m256i x_q_32, __m256& acc) noexcept {
  const uint16_t q0 = ReadU16Le(qs_ib + 0);
  const uint16_t q1 = ReadU16Le(qs_ib + 2);
  const uint16_t q2 = ReadU16Le(qs_ib + 4);
  const uint16_t q3 = ReadU16Le(qs_ib + 6);
  const __m256i grid_u8 = _mm256_set_epi64x(
      static_cast<int64_t>(iq2xs_grid[q3 & 511u]),
      static_cast<int64_t>(iq2xs_grid[q2 & 511u]),
      static_cast<int64_t>(iq2xs_grid[q1 & 511u]),
      static_cast<int64_t>(iq2xs_grid[q0 & 511u]));
  const __m256i signs = _mm256_set_epi64x(
      static_cast<int64_t>(kSignBytes7Bit.mask[q3 >> 9]),
      static_cast<int64_t>(kSignBytes7Bit.mask[q2 >> 9]),
      static_cast<int64_t>(kSignBytes7Bit.mask[q1 >> 9]),
      static_cast<int64_t>(kSignBytes7Bit.mask[q0 >> 9]));
  const __m256i p16 = _mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_q_32, signs));
  const __m128i ls_lo = _mm_set1_epi16(static_cast<int16_t>(2 * (sc & 0xF) + 1));
  const __m128i ls_hi = _mm_set1_epi16(static_cast<int16_t>(2 * (sc >> 4) + 1));
  const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set_m128i(ls_hi, ls_lo));
  acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(d_0125_q8d), acc);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq2S32Q8Avx2(
    const uint8_t* qs_ib, const uint8_t* signs_ib, uint32_t qh_b, uint8_t sc,
    float d_0125_q8d, __m256i x_q_32, __m256& acc) noexcept {
  const __m256i grid_u8 = _mm256_set_epi64x(
      static_cast<int64_t>(iq2s_grid[qs_ib[3] | ((qh_b << 2) & 0x300u)]),
      static_cast<int64_t>(iq2s_grid[qs_ib[2] | ((qh_b << 4) & 0x300u)]),
      static_cast<int64_t>(iq2s_grid[qs_ib[1] | ((qh_b << 6) & 0x300u)]),
      static_cast<int64_t>(iq2s_grid[qs_ib[0] | ((qh_b << 8) & 0x300u)]));
  const __m256i signs = _mm256_set_epi64x(
      static_cast<int64_t>(kSignBytes8Bit.mask[signs_ib[3]]),
      static_cast<int64_t>(kSignBytes8Bit.mask[signs_ib[2]]),
      static_cast<int64_t>(kSignBytes8Bit.mask[signs_ib[1]]),
      static_cast<int64_t>(kSignBytes8Bit.mask[signs_ib[0]]));
  const __m256i p16 = _mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_q_32, signs));
  const __m128i ls_lo = _mm_set1_epi16(static_cast<int16_t>(2 * (sc & 0xF) + 1));
  const __m128i ls_hi = _mm_set1_epi16(static_cast<int16_t>(2 * (sc >> 4) + 1));
  const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set_m128i(ls_hi, ls_lo));
  acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(d_0125_q8d), acc);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq3Xxs32Q8Avx2(
    const uint8_t* qs_ib, uint32_t aux32, float d_025_q8d, __m256i x_q_32, __m256& acc) noexcept {
  const __m256i grid_u8 = _mm256_setr_epi32(
      static_cast<int>(iq3xxs_grid[qs_ib[0]]),
      static_cast<int>(iq3xxs_grid[qs_ib[1]]),
      static_cast<int>(iq3xxs_grid[qs_ib[2]]),
      static_cast<int>(iq3xxs_grid[qs_ib[3]]),
      static_cast<int>(iq3xxs_grid[qs_ib[4]]),
      static_cast<int>(iq3xxs_grid[qs_ib[5]]),
      static_cast<int>(iq3xxs_grid[qs_ib[6]]),
      static_cast<int>(iq3xxs_grid[qs_ib[7]]));
  const __m256i signs = _mm256_set_epi64x(
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 21) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 14) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 7) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 0) & 127u]));
  const __m256i p16 = _mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_q_32, signs));
  const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(static_cast<int16_t>(2 * (aux32 >> 28) + 1)));
  acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(d_025_q8d), acc);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq4Nl32Q8Avx2(
    const uint8_t* qs16, __m256i lut_iq4nl_256, __m256i mask4_256,
    __m256i ax_u8, __m256i x_q_32, int16_t sub_scale_i16, float scale_f32, __m256& acc) noexcept {
  const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs16));
  const __m256i nibbles = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw16, 4), raw16), mask4_256);
  const __m256i w_i8 = _mm256_shuffle_epi8(lut_iq4nl_256, nibbles);
  const __m256i p16 = _mm256_maddubs_epi16(ax_u8, _mm256_sign_epi8(w_i8, x_q_32));
  const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(sub_scale_i16));
  acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(scale_f32), acc);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq2Xxs32Q8Batch2Avx2(
    const uint8_t* aux8,
    float d_a, __m256i x_qa, __m256& acc_a,
    float d_b, __m256i x_qb, __m256& acc_b) noexcept {
  const uint32_t aux32 = ReadU32Le(aux8 + 4);
  const __m256i grid_u8 = _mm256_set_epi64x(
      static_cast<int64_t>(iq2xxs_grid[aux8[3]]),
      static_cast<int64_t>(iq2xxs_grid[aux8[2]]),
      static_cast<int64_t>(iq2xxs_grid[aux8[1]]),
      static_cast<int64_t>(iq2xxs_grid[aux8[0]]));
  const __m256i signs = _mm256_set_epi64x(
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 21) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 14) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 7) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 0) & 127u]));
  const __m256i ls_v = _mm256_set1_epi16(static_cast<int16_t>(2 * (aux32 >> 28) + 1));
  const __m256i p32_a = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qa, signs)), ls_v);
  const __m256i p32_b = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qb, signs)), ls_v);
  acc_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_a), _mm256_set1_ps(d_a), acc_a);
  acc_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_b), _mm256_set1_ps(d_b), acc_b);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq2Xs32Q8Batch2Avx2(
    const uint8_t* qs_ib, uint8_t sc,
    float d_a, __m256i x_qa, __m256& acc_a,
    float d_b, __m256i x_qb, __m256& acc_b) noexcept {
  const uint16_t q0 = ReadU16Le(qs_ib + 0);
  const uint16_t q1 = ReadU16Le(qs_ib + 2);
  const uint16_t q2 = ReadU16Le(qs_ib + 4);
  const uint16_t q3 = ReadU16Le(qs_ib + 6);
  const __m256i grid_u8 = _mm256_set_epi64x(
      static_cast<int64_t>(iq2xs_grid[q3 & 511u]),
      static_cast<int64_t>(iq2xs_grid[q2 & 511u]),
      static_cast<int64_t>(iq2xs_grid[q1 & 511u]),
      static_cast<int64_t>(iq2xs_grid[q0 & 511u]));
  const __m256i signs = _mm256_set_epi64x(
      static_cast<int64_t>(kSignBytes7Bit.mask[q3 >> 9]),
      static_cast<int64_t>(kSignBytes7Bit.mask[q2 >> 9]),
      static_cast<int64_t>(kSignBytes7Bit.mask[q1 >> 9]),
      static_cast<int64_t>(kSignBytes7Bit.mask[q0 >> 9]));
  const __m256i ls_v = _mm256_set_m128i(
      _mm_set1_epi16(static_cast<int16_t>(2 * (sc >> 4) + 1)),
      _mm_set1_epi16(static_cast<int16_t>(2 * (sc & 0xF) + 1)));
  const __m256i p32_a = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qa, signs)), ls_v);
  const __m256i p32_b = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qb, signs)), ls_v);
  acc_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_a), _mm256_set1_ps(d_a), acc_a);
  acc_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_b), _mm256_set1_ps(d_b), acc_b);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq2S32Q8Batch2Avx2(
    const uint8_t* qs_ib, const uint8_t* signs_ib, uint32_t qh_b, uint8_t sc,
    float d_a, __m256i x_qa, __m256& acc_a,
    float d_b, __m256i x_qb, __m256& acc_b) noexcept {
  const __m256i grid_u8 = _mm256_set_epi64x(
      static_cast<int64_t>(iq2s_grid[qs_ib[3] | ((qh_b << 2) & 0x300u)]),
      static_cast<int64_t>(iq2s_grid[qs_ib[2] | ((qh_b << 4) & 0x300u)]),
      static_cast<int64_t>(iq2s_grid[qs_ib[1] | ((qh_b << 6) & 0x300u)]),
      static_cast<int64_t>(iq2s_grid[qs_ib[0] | ((qh_b << 8) & 0x300u)]));
  const __m256i signs = _mm256_set_epi64x(
      static_cast<int64_t>(kSignBytes8Bit.mask[signs_ib[3]]),
      static_cast<int64_t>(kSignBytes8Bit.mask[signs_ib[2]]),
      static_cast<int64_t>(kSignBytes8Bit.mask[signs_ib[1]]),
      static_cast<int64_t>(kSignBytes8Bit.mask[signs_ib[0]]));
  const __m256i ls_v = _mm256_set_m128i(
      _mm_set1_epi16(static_cast<int16_t>(2 * (sc >> 4) + 1)),
      _mm_set1_epi16(static_cast<int16_t>(2 * (sc & 0xF) + 1)));
  const __m256i p32_a = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qa, signs)), ls_v);
  const __m256i p32_b = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qb, signs)), ls_v);
  acc_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_a), _mm256_set1_ps(d_a), acc_a);
  acc_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_b), _mm256_set1_ps(d_b), acc_b);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq3Xxs32Q8Batch2Avx2(
    const uint8_t* qs_ib, uint32_t aux32,
    float d_a, __m256i x_qa, __m256& acc_a,
    float d_b, __m256i x_qb, __m256& acc_b) noexcept {
  const __m256i grid_u8 = _mm256_setr_epi32(
      static_cast<int>(iq3xxs_grid[qs_ib[0]]),
      static_cast<int>(iq3xxs_grid[qs_ib[1]]),
      static_cast<int>(iq3xxs_grid[qs_ib[2]]),
      static_cast<int>(iq3xxs_grid[qs_ib[3]]),
      static_cast<int>(iq3xxs_grid[qs_ib[4]]),
      static_cast<int>(iq3xxs_grid[qs_ib[5]]),
      static_cast<int>(iq3xxs_grid[qs_ib[6]]),
      static_cast<int>(iq3xxs_grid[qs_ib[7]]));
  const __m256i signs = _mm256_set_epi64x(
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 21) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 14) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 7) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 0) & 127u]));
  const __m256i ls_v = _mm256_set1_epi16(static_cast<int16_t>(2 * (aux32 >> 28) + 1));
  const __m256i p32_a = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qa, signs)), ls_v);
  const __m256i p32_b = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qb, signs)), ls_v);
  acc_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_a), _mm256_set1_ps(d_a), acc_a);
  acc_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_b), _mm256_set1_ps(d_b), acc_b);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq4Nl32Q8Batch2Avx2(
    const uint8_t* qs16, __m256i lut_iq4nl_256, __m256i mask4_256, int16_t sub_scale_i16,
    __m256i ax_a, __m256i x_qa, float scale_a, __m256& acc_a,
    __m256i ax_b, __m256i x_qb, float scale_b, __m256& acc_b) noexcept {
  const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs16));
  const __m256i nibbles = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw16, 4), raw16), mask4_256);
  const __m256i w_i8 = _mm256_shuffle_epi8(lut_iq4nl_256, nibbles);
  const __m256i ls_v = _mm256_set1_epi16(sub_scale_i16);
  const __m256i p32_a = _mm256_madd_epi16(_mm256_maddubs_epi16(ax_a, _mm256_sign_epi8(w_i8, x_qa)), ls_v);
  const __m256i p32_b = _mm256_madd_epi16(_mm256_maddubs_epi16(ax_b, _mm256_sign_epi8(w_i8, x_qb)), ls_v);
  acc_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_a), _mm256_set1_ps(scale_a), acc_a);
  acc_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_b), _mm256_set1_ps(scale_b), acc_b);
}

__attribute__((target("avx2,fma,f16c"))) inline __m256i UnpackQ2_0_32Avx2(
    const uint8_t* qs8, __m256i mask2_256) noexcept {
  const __m256i v0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs8)));
  const __m256i v1 = _mm256_or_si256(v0, _mm256_slli_epi32(v0, 12));
  const __m256i v2 = _mm256_or_si256(v1, _mm256_slli_epi32(v1, 6));
  return _mm256_and_si256(v2, mask2_256);
}

__attribute__((target("avx2,fma,f16c"))) inline __m256i UnpackQ5_0_32Avx2(
    const uint8_t* blk, __m256i mask4_256) noexcept {
  const __m256i hi_bits = _mm256_set_epi64x(
      static_cast<int64_t>(kHighBit16Lut.mask[blk[5]]),
      static_cast<int64_t>(kHighBit16Lut.mask[blk[4]]),
      static_cast<int64_t>(kHighBit16Lut.mask[blk[3]]),
      static_cast<int64_t>(kHighBit16Lut.mask[blk[2]]));
  const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 6));
  const __m256i nibbles = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw16, 4), raw16), mask4_256);
  return _mm256_or_si256(nibbles, hi_bits);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumQ5KPair64Q8Avx2(
    const uint8_t* blk,
    int p,
    __m256i qh_v,
    __m256i x_q0,
    float qd0,
    int32_t qbsum0,
    __m256i x_q1,
    float qd1,
    int32_t qbsum1,
    float d,
    float min,
    __m256& acc,
    float& bsum_acc) noexcept {
  uint8_t sc0 = 0, m0 = 0, sc1 = 0, m1 = 0;
  GetScaleMinK4Cpu(2 * p + 0, blk + 4, sc0, m0);
  GetScaleMinK4Cpu(2 * p + 1, blk + 4, sc1, m1);
  const __m256i ql_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 48 + p * 32));
  const __m256i mask4 = _mm256_set1_epi8(0x0F);
  const __m256i bit16 = _mm256_set1_epi8(0x10);
  const __m256i zero = _mm256_setzero_si256();
  const __m256i ones_16 = _mm256_set1_epi16(1);
  const __m256i u1_v = _mm256_set1_epi8(static_cast<char>(1u << (2 * p)));
  const __m256i u2_v = _mm256_set1_epi8(static_cast<char>(2u << (2 * p)));
  const __m256i hb0 = _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh_v, u1_v), zero), bit16);
  const __m256i hb1 = _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh_v, u2_v), zero), bit16);
  const __m256i q_u8_0 = _mm256_or_si256(_mm256_and_si256(ql_v, mask4), hb0);
  const __m256i q_u8_1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql_v, 4), mask4), hb1);
  const __m256i p32_0 = _mm256_madd_epi16(_mm256_maddubs_epi16(q_u8_0, x_q0), ones_16);
  const __m256i p32_1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q_u8_1, x_q1), ones_16);
  const float s0 = (d * static_cast<float>(sc0)) * qd0;
  const float s1 = (d * static_cast<float>(sc1)) * qd1;
  acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_0), _mm256_set1_ps(s0), acc);
  acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_1), _mm256_set1_ps(s1), acc);
  bsum_acc = std::fma((min * static_cast<float>(m0)) * qd0, static_cast<float>(qbsum0), bsum_acc);
  bsum_acc = std::fma((min * static_cast<float>(m1)) * qd1, static_cast<float>(qbsum1), bsum_acc);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumQ6KHalf128Q8Avx2(
    const uint8_t* ql_h,
    const uint8_t* qh_h,
    const int8_t* sc_h,
    float d,
    const GgufQ8Block* q8_h,
    const __m256i (&x_q)[4],
    const __m256i (&ax_u8)[4],
    __m256& acc) noexcept {
  const __m256i mask4 = _mm256_set1_epi8(0x0F);
  const __m256i mask30 = _mm256_set1_epi8(0x30);
  const __m256i off32 = _mm256_set1_epi8(32);
  const __m256i qh_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh_h));
  const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql_h + 0));
  const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql_h + 32));
  const __m256i q_i8[4] = {
      _mm256_sub_epi8(
          _mm256_or_si256(_mm256_and_si256(ql0, mask4), _mm256_and_si256(_mm256_slli_epi16(qh_v, 4), mask30)), off32),
      _mm256_sub_epi8(
          _mm256_or_si256(_mm256_and_si256(ql1, mask4), _mm256_and_si256(_mm256_slli_epi16(qh_v, 2), mask30)), off32),
      _mm256_sub_epi8(
          _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask4), _mm256_and_si256(qh_v, mask30)), off32),
      _mm256_sub_epi8(
          _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask4),
                          _mm256_and_si256(_mm256_srli_epi16(qh_v, 2), mask30)),
          off32)};
  for (int s = 0; s < 4; ++s) {
    const __m256i p16 = _mm256_maddubs_epi16(ax_u8[s], _mm256_sign_epi8(q_i8[s], x_q[s]));
    const __m128i sc_lo = _mm_set1_epi16(static_cast<int16_t>(sc_h[2 * s + 0]));
    const __m128i sc_hi = _mm_set1_epi16(static_cast<int16_t>(sc_h[2 * s + 1]));
    const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set_m128i(sc_hi, sc_lo));
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(d * q8_h[s].d), acc);
  }
}

__attribute__((target("avx2,fma,f16c"))) inline float VecDotGgufRowQ8Avx2(
    int quant_type, const uint8_t* weight_row, const GgufQ8Block* x_q8, int64_t K) {
  __m256 acc = _mm256_setzero_ps();

  switch (quant_type) {
    case kGgmlTypeIQ2_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 66;
        const float d_0125 = Fp16ToFloatF16c(ReadU16Le(blk)) * 0.125f;
        const uint8_t* qs = blk + 2;
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8[ib32].qs));
          AccumIq2Xxs32Q8Avx2(qs + 8 * ib32, d_0125 * q8[ib32].d, x_q_32, acc);
        }
      }
      return HsumAvx2(acc);
    }
    case kGgmlTypeIQ2_XS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 74;
        const float d_0125 = Fp16ToFloatF16c(ReadU16Le(blk)) * 0.125f;
        const uint8_t* qs = blk + 2;
        const uint8_t* sc = blk + 66;
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8[ib32].qs));
          AccumIq2Xs32Q8Avx2(qs + 8 * ib32, sc[ib32], d_0125 * q8[ib32].d, x_q_32, acc);
        }
      }
      return HsumAvx2(acc);
    }
    case kGgmlTypeIQ2_S: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 82;
        const float d_0125 = Fp16ToFloatF16c(ReadU16Le(blk)) * 0.125f;
        const uint8_t* qs = blk + 2;
        const uint8_t* signs = qs + 32;
        const uint8_t* qh = blk + 66;
        const uint8_t* sc = blk + 74;
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8[ib32].qs));
          AccumIq2S32Q8Avx2(qs + 4 * ib32, signs + 4 * ib32, qh[ib32], sc[ib32], d_0125 * q8[ib32].d, x_q_32, acc);
        }
      }
      return HsumAvx2(acc);
    }
    case kGgmlTypeIQ3_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 98;
        const float d_025 = Fp16ToFloatF16c(ReadU16Le(blk)) * 0.25f;
        const uint8_t* qs = blk + 2;
        const uint8_t* ss = qs + 64;
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8[ib32].qs));
          AccumIq3Xxs32Q8Avx2(qs + 8 * ib32, ReadU32Le(ss + 4 * ib32), d_025 * q8[ib32].d, x_q_32, acc);
        }
      }
      return HsumAvx2(acc);
    }
    case kGgmlTypeIQ4_NL: {
      const int64_t nb = K / 32;
      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
      const __m256i lut_256 = _mm256_set_m128i(lut128, lut128);
      const __m256i mask4_256 = _mm256_set1_epi8(0x0F);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 18;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk));
        const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8[i].qs));
        const __m256i ax_u8 = _mm256_abs_epi8(x_q_32);
        AccumIq4Nl32Q8Avx2(blk + 2, lut_256, mask4_256, ax_u8, x_q_32, 1, d * x_q8[i].d, acc);
      }
      return HsumAvx2(acc);
    }
    case kGgmlTypeIQ4_XS: {
      const int64_t nb = K / 256;
      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
      const __m256i lut_256 = _mm256_set_m128i(lut128, lut128);
      const __m256i mask4_256 = _mm256_set1_epi8(0x0F);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 136;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk));
        const uint16_t sh = ReadU16Le(blk + 2);
        const uint8_t* sl = blk + 4;
        const uint8_t* qs = blk + 8;
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int ib = 0; ib < 8; ++ib) {
          const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
          const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8[ib].qs));
          const __m256i ax_u8 = _mm256_abs_epi8(x_q_32);
          AccumIq4Nl32Q8Avx2(
              qs + ib * 16, lut_256, mask4_256, ax_u8, x_q_32,
              static_cast<int16_t>(ls - 32), d * q8[ib].d, acc);
        }
      }
      return HsumAvx2(acc);
    }
    case kGgmlTypeQ2_0: {
      const int64_t nb = K / 64;
      const __m256i mask2_256 = _mm256_set1_epi8(0x03);
      const __m256i ones_16 = _mm256_set1_epi16(1);
      float bsum_acc = 0.0f;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 18;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk));
        const GgufQ8Block& q8_0 = x_q8[2 * i + 0];
        const GgufQ8Block& q8_1 = x_q8[2 * i + 1];
        const __m256i x_q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_0.qs));
        const __m256i x_q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_1.qs));
        const __m256i p32_0 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(UnpackQ2_0_32Avx2(blk + 2, mask2_256), x_q0), ones_16);
        const __m256i p32_1 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(UnpackQ2_0_32Avx2(blk + 10, mask2_256), x_q1), ones_16);
        const float s0 = d * q8_0.d;
        const float s1 = d * q8_1.d;
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_0), _mm256_set1_ps(s0), acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_1), _mm256_set1_ps(s1), acc);
        bsum_acc = std::fma(s0, static_cast<float>(q8_0.bsum), bsum_acc);
        bsum_acc = std::fma(s1, static_cast<float>(q8_1.bsum), bsum_acc);
      }
      return HsumAvx2(acc) - bsum_acc;
    }
    case kGgmlTypeQ5_0: {
      const int64_t nb = K / 32;
      const __m256i mask4_256 = _mm256_set1_epi8(0x0F);
      const __m256i ones_16 = _mm256_set1_epi16(1);
      float bsum_acc = 0.0f;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 22;
        const float s = Fp16ToFloatF16c(ReadU16Le(blk)) * x_q8[i].d;
        const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8[i].qs));
        const __m256i q_u8 = UnpackQ5_0_32Avx2(blk, mask4_256);
        const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q_u8, x_q_32), ones_16);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(s), acc);
        bsum_acc = std::fma(s, static_cast<float>(x_q8[i].bsum), bsum_acc);
      }
      return HsumAvx2(acc) - 16.0f * bsum_acc;
    }
    case kGgmlTypeQ4_0: {
      const int64_t nb = K / 32;
      const __m256i mask4_256 = _mm256_set1_epi8(0x0F);
      const __m256i ones_16 = _mm256_set1_epi16(1);
      float bsum_acc = 0.0f;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 18;
        const float s = Fp16ToFloatF16c(ReadU16Le(blk)) * x_q8[i].d;
        const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
        const __m256i q_u8 = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw16, 4), raw16), mask4_256);
        const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8[i].qs));
        const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q_u8, x_q_32), ones_16);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(s), acc);
        bsum_acc = std::fma(s, static_cast<float>(x_q8[i].bsum), bsum_acc);
      }
      return HsumAvx2(acc) - 8.0f * bsum_acc;
    }
    case kGgmlTypeQ8_0: {
      const int64_t nb = K / 32;
      const __m256i ones_16 = _mm256_set1_epi16(1);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 34;
        const float s = Fp16ToFloatF16c(ReadU16Le(blk)) * x_q8[i].d;
        const __m256i w_i8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 2));
        const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8[i].qs));
        const __m256i p32 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(_mm256_abs_epi8(w_i8), _mm256_sign_epi8(x_q_32, w_i8)), ones_16);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(s), acc);
      }
      return HsumAvx2(acc);
    }
    case kGgmlTypeQ5_K: {
      const int64_t nb = K / 256;
      float bsum_acc = 0.0f;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 176;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk));
        const float min = Fp16ToFloatF16c(ReadU16Le(blk + 2));
        const __m256i qh_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 16));
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int p = 0; p < 4; ++p) {
          const GgufQ8Block& q8_0 = q8[2 * p + 0];
          const GgufQ8Block& q8_1 = q8[2 * p + 1];
          const __m256i x_q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_0.qs));
          const __m256i x_q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_1.qs));
          AccumQ5KPair64Q8Avx2(
              blk, p, qh_v, x_q0, q8_0.d, q8_0.bsum, x_q1, q8_1.d, q8_1.bsum, d, min, acc, bsum_acc);
        }
      }
      return HsumAvx2(acc) - bsum_acc;
    }
    case kGgmlTypeQ6_K: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* blk = weight_row + i * 210;
        const float d = Fp16ToFloatF16c(ReadU16Le(blk + 208));
        const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int half = 0; half < 2; ++half) {
          const GgufQ8Block* q8_h = q8 + half * 4;
          __m256i x_q[4];
          __m256i ax_u8[4];
          for (int s = 0; s < 4; ++s) {
            x_q[s] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_h[s].qs));
            ax_u8[s] = _mm256_abs_epi8(x_q[s]);
          }
          AccumQ6KHalf128Q8Avx2(
              blk + half * 64, blk + 128 + half * 32, sc + half * 8, d, q8_h, x_q, ax_u8, acc);
        }
      }
      return HsumAvx2(acc);
    }
    default: {
      int64_t block_elems = 0;
      int64_t block_bytes = 0;
      if (!GgmlTypeBlockInfo(quant_type, block_elems, block_bytes) || block_elems <= 0) {
        return 0.0f;
      }
      const int64_t chunk_elems = (block_elems < 32) ? 32 : block_elems;
      const int64_t chunk_bytes = (chunk_elems / block_elems) * block_bytes;
      alignas(32) float buf[256];
      const int64_t n_chunks = K / chunk_elems;
      for (int64_t c = 0; c < n_chunks; ++c) {
        DequantizeGgufRowCpu(weight_row + c * chunk_bytes, quant_type, chunk_elems, buf);
        for (int64_t b = 0; b < chunk_elems / 32; ++b) {
          const GgufQ8Block& qb = x_q8[(c * chunk_elems) / 32 + b];
          const __m256 vd = _mm256_set1_ps(qb.d);
          const __m128i q_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qb.qs + 0));
          const __m128i q_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qb.qs + 16));
          const float* bp = buf + b * 32;
          acc = _mm256_fmadd_ps(
              _mm256_load_ps(bp + 0),
              _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo)), vd), acc);
          acc = _mm256_fmadd_ps(
              _mm256_load_ps(bp + 8),
              _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(q_lo, 8))), vd), acc);
          acc = _mm256_fmadd_ps(
              _mm256_load_ps(bp + 16),
              _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi)), vd), acc);
          acc = _mm256_fmadd_ps(
              _mm256_load_ps(bp + 24),
              _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(q_hi, 8))), vd), acc);
        }
      }
      return HsumAvx2(acc);
    }
  }
}

__attribute__((target("avx2,fma,f16c"))) inline void VecDotGgufRow4xQ8Avx2(
    int quant_type,
    const uint8_t* row0,
    const uint8_t* row1,
    const uint8_t* row2,
    const uint8_t* row3,
    const GgufQ8Block* x_q8,
    int64_t K,
    float* out4) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  switch (quant_type) {
    case kGgmlTypeIQ2_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 66;
        const uint8_t* b1 = row1 + i * 66;
        const uint8_t* b2 = row2 + i * 66;
        const uint8_t* b3 = row3 + i * 66;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0)) * 0.125f;
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1)) * 0.125f;
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2)) * 0.125f;
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3)) * 0.125f;
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float qd = q8[ib32].d;
          const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8[ib32].qs));
          AccumIq2Xxs32Q8Avx2(b0 + 2 + 8 * ib32, d0 * qd, x_q_32, acc0);
          AccumIq2Xxs32Q8Avx2(b1 + 2 + 8 * ib32, d1 * qd, x_q_32, acc1);
          AccumIq2Xxs32Q8Avx2(b2 + 2 + 8 * ib32, d2 * qd, x_q_32, acc2);
          AccumIq2Xxs32Q8Avx2(b3 + 2 + 8 * ib32, d3 * qd, x_q_32, acc3);
        }
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      return;
    }
    case kGgmlTypeIQ2_XS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 74;
        const uint8_t* b1 = row1 + i * 74;
        const uint8_t* b2 = row2 + i * 74;
        const uint8_t* b3 = row3 + i * 74;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0)) * 0.125f;
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1)) * 0.125f;
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2)) * 0.125f;
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3)) * 0.125f;
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float qd = q8[ib32].d;
          const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8[ib32].qs));
          AccumIq2Xs32Q8Avx2(b0 + 2 + 8 * ib32, b0[66 + ib32], d0 * qd, x_q_32, acc0);
          AccumIq2Xs32Q8Avx2(b1 + 2 + 8 * ib32, b1[66 + ib32], d1 * qd, x_q_32, acc1);
          AccumIq2Xs32Q8Avx2(b2 + 2 + 8 * ib32, b2[66 + ib32], d2 * qd, x_q_32, acc2);
          AccumIq2Xs32Q8Avx2(b3 + 2 + 8 * ib32, b3[66 + ib32], d3 * qd, x_q_32, acc3);
        }
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      return;
    }
    case kGgmlTypeIQ2_S: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 82;
        const uint8_t* b1 = row1 + i * 82;
        const uint8_t* b2 = row2 + i * 82;
        const uint8_t* b3 = row3 + i * 82;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0)) * 0.125f;
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1)) * 0.125f;
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2)) * 0.125f;
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3)) * 0.125f;
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float qd = q8[ib32].d;
          const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8[ib32].qs));
          AccumIq2S32Q8Avx2(b0 + 2 + 4 * ib32, b0 + 34 + 4 * ib32, b0[66 + ib32], b0[74 + ib32], d0 * qd, x_q_32, acc0);
          AccumIq2S32Q8Avx2(b1 + 2 + 4 * ib32, b1 + 34 + 4 * ib32, b1[66 + ib32], b1[74 + ib32], d1 * qd, x_q_32, acc1);
          AccumIq2S32Q8Avx2(b2 + 2 + 4 * ib32, b2 + 34 + 4 * ib32, b2[66 + ib32], b2[74 + ib32], d2 * qd, x_q_32, acc2);
          AccumIq2S32Q8Avx2(b3 + 2 + 4 * ib32, b3 + 34 + 4 * ib32, b3[66 + ib32], b3[74 + ib32], d3 * qd, x_q_32, acc3);
        }
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      return;
    }
    case kGgmlTypeIQ3_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 98;
        const uint8_t* b1 = row1 + i * 98;
        const uint8_t* b2 = row2 + i * 98;
        const uint8_t* b3 = row3 + i * 98;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0)) * 0.25f;
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1)) * 0.25f;
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2)) * 0.25f;
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3)) * 0.25f;
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float qd = q8[ib32].d;
          const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8[ib32].qs));
          AccumIq3Xxs32Q8Avx2(b0 + 2 + 8 * ib32, ReadU32Le(b0 + 66 + 4 * ib32), d0 * qd, x_q_32, acc0);
          AccumIq3Xxs32Q8Avx2(b1 + 2 + 8 * ib32, ReadU32Le(b1 + 66 + 4 * ib32), d1 * qd, x_q_32, acc1);
          AccumIq3Xxs32Q8Avx2(b2 + 2 + 8 * ib32, ReadU32Le(b2 + 66 + 4 * ib32), d2 * qd, x_q_32, acc2);
          AccumIq3Xxs32Q8Avx2(b3 + 2 + 8 * ib32, ReadU32Le(b3 + 66 + 4 * ib32), d3 * qd, x_q_32, acc3);
        }
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      return;
    }
    case kGgmlTypeIQ4_NL: {
      const int64_t nb = K / 32;
      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
      const __m256i lut_256 = _mm256_set_m128i(lut128, lut128);
      const __m256i mask4_256 = _mm256_set1_epi8(0x0F);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 18;
        const uint8_t* b1 = row1 + i * 18;
        const uint8_t* b2 = row2 + i * 18;
        const uint8_t* b3 = row3 + i * 18;
        const float qd = x_q8[i].d;
        const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8[i].qs));
        const __m256i ax_u8 = _mm256_abs_epi8(x_q_32);
        AccumIq4Nl32Q8Avx2(b0 + 2, lut_256, mask4_256, ax_u8, x_q_32, 1, Fp16ToFloatF16c(ReadU16Le(b0)) * qd, acc0);
        AccumIq4Nl32Q8Avx2(b1 + 2, lut_256, mask4_256, ax_u8, x_q_32, 1, Fp16ToFloatF16c(ReadU16Le(b1)) * qd, acc1);
        AccumIq4Nl32Q8Avx2(b2 + 2, lut_256, mask4_256, ax_u8, x_q_32, 1, Fp16ToFloatF16c(ReadU16Le(b2)) * qd, acc2);
        AccumIq4Nl32Q8Avx2(b3 + 2, lut_256, mask4_256, ax_u8, x_q_32, 1, Fp16ToFloatF16c(ReadU16Le(b3)) * qd, acc3);
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      return;
    }
    case kGgmlTypeIQ4_XS: {
      const int64_t nb = K / 256;
      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
      const __m256i lut_256 = _mm256_set_m128i(lut128, lut128);
      const __m256i mask4_256 = _mm256_set1_epi8(0x0F);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 136;
        const uint8_t* b1 = row1 + i * 136;
        const uint8_t* b2 = row2 + i * 136;
        const uint8_t* b3 = row3 + i * 136;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0));
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1));
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2));
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3));
        const uint16_t sh0 = ReadU16Le(b0 + 2);
        const uint16_t sh1 = ReadU16Le(b1 + 2);
        const uint16_t sh2 = ReadU16Le(b2 + 2);
        const uint16_t sh3 = ReadU16Le(b3 + 2);
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int ib = 0; ib < 8; ++ib) {
          const int ls0 = ((b0[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh0 >> (2 * ib)) & 3) << 4);
          const int ls1 = ((b1[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh1 >> (2 * ib)) & 3) << 4);
          const int ls2 = ((b2[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh2 >> (2 * ib)) & 3) << 4);
          const int ls3 = ((b3[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh3 >> (2 * ib)) & 3) << 4);
          const float qd = q8[ib].d;
          const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8[ib].qs));
          const __m256i ax_u8 = _mm256_abs_epi8(x_q_32);
          AccumIq4Nl32Q8Avx2(b0 + 8 + ib * 16, lut_256, mask4_256, ax_u8, x_q_32, static_cast<int16_t>(ls0 - 32), d0 * qd, acc0);
          AccumIq4Nl32Q8Avx2(b1 + 8 + ib * 16, lut_256, mask4_256, ax_u8, x_q_32, static_cast<int16_t>(ls1 - 32), d1 * qd, acc1);
          AccumIq4Nl32Q8Avx2(b2 + 8 + ib * 16, lut_256, mask4_256, ax_u8, x_q_32, static_cast<int16_t>(ls2 - 32), d2 * qd, acc2);
          AccumIq4Nl32Q8Avx2(b3 + 8 + ib * 16, lut_256, mask4_256, ax_u8, x_q_32, static_cast<int16_t>(ls3 - 32), d3 * qd, acc3);
        }
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      return;
    }
    case kGgmlTypeQ2_0: {
      const int64_t nb = K / 64;
      const __m256i mask2_256 = _mm256_set1_epi8(0x03);
      const __m256i ones_16 = _mm256_set1_epi16(1);
      __m128 bsum_acc4 = _mm_setzero_ps();
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 18;
        const uint8_t* b1 = row1 + i * 18;
        const uint8_t* b2 = row2 + i * 18;
        const uint8_t* b3 = row3 + i * 18;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0));
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1));
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2));
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3));
        const __m128 d4_v = _mm_setr_ps(d0, d1, d2, d3);

        const GgufQ8Block& q8_0 = x_q8[2 * i + 0];
        const GgufQ8Block& q8_1 = x_q8[2 * i + 1];
        const __m256i x_q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_0.qs));
        const __m256i x_q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_1.qs));

        acc0 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(UnpackQ2_0_32Avx2(b0 + 2, mask2_256), x_q0), ones_16)),
            _mm256_set1_ps(d0 * q8_0.d), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(UnpackQ2_0_32Avx2(b1 + 2, mask2_256), x_q0), ones_16)),
            _mm256_set1_ps(d1 * q8_0.d), acc1);
        acc2 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(UnpackQ2_0_32Avx2(b2 + 2, mask2_256), x_q0), ones_16)),
            _mm256_set1_ps(d2 * q8_0.d), acc2);
        acc3 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(UnpackQ2_0_32Avx2(b3 + 2, mask2_256), x_q0), ones_16)),
            _mm256_set1_ps(d3 * q8_0.d), acc3);

        acc0 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(UnpackQ2_0_32Avx2(b0 + 10, mask2_256), x_q1), ones_16)),
            _mm256_set1_ps(d0 * q8_1.d), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(UnpackQ2_0_32Avx2(b1 + 10, mask2_256), x_q1), ones_16)),
            _mm256_set1_ps(d1 * q8_1.d), acc1);
        acc2 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(UnpackQ2_0_32Avx2(b2 + 10, mask2_256), x_q1), ones_16)),
            _mm256_set1_ps(d2 * q8_1.d), acc2);
        acc3 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(UnpackQ2_0_32Avx2(b3 + 10, mask2_256), x_q1), ones_16)),
            _mm256_set1_ps(d3 * q8_1.d), acc3);

        const float qb = q8_0.d * static_cast<float>(q8_0.bsum) + q8_1.d * static_cast<float>(q8_1.bsum);
        bsum_acc4 = _mm_fmadd_ps(d4_v, _mm_set1_ps(qb), bsum_acc4);
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      _mm_storeu_ps(out4, _mm_sub_ps(_mm_loadu_ps(out4), bsum_acc4));
      return;
    }
    case kGgmlTypeQ5_0: {
      const int64_t nb = K / 32;
      const __m256i mask4_256 = _mm256_set1_epi8(0x0F);
      const __m256i ones_16 = _mm256_set1_epi16(1);
      __m128 bsum_acc4 = _mm_setzero_ps();
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 22;
        const uint8_t* b1 = row1 + i * 22;
        const uint8_t* b2 = row2 + i * 22;
        const uint8_t* b3 = row3 + i * 22;
        const float qd = x_q8[i].d;
        const float s0 = Fp16ToFloatF16c(ReadU16Le(b0)) * qd;
        const float s1 = Fp16ToFloatF16c(ReadU16Le(b1)) * qd;
        const float s2 = Fp16ToFloatF16c(ReadU16Le(b2)) * qd;
        const float s3 = Fp16ToFloatF16c(ReadU16Le(b3)) * qd;
        const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8[i].qs));
        acc0 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(UnpackQ5_0_32Avx2(b0, mask4_256), x_q_32), ones_16)),
            _mm256_set1_ps(s0), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(UnpackQ5_0_32Avx2(b1, mask4_256), x_q_32), ones_16)),
            _mm256_set1_ps(s1), acc1);
        acc2 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(UnpackQ5_0_32Avx2(b2, mask4_256), x_q_32), ones_16)),
            _mm256_set1_ps(s2), acc2);
        acc3 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(UnpackQ5_0_32Avx2(b3, mask4_256), x_q_32), ones_16)),
            _mm256_set1_ps(s3), acc3);
        bsum_acc4 = _mm_fmadd_ps(
            _mm_setr_ps(s0, s1, s2, s3), _mm_set1_ps(static_cast<float>(x_q8[i].bsum)), bsum_acc4);
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      _mm_storeu_ps(out4, _mm_sub_ps(_mm_loadu_ps(out4), _mm_mul_ps(bsum_acc4, _mm_set1_ps(16.0f))));
      return;
    }
    case kGgmlTypeQ4_0: {
      const int64_t nb = K / 32;
      const __m256i mask4_256 = _mm256_set1_epi8(0x0F);
      const __m256i ones_16 = _mm256_set1_epi16(1);
      __m128 bsum_acc4 = _mm_setzero_ps();
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 18;
        const uint8_t* b1 = row1 + i * 18;
        const uint8_t* b2 = row2 + i * 18;
        const uint8_t* b3 = row3 + i * 18;
        const float qd = x_q8[i].d;
        const float s0 = Fp16ToFloatF16c(ReadU16Le(b0)) * qd;
        const float s1 = Fp16ToFloatF16c(ReadU16Le(b1)) * qd;
        const float s2 = Fp16ToFloatF16c(ReadU16Le(b2)) * qd;
        const float s3 = Fp16ToFloatF16c(ReadU16Le(b3)) * qd;
        const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8[i].qs));
        const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b0 + 2));
        const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b1 + 2));
        const __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b2 + 2));
        const __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b3 + 2));
        acc0 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(r0, 4), r0), mask4_256), x_q_32), ones_16)),
            _mm256_set1_ps(s0), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(r1, 4), r1), mask4_256), x_q_32), ones_16)),
            _mm256_set1_ps(s1), acc1);
        acc2 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(r2, 4), r2), mask4_256), x_q_32), ones_16)),
            _mm256_set1_ps(s2), acc2);
        acc3 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(r3, 4), r3), mask4_256), x_q_32), ones_16)),
            _mm256_set1_ps(s3), acc3);
        bsum_acc4 = _mm_fmadd_ps(
            _mm_setr_ps(s0, s1, s2, s3), _mm_set1_ps(static_cast<float>(x_q8[i].bsum)), bsum_acc4);
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      _mm_storeu_ps(out4, _mm_sub_ps(_mm_loadu_ps(out4), _mm_mul_ps(bsum_acc4, _mm_set1_ps(8.0f))));
      return;
    }
    case kGgmlTypeQ8_0: {
      const int64_t nb = K / 32;
      const __m256i ones_16 = _mm256_set1_epi16(1);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 34;
        const uint8_t* b1 = row1 + i * 34;
        const uint8_t* b2 = row2 + i * 34;
        const uint8_t* b3 = row3 + i * 34;
        const float qd = x_q8[i].d;
        const __m256i x_q_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8[i].qs));
        const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0 + 2));
        const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1 + 2));
        const __m256i w2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b2 + 2));
        const __m256i w3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b3 + 2));
        acc0 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w0), _mm256_sign_epi8(x_q_32, w0)), ones_16)),
            _mm256_set1_ps(Fp16ToFloatF16c(ReadU16Le(b0)) * qd), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w1), _mm256_sign_epi8(x_q_32, w1)), ones_16)),
            _mm256_set1_ps(Fp16ToFloatF16c(ReadU16Le(b1)) * qd), acc1);
        acc2 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w2), _mm256_sign_epi8(x_q_32, w2)), ones_16)),
            _mm256_set1_ps(Fp16ToFloatF16c(ReadU16Le(b2)) * qd), acc2);
        acc3 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w3), _mm256_sign_epi8(x_q_32, w3)), ones_16)),
            _mm256_set1_ps(Fp16ToFloatF16c(ReadU16Le(b3)) * qd), acc3);
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      return;
    }
    case kGgmlTypeQ5_K: {
      const int64_t nb = K / 256;
      float bsum0 = 0.0f, bsum1 = 0.0f, bsum2 = 0.0f, bsum3 = 0.0f;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 176;
        const uint8_t* b1 = row1 + i * 176;
        const uint8_t* b2 = row2 + i * 176;
        const uint8_t* b3 = row3 + i * 176;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0));
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1));
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2));
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3));
        const float m0 = Fp16ToFloatF16c(ReadU16Le(b0 + 2));
        const float m1 = Fp16ToFloatF16c(ReadU16Le(b1 + 2));
        const float m2 = Fp16ToFloatF16c(ReadU16Le(b2 + 2));
        const float m3 = Fp16ToFloatF16c(ReadU16Le(b3 + 2));
        const __m256i qh0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0 + 16));
        const __m256i qh1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1 + 16));
        const __m256i qh2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b2 + 16));
        const __m256i qh3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b3 + 16));
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int p = 0; p < 4; ++p) {
          const GgufQ8Block& q8_0 = q8[2 * p + 0];
          const GgufQ8Block& q8_1 = q8[2 * p + 1];
          const __m256i x_q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_0.qs));
          const __m256i x_q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_1.qs));
          AccumQ5KPair64Q8Avx2(b0, p, qh0, x_q0, q8_0.d, q8_0.bsum, x_q1, q8_1.d, q8_1.bsum, d0, m0, acc0, bsum0);
          AccumQ5KPair64Q8Avx2(b1, p, qh1, x_q0, q8_0.d, q8_0.bsum, x_q1, q8_1.d, q8_1.bsum, d1, m1, acc1, bsum1);
          AccumQ5KPair64Q8Avx2(b2, p, qh2, x_q0, q8_0.d, q8_0.bsum, x_q1, q8_1.d, q8_1.bsum, d2, m2, acc2, bsum2);
          AccumQ5KPair64Q8Avx2(b3, p, qh3, x_q0, q8_0.d, q8_0.bsum, x_q1, q8_1.d, q8_1.bsum, d3, m3, acc3, bsum3);
        }
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      out4[0] -= bsum0;
      out4[1] -= bsum1;
      out4[2] -= bsum2;
      out4[3] -= bsum3;
      return;
    }
    case kGgmlTypeQ6_K: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 210;
        const uint8_t* b1 = row1 + i * 210;
        const uint8_t* b2 = row2 + i * 210;
        const uint8_t* b3 = row3 + i * 210;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0 + 208));
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1 + 208));
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2 + 208));
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3 + 208));
        const int8_t* sc0 = reinterpret_cast<const int8_t*>(b0 + 192);
        const int8_t* sc1 = reinterpret_cast<const int8_t*>(b1 + 192);
        const int8_t* sc2 = reinterpret_cast<const int8_t*>(b2 + 192);
        const int8_t* sc3 = reinterpret_cast<const int8_t*>(b3 + 192);
        const GgufQ8Block* q8 = x_q8 + i * 8;
        for (int half = 0; half < 2; ++half) {
          const GgufQ8Block* q8_h = q8 + half * 4;
          __m256i x_q[4];
          __m256i ax_u8[4];
          for (int s = 0; s < 4; ++s) {
            x_q[s] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_h[s].qs));
            ax_u8[s] = _mm256_abs_epi8(x_q[s]);
          }
          AccumQ6KHalf128Q8Avx2(b0 + half * 64, b0 + 128 + half * 32, sc0 + half * 8, d0, q8_h, x_q, ax_u8, acc0);
          AccumQ6KHalf128Q8Avx2(b1 + half * 64, b1 + 128 + half * 32, sc1 + half * 8, d1, q8_h, x_q, ax_u8, acc1);
          AccumQ6KHalf128Q8Avx2(b2 + half * 64, b2 + 128 + half * 32, sc2 + half * 8, d2, q8_h, x_q, ax_u8, acc2);
          AccumQ6KHalf128Q8Avx2(b3 + half * 64, b3 + 128 + half * 32, sc3 + half * 8, d3, q8_h, x_q, ax_u8, acc3);
        }
      }
      Hsum4Avx2(acc0, acc1, acc2, acc3, out4);
      return;
    }
    default: {
      out4[0] = VecDotGgufRowQ8Avx2(quant_type, row0, x_q8, K);
      out4[1] = VecDotGgufRowQ8Avx2(quant_type, row1, x_q8, K);
      out4[2] = VecDotGgufRowQ8Avx2(quant_type, row2, x_q8, K);
      out4[3] = VecDotGgufRowQ8Avx2(quant_type, row3, x_q8, K);
      return;
    }
  }
}

__attribute__((target("avx2,fma,f16c"))) inline void VecDotGgufRow4xQ8Batch2Avx2(
    int quant_type,
    const uint8_t* row0,
    const uint8_t* row1,
    const uint8_t* row2,
    const uint8_t* row3,
    const GgufQ8Block* x_qa,
    const GgufQ8Block* x_qb,
    int64_t K,
    float* out4_a,
    float* out4_b) {
  __m256 acc0_a = _mm256_setzero_ps();
  __m256 acc1_a = _mm256_setzero_ps();
  __m256 acc2_a = _mm256_setzero_ps();
  __m256 acc3_a = _mm256_setzero_ps();
  __m256 acc0_b = _mm256_setzero_ps();
  __m256 acc1_b = _mm256_setzero_ps();
  __m256 acc2_b = _mm256_setzero_ps();
  __m256 acc3_b = _mm256_setzero_ps();

  switch (quant_type) {
    case kGgmlTypeIQ2_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 66;
        const uint8_t* b1 = row1 + i * 66;
        const uint8_t* b2 = row2 + i * 66;
        const uint8_t* b3 = row3 + i * 66;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0)) * 0.125f;
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1)) * 0.125f;
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2)) * 0.125f;
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3)) * 0.125f;
        const GgufQ8Block* qa = x_qa + i * 8;
        const GgufQ8Block* qb = x_qb + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float qda = qa[ib32].d;
          const float qdb = qb[ib32].d;
          const __m256i x_qa_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa[ib32].qs));
          const __m256i x_qb_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qb[ib32].qs));
          AccumIq2Xxs32Q8Batch2Avx2(b0 + 2 + 8 * ib32, d0 * qda, x_qa_32, acc0_a, d0 * qdb, x_qb_32, acc0_b);
          AccumIq2Xxs32Q8Batch2Avx2(b1 + 2 + 8 * ib32, d1 * qda, x_qa_32, acc1_a, d1 * qdb, x_qb_32, acc1_b);
          AccumIq2Xxs32Q8Batch2Avx2(b2 + 2 + 8 * ib32, d2 * qda, x_qa_32, acc2_a, d2 * qdb, x_qb_32, acc2_b);
          AccumIq2Xxs32Q8Batch2Avx2(b3 + 2 + 8 * ib32, d3 * qda, x_qa_32, acc3_a, d3 * qdb, x_qb_32, acc3_b);
        }
      }
      Hsum4Avx2(acc0_a, acc1_a, acc2_a, acc3_a, out4_a);
      Hsum4Avx2(acc0_b, acc1_b, acc2_b, acc3_b, out4_b);
      return;
    }
    case kGgmlTypeIQ2_XS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 74;
        const uint8_t* b1 = row1 + i * 74;
        const uint8_t* b2 = row2 + i * 74;
        const uint8_t* b3 = row3 + i * 74;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0)) * 0.125f;
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1)) * 0.125f;
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2)) * 0.125f;
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3)) * 0.125f;
        const GgufQ8Block* qa = x_qa + i * 8;
        const GgufQ8Block* qb = x_qb + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float qda = qa[ib32].d;
          const float qdb = qb[ib32].d;
          const __m256i x_qa_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa[ib32].qs));
          const __m256i x_qb_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qb[ib32].qs));
          AccumIq2Xs32Q8Batch2Avx2(b0 + 2 + 8 * ib32, b0[66 + ib32], d0 * qda, x_qa_32, acc0_a, d0 * qdb, x_qb_32, acc0_b);
          AccumIq2Xs32Q8Batch2Avx2(b1 + 2 + 8 * ib32, b1[66 + ib32], d1 * qda, x_qa_32, acc1_a, d1 * qdb, x_qb_32, acc1_b);
          AccumIq2Xs32Q8Batch2Avx2(b2 + 2 + 8 * ib32, b2[66 + ib32], d2 * qda, x_qa_32, acc2_a, d2 * qdb, x_qb_32, acc2_b);
          AccumIq2Xs32Q8Batch2Avx2(b3 + 2 + 8 * ib32, b3[66 + ib32], d3 * qda, x_qa_32, acc3_a, d3 * qdb, x_qb_32, acc3_b);
        }
      }
      Hsum4Avx2(acc0_a, acc1_a, acc2_a, acc3_a, out4_a);
      Hsum4Avx2(acc0_b, acc1_b, acc2_b, acc3_b, out4_b);
      return;
    }
    case kGgmlTypeIQ2_S: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 82;
        const uint8_t* b1 = row1 + i * 82;
        const uint8_t* b2 = row2 + i * 82;
        const uint8_t* b3 = row3 + i * 82;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0)) * 0.125f;
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1)) * 0.125f;
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2)) * 0.125f;
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3)) * 0.125f;
        const GgufQ8Block* qa = x_qa + i * 8;
        const GgufQ8Block* qb = x_qb + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float qda = qa[ib32].d;
          const float qdb = qb[ib32].d;
          const __m256i x_qa_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa[ib32].qs));
          const __m256i x_qb_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qb[ib32].qs));
          AccumIq2S32Q8Batch2Avx2(
              b0 + 2 + 4 * ib32, b0 + 34 + 4 * ib32, b0[66 + ib32], b0[74 + ib32],
              d0 * qda, x_qa_32, acc0_a, d0 * qdb, x_qb_32, acc0_b);
          AccumIq2S32Q8Batch2Avx2(
              b1 + 2 + 4 * ib32, b1 + 34 + 4 * ib32, b1[66 + ib32], b1[74 + ib32],
              d1 * qda, x_qa_32, acc1_a, d1 * qdb, x_qb_32, acc1_b);
          AccumIq2S32Q8Batch2Avx2(
              b2 + 2 + 4 * ib32, b2 + 34 + 4 * ib32, b2[66 + ib32], b2[74 + ib32],
              d2 * qda, x_qa_32, acc2_a, d2 * qdb, x_qb_32, acc2_b);
          AccumIq2S32Q8Batch2Avx2(
              b3 + 2 + 4 * ib32, b3 + 34 + 4 * ib32, b3[66 + ib32], b3[74 + ib32],
              d3 * qda, x_qa_32, acc3_a, d3 * qdb, x_qb_32, acc3_b);
        }
      }
      Hsum4Avx2(acc0_a, acc1_a, acc2_a, acc3_a, out4_a);
      Hsum4Avx2(acc0_b, acc1_b, acc2_b, acc3_b, out4_b);
      return;
    }
    case kGgmlTypeIQ3_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 98;
        const uint8_t* b1 = row1 + i * 98;
        const uint8_t* b2 = row2 + i * 98;
        const uint8_t* b3 = row3 + i * 98;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0)) * 0.25f;
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1)) * 0.25f;
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2)) * 0.25f;
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3)) * 0.25f;
        const GgufQ8Block* qa = x_qa + i * 8;
        const GgufQ8Block* qb = x_qb + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float qda = qa[ib32].d;
          const float qdb = qb[ib32].d;
          const __m256i x_qa_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa[ib32].qs));
          const __m256i x_qb_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qb[ib32].qs));
          AccumIq3Xxs32Q8Batch2Avx2(
              b0 + 2 + 8 * ib32, ReadU32Le(b0 + 66 + 4 * ib32),
              d0 * qda, x_qa_32, acc0_a, d0 * qdb, x_qb_32, acc0_b);
          AccumIq3Xxs32Q8Batch2Avx2(
              b1 + 2 + 8 * ib32, ReadU32Le(b1 + 66 + 4 * ib32),
              d1 * qda, x_qa_32, acc1_a, d1 * qdb, x_qb_32, acc1_b);
          AccumIq3Xxs32Q8Batch2Avx2(
              b2 + 2 + 8 * ib32, ReadU32Le(b2 + 66 + 4 * ib32),
              d2 * qda, x_qa_32, acc2_a, d2 * qdb, x_qb_32, acc2_b);
          AccumIq3Xxs32Q8Batch2Avx2(
              b3 + 2 + 8 * ib32, ReadU32Le(b3 + 66 + 4 * ib32),
              d3 * qda, x_qa_32, acc3_a, d3 * qdb, x_qb_32, acc3_b);
        }
      }
      Hsum4Avx2(acc0_a, acc1_a, acc2_a, acc3_a, out4_a);
      Hsum4Avx2(acc0_b, acc1_b, acc2_b, acc3_b, out4_b);
      return;
    }
    case kGgmlTypeIQ4_NL: {
      const int64_t nb = K / 32;
      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
      const __m256i lut_256 = _mm256_set_m128i(lut128, lut128);
      const __m256i mask4_256 = _mm256_set1_epi8(0x0F);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 18;
        const uint8_t* b1 = row1 + i * 18;
        const uint8_t* b2 = row2 + i * 18;
        const uint8_t* b3 = row3 + i * 18;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0));
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1));
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2));
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3));
        const float qda = x_qa[i].d;
        const float qdb = x_qb[i].d;
        const __m256i x_qa_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_qa[i].qs));
        const __m256i x_qb_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_qb[i].qs));
        const __m256i ax_a = _mm256_abs_epi8(x_qa_32);
        const __m256i ax_b = _mm256_abs_epi8(x_qb_32);
        AccumIq4Nl32Q8Batch2Avx2(b0 + 2, lut_256, mask4_256, 1, ax_a, x_qa_32, d0 * qda, acc0_a, ax_b, x_qb_32, d0 * qdb, acc0_b);
        AccumIq4Nl32Q8Batch2Avx2(b1 + 2, lut_256, mask4_256, 1, ax_a, x_qa_32, d1 * qda, acc1_a, ax_b, x_qb_32, d1 * qdb, acc1_b);
        AccumIq4Nl32Q8Batch2Avx2(b2 + 2, lut_256, mask4_256, 1, ax_a, x_qa_32, d2 * qda, acc2_a, ax_b, x_qb_32, d2 * qdb, acc2_b);
        AccumIq4Nl32Q8Batch2Avx2(b3 + 2, lut_256, mask4_256, 1, ax_a, x_qa_32, d3 * qda, acc3_a, ax_b, x_qb_32, d3 * qdb, acc3_b);
      }
      Hsum4Avx2(acc0_a, acc1_a, acc2_a, acc3_a, out4_a);
      Hsum4Avx2(acc0_b, acc1_b, acc2_b, acc3_b, out4_b);
      return;
    }
    case kGgmlTypeIQ4_XS: {
      const int64_t nb = K / 256;
      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
      const __m256i lut_256 = _mm256_set_m128i(lut128, lut128);
      const __m256i mask4_256 = _mm256_set1_epi8(0x0F);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 136;
        const uint8_t* b1 = row1 + i * 136;
        const uint8_t* b2 = row2 + i * 136;
        const uint8_t* b3 = row3 + i * 136;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0));
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1));
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2));
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3));
        const uint16_t sh0 = ReadU16Le(b0 + 2);
        const uint16_t sh1 = ReadU16Le(b1 + 2);
        const uint16_t sh2 = ReadU16Le(b2 + 2);
        const uint16_t sh3 = ReadU16Le(b3 + 2);
        const GgufQ8Block* qa = x_qa + i * 8;
        const GgufQ8Block* qb = x_qb + i * 8;
        for (int ib = 0; ib < 8; ++ib) {
          const int ls0 = ((b0[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh0 >> (2 * ib)) & 3) << 4);
          const int ls1 = ((b1[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh1 >> (2 * ib)) & 3) << 4);
          const int ls2 = ((b2[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh2 >> (2 * ib)) & 3) << 4);
          const int ls3 = ((b3[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh3 >> (2 * ib)) & 3) << 4);
          const float qda = qa[ib].d;
          const float qdb = qb[ib].d;
          const __m256i x_qa_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa[ib].qs));
          const __m256i x_qb_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qb[ib].qs));
          const __m256i ax_a = _mm256_abs_epi8(x_qa_32);
          const __m256i ax_b = _mm256_abs_epi8(x_qb_32);
          AccumIq4Nl32Q8Batch2Avx2(b0 + 8 + ib * 16, lut_256, mask4_256, static_cast<int16_t>(ls0 - 32), ax_a, x_qa_32, d0 * qda, acc0_a, ax_b, x_qb_32, d0 * qdb, acc0_b);
          AccumIq4Nl32Q8Batch2Avx2(b1 + 8 + ib * 16, lut_256, mask4_256, static_cast<int16_t>(ls1 - 32), ax_a, x_qa_32, d1 * qda, acc1_a, ax_b, x_qb_32, d1 * qdb, acc1_b);
          AccumIq4Nl32Q8Batch2Avx2(b2 + 8 + ib * 16, lut_256, mask4_256, static_cast<int16_t>(ls2 - 32), ax_a, x_qa_32, d2 * qda, acc2_a, ax_b, x_qb_32, d2 * qdb, acc2_b);
          AccumIq4Nl32Q8Batch2Avx2(b3 + 8 + ib * 16, lut_256, mask4_256, static_cast<int16_t>(ls3 - 32), ax_a, x_qa_32, d3 * qda, acc3_a, ax_b, x_qb_32, d3 * qdb, acc3_b);
        }
      }
      Hsum4Avx2(acc0_a, acc1_a, acc2_a, acc3_a, out4_a);
      Hsum4Avx2(acc0_b, acc1_b, acc2_b, acc3_b, out4_b);
      return;
    }
    case kGgmlTypeQ2_0: {
      const int64_t nb = K / 64;
      const __m256i mask2_256 = _mm256_set1_epi8(0x03);
      const __m256i ones_16 = _mm256_set1_epi16(1);
      __m128 bsum_acc4_a = _mm_setzero_ps();
      __m128 bsum_acc4_b = _mm_setzero_ps();
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 18;
        const uint8_t* b1 = row1 + i * 18;
        const uint8_t* b2 = row2 + i * 18;
        const uint8_t* b3 = row3 + i * 18;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0));
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1));
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2));
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3));
        const __m128 d4_v = _mm_setr_ps(d0, d1, d2, d3);

        const GgufQ8Block& qa_0 = x_qa[2 * i + 0];
        const GgufQ8Block& qa_1 = x_qa[2 * i + 1];
        const GgufQ8Block& qb_0 = x_qb[2 * i + 0];
        const GgufQ8Block& qb_1 = x_qb[2 * i + 1];
        const __m256i x_qa0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa_0.qs));
        const __m256i x_qa1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa_1.qs));
        const __m256i x_qb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qb_0.qs));
        const __m256i x_qb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qb_1.qs));

        const __m256i u0_0 = UnpackQ2_0_32Avx2(b0 + 2, mask2_256);
        const __m256i u1_0 = UnpackQ2_0_32Avx2(b1 + 2, mask2_256);
        const __m256i u2_0 = UnpackQ2_0_32Avx2(b2 + 2, mask2_256);
        const __m256i u3_0 = UnpackQ2_0_32Avx2(b3 + 2, mask2_256);
        acc0_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u0_0, x_qa0), ones_16)), _mm256_set1_ps(d0 * qa_0.d), acc0_a);
        acc1_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u1_0, x_qa0), ones_16)), _mm256_set1_ps(d1 * qa_0.d), acc1_a);
        acc2_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u2_0, x_qa0), ones_16)), _mm256_set1_ps(d2 * qa_0.d), acc2_a);
        acc3_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u3_0, x_qa0), ones_16)), _mm256_set1_ps(d3 * qa_0.d), acc3_a);
        acc0_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u0_0, x_qb0), ones_16)), _mm256_set1_ps(d0 * qb_0.d), acc0_b);
        acc1_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u1_0, x_qb0), ones_16)), _mm256_set1_ps(d1 * qb_0.d), acc1_b);
        acc2_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u2_0, x_qb0), ones_16)), _mm256_set1_ps(d2 * qb_0.d), acc2_b);
        acc3_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u3_0, x_qb0), ones_16)), _mm256_set1_ps(d3 * qb_0.d), acc3_b);

        const __m256i u0_1 = UnpackQ2_0_32Avx2(b0 + 10, mask2_256);
        const __m256i u1_1 = UnpackQ2_0_32Avx2(b1 + 10, mask2_256);
        const __m256i u2_1 = UnpackQ2_0_32Avx2(b2 + 10, mask2_256);
        const __m256i u3_1 = UnpackQ2_0_32Avx2(b3 + 10, mask2_256);
        acc0_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u0_1, x_qa1), ones_16)), _mm256_set1_ps(d0 * qa_1.d), acc0_a);
        acc1_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u1_1, x_qa1), ones_16)), _mm256_set1_ps(d1 * qa_1.d), acc1_a);
        acc2_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u2_1, x_qa1), ones_16)), _mm256_set1_ps(d2 * qa_1.d), acc2_a);
        acc3_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u3_1, x_qa1), ones_16)), _mm256_set1_ps(d3 * qa_1.d), acc3_a);
        acc0_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u0_1, x_qb1), ones_16)), _mm256_set1_ps(d0 * qb_1.d), acc0_b);
        acc1_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u1_1, x_qb1), ones_16)), _mm256_set1_ps(d1 * qb_1.d), acc1_b);
        acc2_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u2_1, x_qb1), ones_16)), _mm256_set1_ps(d2 * qb_1.d), acc2_b);
        acc3_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u3_1, x_qb1), ones_16)), _mm256_set1_ps(d3 * qb_1.d), acc3_b);

        const float qba = qa_0.d * static_cast<float>(qa_0.bsum) + qa_1.d * static_cast<float>(qa_1.bsum);
        const float qbb = qb_0.d * static_cast<float>(qb_0.bsum) + qb_1.d * static_cast<float>(qb_1.bsum);
        bsum_acc4_a = _mm_fmadd_ps(d4_v, _mm_set1_ps(qba), bsum_acc4_a);
        bsum_acc4_b = _mm_fmadd_ps(d4_v, _mm_set1_ps(qbb), bsum_acc4_b);
      }
      Hsum4Avx2(acc0_a, acc1_a, acc2_a, acc3_a, out4_a);
      Hsum4Avx2(acc0_b, acc1_b, acc2_b, acc3_b, out4_b);
      _mm_storeu_ps(out4_a, _mm_sub_ps(_mm_loadu_ps(out4_a), bsum_acc4_a));
      _mm_storeu_ps(out4_b, _mm_sub_ps(_mm_loadu_ps(out4_b), bsum_acc4_b));
      return;
    }
    default: {
      VecDotGgufRow4xQ8Avx2(quant_type, row0, row1, row2, row3, x_qa, K, out4_a);
      VecDotGgufRow4xQ8Avx2(quant_type, row0, row1, row2, row3, x_qb, K, out4_b);
      return;
    }
  }
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq2S32Q8Batch4Avx2(
    const uint8_t* qs_ib, const uint8_t* signs_ib, uint32_t qh_b, uint8_t sc,
    float d_a, __m256i x_qa, __m256& acc_a,
    float d_b, __m256i x_qb, __m256& acc_b,
    float d_c, __m256i x_qc, __m256& acc_c,
    float d_d, __m256i x_qd, __m256& acc_d) noexcept {
  const __m256i grid_u8 = _mm256_set_epi64x(
      static_cast<int64_t>(iq2s_grid[qs_ib[3] | ((qh_b << 2) & 0x300u)]),
      static_cast<int64_t>(iq2s_grid[qs_ib[2] | ((qh_b << 4) & 0x300u)]),
      static_cast<int64_t>(iq2s_grid[qs_ib[1] | ((qh_b << 6) & 0x300u)]),
      static_cast<int64_t>(iq2s_grid[qs_ib[0] | ((qh_b << 8) & 0x300u)]));
  const __m256i signs = _mm256_set_epi64x(
      static_cast<int64_t>(kSignBytes8Bit.mask[signs_ib[3]]),
      static_cast<int64_t>(kSignBytes8Bit.mask[signs_ib[2]]),
      static_cast<int64_t>(kSignBytes8Bit.mask[signs_ib[1]]),
      static_cast<int64_t>(kSignBytes8Bit.mask[signs_ib[0]]));
  const __m256i ls_v = _mm256_set_m128i(
      _mm_set1_epi16(static_cast<int16_t>(2 * (sc >> 4) + 1)),
      _mm_set1_epi16(static_cast<int16_t>(2 * (sc & 0xF) + 1)));
  const __m256i p32_a = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qa, signs)), ls_v);
  const __m256i p32_b = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qb, signs)), ls_v);
  const __m256i p32_c = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qc, signs)), ls_v);
  const __m256i p32_d = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qd, signs)), ls_v);
  acc_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_a), _mm256_set1_ps(d_a), acc_a);
  acc_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_b), _mm256_set1_ps(d_b), acc_b);
  acc_c = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_c), _mm256_set1_ps(d_c), acc_c);
  acc_d = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_d), _mm256_set1_ps(d_d), acc_d);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq3Xxs32Q8Batch4Avx2(
    const uint8_t* qs_ib, uint32_t aux32,
    float d_a, __m256i x_qa, __m256& acc_a,
    float d_b, __m256i x_qb, __m256& acc_b,
    float d_c, __m256i x_qc, __m256& acc_c,
    float d_d, __m256i x_qd, __m256& acc_d) noexcept {
  const __m256i grid_u8 = _mm256_setr_epi32(
      static_cast<int>(iq3xxs_grid[qs_ib[0]]),
      static_cast<int>(iq3xxs_grid[qs_ib[1]]),
      static_cast<int>(iq3xxs_grid[qs_ib[2]]),
      static_cast<int>(iq3xxs_grid[qs_ib[3]]),
      static_cast<int>(iq3xxs_grid[qs_ib[4]]),
      static_cast<int>(iq3xxs_grid[qs_ib[5]]),
      static_cast<int>(iq3xxs_grid[qs_ib[6]]),
      static_cast<int>(iq3xxs_grid[qs_ib[7]]));
  const __m256i signs = _mm256_set_epi64x(
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 21) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 14) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 7) & 127u]),
      static_cast<int64_t>(kSignBytes7Bit.mask[(aux32 >> 0) & 127u]));
  const __m256i ls_v = _mm256_set1_epi16(static_cast<int16_t>(2 * (aux32 >> 28) + 1));
  const __m256i p32_a = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qa, signs)), ls_v);
  const __m256i p32_b = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qb, signs)), ls_v);
  const __m256i p32_c = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qc, signs)), ls_v);
  const __m256i p32_d = _mm256_madd_epi16(_mm256_maddubs_epi16(grid_u8, _mm256_sign_epi8(x_qd, signs)), ls_v);
  acc_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_a), _mm256_set1_ps(d_a), acc_a);
  acc_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_b), _mm256_set1_ps(d_b), acc_b);
  acc_c = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_c), _mm256_set1_ps(d_c), acc_c);
  acc_d = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_d), _mm256_set1_ps(d_d), acc_d);
}

__attribute__((target("avx2,fma,f16c"))) inline void AccumIq4Nl32Q8Batch4Avx2(
    const uint8_t* qs16, __m256i lut_iq4nl_256, __m256i mask4_256, int16_t sub_scale_i16,
    __m256i ax_a, __m256i x_qa, float scale_a, __m256& acc_a,
    __m256i ax_b, __m256i x_qb, float scale_b, __m256& acc_b,
    __m256i ax_c, __m256i x_qc, float scale_c, __m256& acc_c,
    __m256i ax_d, __m256i x_qd, float scale_d, __m256& acc_d) noexcept {
  const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs16));
  const __m256i nibbles = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw16, 4), raw16), mask4_256);
  const __m256i w_i8 = _mm256_shuffle_epi8(lut_iq4nl_256, nibbles);
  const __m256i ls_v = _mm256_set1_epi16(sub_scale_i16);
  const __m256i p32_a = _mm256_madd_epi16(_mm256_maddubs_epi16(ax_a, _mm256_sign_epi8(w_i8, x_qa)), ls_v);
  const __m256i p32_b = _mm256_madd_epi16(_mm256_maddubs_epi16(ax_b, _mm256_sign_epi8(w_i8, x_qb)), ls_v);
  const __m256i p32_c = _mm256_madd_epi16(_mm256_maddubs_epi16(ax_c, _mm256_sign_epi8(w_i8, x_qc)), ls_v);
  const __m256i p32_d = _mm256_madd_epi16(_mm256_maddubs_epi16(ax_d, _mm256_sign_epi8(w_i8, x_qd)), ls_v);
  acc_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_a), _mm256_set1_ps(scale_a), acc_a);
  acc_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_b), _mm256_set1_ps(scale_b), acc_b);
  acc_c = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_c), _mm256_set1_ps(scale_c), acc_c);
  acc_d = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_d), _mm256_set1_ps(scale_d), acc_d);
}

__attribute__((target("avx2,fma,f16c"))) inline void VecDotGgufRow4xQ8Batch4Avx2(
    int quant_type,
    const uint8_t* row0,
    const uint8_t* row1,
    const uint8_t* row2,
    const uint8_t* row3,
    const GgufQ8Block* x_qa,
    const GgufQ8Block* x_qb,
    const GgufQ8Block* x_qc,
    const GgufQ8Block* x_qd,
    int64_t K,
    float* out4_a,
    float* out4_b,
    float* out4_c,
    float* out4_d) {
  __m256 acc0_a = _mm256_setzero_ps(), acc1_a = _mm256_setzero_ps(), acc2_a = _mm256_setzero_ps(), acc3_a = _mm256_setzero_ps();
  __m256 acc0_b = _mm256_setzero_ps(), acc1_b = _mm256_setzero_ps(), acc2_b = _mm256_setzero_ps(), acc3_b = _mm256_setzero_ps();
  __m256 acc0_c = _mm256_setzero_ps(), acc1_c = _mm256_setzero_ps(), acc2_c = _mm256_setzero_ps(), acc3_c = _mm256_setzero_ps();
  __m256 acc0_d = _mm256_setzero_ps(), acc1_d = _mm256_setzero_ps(), acc2_d = _mm256_setzero_ps(), acc3_d = _mm256_setzero_ps();

  switch (quant_type) {
    case kGgmlTypeIQ2_S: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 82;
        const uint8_t* b1 = row1 + i * 82;
        const uint8_t* b2 = row2 + i * 82;
        const uint8_t* b3 = row3 + i * 82;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0)) * 0.125f;
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1)) * 0.125f;
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2)) * 0.125f;
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3)) * 0.125f;
        const GgufQ8Block* qa = x_qa + i * 8;
        const GgufQ8Block* qb = x_qb + i * 8;
        const GgufQ8Block* qc = x_qc + i * 8;
        const GgufQ8Block* qd = x_qd + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float qda = qa[ib32].d, qdb = qb[ib32].d, qdc = qc[ib32].d, qdd = qd[ib32].d;
          const __m256i x_qa_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa[ib32].qs));
          const __m256i x_qb_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qb[ib32].qs));
          const __m256i x_qc_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qc[ib32].qs));
          const __m256i x_qd_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qd[ib32].qs));
          AccumIq2S32Q8Batch4Avx2(b0 + 2 + 4 * ib32, b0 + 34 + 4 * ib32, b0[66 + ib32], b0[74 + ib32],
                                  d0 * qda, x_qa_32, acc0_a, d0 * qdb, x_qb_32, acc0_b, d0 * qdc, x_qc_32, acc0_c, d0 * qdd, x_qd_32, acc0_d);
          AccumIq2S32Q8Batch4Avx2(b1 + 2 + 4 * ib32, b1 + 34 + 4 * ib32, b1[66 + ib32], b1[74 + ib32],
                                  d1 * qda, x_qa_32, acc1_a, d1 * qdb, x_qb_32, acc1_b, d1 * qdc, x_qc_32, acc1_c, d1 * qdd, x_qd_32, acc1_d);
          AccumIq2S32Q8Batch4Avx2(b2 + 2 + 4 * ib32, b2 + 34 + 4 * ib32, b2[66 + ib32], b2[74 + ib32],
                                  d2 * qda, x_qa_32, acc2_a, d2 * qdb, x_qb_32, acc2_b, d2 * qdc, x_qc_32, acc2_c, d2 * qdd, x_qd_32, acc2_d);
          AccumIq2S32Q8Batch4Avx2(b3 + 2 + 4 * ib32, b3 + 34 + 4 * ib32, b3[66 + ib32], b3[74 + ib32],
                                  d3 * qda, x_qa_32, acc3_a, d3 * qdb, x_qb_32, acc3_b, d3 * qdc, x_qc_32, acc3_c, d3 * qdd, x_qd_32, acc3_d);
        }
      }
      Hsum4Avx2(acc0_a, acc1_a, acc2_a, acc3_a, out4_a);
      Hsum4Avx2(acc0_b, acc1_b, acc2_b, acc3_b, out4_b);
      Hsum4Avx2(acc0_c, acc1_c, acc2_c, acc3_c, out4_c);
      Hsum4Avx2(acc0_d, acc1_d, acc2_d, acc3_d, out4_d);
      return;
    }
    case kGgmlTypeIQ3_XXS: {
      const int64_t nb = K / 256;
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 98;
        const uint8_t* b1 = row1 + i * 98;
        const uint8_t* b2 = row2 + i * 98;
        const uint8_t* b3 = row3 + i * 98;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0)) * 0.25f;
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1)) * 0.25f;
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2)) * 0.25f;
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3)) * 0.25f;
        const GgufQ8Block* qa = x_qa + i * 8;
        const GgufQ8Block* qb = x_qb + i * 8;
        const GgufQ8Block* qc = x_qc + i * 8;
        const GgufQ8Block* qd = x_qd + i * 8;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float qda = qa[ib32].d, qdb = qb[ib32].d, qdc = qc[ib32].d, qdd = qd[ib32].d;
          const __m256i x_qa_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa[ib32].qs));
          const __m256i x_qb_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qb[ib32].qs));
          const __m256i x_qc_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qc[ib32].qs));
          const __m256i x_qd_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qd[ib32].qs));
          AccumIq3Xxs32Q8Batch4Avx2(b0 + 2 + 8 * ib32, ReadU32Le(b0 + 66 + 4 * ib32),
                                    d0 * qda, x_qa_32, acc0_a, d0 * qdb, x_qb_32, acc0_b, d0 * qdc, x_qc_32, acc0_c, d0 * qdd, x_qd_32, acc0_d);
          AccumIq3Xxs32Q8Batch4Avx2(b1 + 2 + 8 * ib32, ReadU32Le(b1 + 66 + 4 * ib32),
                                    d1 * qda, x_qa_32, acc1_a, d1 * qdb, x_qb_32, acc1_b, d1 * qdc, x_qc_32, acc1_c, d1 * qdd, x_qd_32, acc1_d);
          AccumIq3Xxs32Q8Batch4Avx2(b2 + 2 + 8 * ib32, ReadU32Le(b2 + 66 + 4 * ib32),
                                    d2 * qda, x_qa_32, acc2_a, d2 * qdb, x_qb_32, acc2_b, d2 * qdc, x_qc_32, acc2_c, d2 * qdd, x_qd_32, acc2_d);
          AccumIq3Xxs32Q8Batch4Avx2(b3 + 2 + 8 * ib32, ReadU32Le(b3 + 66 + 4 * ib32),
                                    d3 * qda, x_qa_32, acc3_a, d3 * qdb, x_qb_32, acc3_b, d3 * qdc, x_qc_32, acc3_c, d3 * qdd, x_qd_32, acc3_d);
        }
      }
      Hsum4Avx2(acc0_a, acc1_a, acc2_a, acc3_a, out4_a);
      Hsum4Avx2(acc0_b, acc1_b, acc2_b, acc3_b, out4_b);
      Hsum4Avx2(acc0_c, acc1_c, acc2_c, acc3_c, out4_c);
      Hsum4Avx2(acc0_d, acc1_d, acc2_d, acc3_d, out4_d);
      return;
    }
    case kGgmlTypeIQ4_NL: {
      const int64_t nb = K / 32;
      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
      const __m256i lut_256 = _mm256_set_m128i(lut128, lut128);
      const __m256i mask4_256 = _mm256_set1_epi8(0x0F);
      for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b0 = row0 + i * 18;
        const uint8_t* b1 = row1 + i * 18;
        const uint8_t* b2 = row2 + i * 18;
        const uint8_t* b3 = row3 + i * 18;
        const float d0 = Fp16ToFloatF16c(ReadU16Le(b0));
        const float d1 = Fp16ToFloatF16c(ReadU16Le(b1));
        const float d2 = Fp16ToFloatF16c(ReadU16Le(b2));
        const float d3 = Fp16ToFloatF16c(ReadU16Le(b3));
        const float qda = x_qa[i].d, qdb = x_qb[i].d, qdc = x_qc[i].d, qdd = x_qd[i].d;
        const __m256i x_qa_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_qa[i].qs));
        const __m256i x_qb_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_qb[i].qs));
        const __m256i x_qc_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_qc[i].qs));
        const __m256i x_qd_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_qd[i].qs));
        const __m256i ax_a = _mm256_abs_epi8(x_qa_32);
        const __m256i ax_b = _mm256_abs_epi8(x_qb_32);
        const __m256i ax_c = _mm256_abs_epi8(x_qc_32);
        const __m256i ax_d = _mm256_abs_epi8(x_qd_32);
        AccumIq4Nl32Q8Batch4Avx2(b0 + 2, lut_256, mask4_256, 1, ax_a, x_qa_32, d0 * qda, acc0_a, ax_b, x_qb_32, d0 * qdb, acc0_b, ax_c, x_qc_32, d0 * qdc, acc0_c, ax_d, x_qd_32, d0 * qdd, acc0_d);
        AccumIq4Nl32Q8Batch4Avx2(b1 + 2, lut_256, mask4_256, 1, ax_a, x_qa_32, d1 * qda, acc1_a, ax_b, x_qb_32, d1 * qdb, acc1_b, ax_c, x_qc_32, d1 * qdc, acc1_c, ax_d, x_qd_32, d1 * qdd, acc1_d);
        AccumIq4Nl32Q8Batch4Avx2(b2 + 2, lut_256, mask4_256, 1, ax_a, x_qa_32, d2 * qda, acc2_a, ax_b, x_qb_32, d2 * qdb, acc2_b, ax_c, x_qc_32, d2 * qdc, acc2_c, ax_d, x_qd_32, d2 * qdd, acc2_d);
        AccumIq4Nl32Q8Batch4Avx2(b3 + 2, lut_256, mask4_256, 1, ax_a, x_qa_32, d3 * qda, acc3_a, ax_b, x_qb_32, d3 * qdb, acc3_b, ax_c, x_qc_32, d3 * qdc, acc3_c, ax_d, x_qd_32, d3 * qdd, acc3_d);
      }
      Hsum4Avx2(acc0_a, acc1_a, acc2_a, acc3_a, out4_a);
      Hsum4Avx2(acc0_b, acc1_b, acc2_b, acc3_b, out4_b);
      Hsum4Avx2(acc0_c, acc1_c, acc2_c, acc3_c, out4_c);
      Hsum4Avx2(acc0_d, acc1_d, acc2_d, acc3_d, out4_d);
      return;
    }
    default: {
      VecDotGgufRow4xQ8Batch2Avx2(quant_type, row0, row1, row2, row3, x_qa, x_qb, K, out4_a, out4_b);
      VecDotGgufRow4xQ8Batch2Avx2(quant_type, row0, row1, row2, row3, x_qc, x_qd, K, out4_c, out4_d);
      return;
    }
  }
}

#endif  // ORT_GGUF_CPU_HAS_AVX2_TARGET

inline float VecDotGgufRowScalar(int quant_type, const uint8_t* weight_row, const float* x, int64_t K) {
  int64_t block_elems = 0;
  int64_t block_bytes = 0;
  if (!GgmlTypeBlockInfo(quant_type, block_elems, block_bytes) || block_elems <= 0) {
    return 0.0f;
  }
  const int64_t chunk_elems = (block_elems < 32) ? 32 : block_elems;
  const int64_t chunk_bytes = (chunk_elems / block_elems) * block_bytes;
  alignas(32) float buf[256];
  const int64_t n_chunks = K / chunk_elems;
  float sum = 0.0f;
  for (int64_t c = 0; c < n_chunks; ++c) {
    DequantizeGgufRowCpu(weight_row + c * chunk_bytes, quant_type, chunk_elems, buf);
    const float* xp = x + c * chunk_elems;
    for (int64_t j = 0; j < chunk_elems; ++j) {
      sum += buf[j] * xp[j];
    }
  }
  return sum;
}

inline bool CpuSupportsAvx2Fma() noexcept {
#if ORT_GGUF_CPU_HAS_AVX2_TARGET
  static const bool supported =
      __builtin_cpu_supports("avx2") &&
      __builtin_cpu_supports("fma") &&
      __builtin_cpu_supports("f16c");
  return supported;
#else
  return false;
#endif
}

inline float VecDotGgufRowCpu(int quant_type, const uint8_t* weight_row, const float* x, int64_t K) {
#if ORT_GGUF_CPU_HAS_AVX2_TARGET
  if (CpuSupportsAvx2Fma()) {
    return VecDotGgufRowAvx2(quant_type, weight_row, x, K);
  }
#endif
  return VecDotGgufRowScalar(quant_type, weight_row, x, K);
}

inline void VecDotGgufGateUpRowCpu(
    int gate_type,
    const uint8_t* gate_row,
    int up_type,
    const uint8_t* up_row,
    const float* x,
    int64_t K,
    float* out_gate,
    float* out_up) {
#if ORT_GGUF_CPU_HAS_AVX2_TARGET
  if (CpuSupportsAvx2Fma()) {
    VecDotGgufGateUpRowAvx2(gate_type, gate_row, up_type, up_row, x, K, out_gate, out_up);
    return;
  }
#endif
  *out_gate = VecDotGgufRowScalar(gate_type, gate_row, x, K);
  *out_up = VecDotGgufRowScalar(up_type, up_row, x, K);
}

inline void VecDotGgufRow4xCpu(
    int quant_type,
    const uint8_t* row0,
    const uint8_t* row1,
    const uint8_t* row2,
    const uint8_t* row3,
    const float* x,
    int64_t K,
    float* out4) {
#if ORT_GGUF_CPU_HAS_AVX2_TARGET
  if (CpuSupportsAvx2Fma()) {
    VecDotGgufRow4xAvx2(quant_type, row0, row1, row2, row3, x, K, out4);
    return;
  }
#endif
  out4[0] = VecDotGgufRowScalar(quant_type, row0, x, K);
  out4[1] = VecDotGgufRowScalar(quant_type, row1, x, K);
  out4[2] = VecDotGgufRowScalar(quant_type, row2, x, K);
  out4[3] = VecDotGgufRowScalar(quant_type, row3, x, K);
}

inline void QuantizeBlockQ8Cpu(const float* x, GgufQ8Block* blk) noexcept {
#if ORT_GGUF_CPU_HAS_AVX2_TARGET
  if (CpuSupportsAvx2Fma()) {
    QuantizeBlockQ8Avx2(x, blk);
    return;
  }
#endif
  QuantizeBlockQ8Scalar(x, blk);
}

inline void QuantizeRowQ8Cpu(const float* x, int64_t K, GgufQ8Block* out_q8) noexcept {
  const int64_t nb = K / 32;
#if ORT_GGUF_CPU_HAS_AVX2_TARGET
  if (CpuSupportsAvx2Fma()) {
    for (int64_t b = 0; b < nb; ++b) {
      QuantizeBlockQ8Avx2(x + b * 32, out_q8 + b);
    }
    return;
  }
#endif
  for (int64_t b = 0; b < nb; ++b) {
    QuantizeBlockQ8Scalar(x + b * 32, out_q8 + b);
  }
}

inline float VecDotGgufRowQ8Scalar(
    int quant_type, const uint8_t* weight_row, const GgufQ8Block* x_q8, int64_t K) {
  int64_t block_elems = 0;
  int64_t block_bytes = 0;
  if (!GgmlTypeBlockInfo(quant_type, block_elems, block_bytes) || block_elems <= 0) {
    return 0.0f;
  }
  const int64_t chunk_elems = (block_elems < 32) ? 32 : block_elems;
  const int64_t chunk_bytes = (chunk_elems / block_elems) * block_bytes;
  alignas(32) float buf[256];
  const int64_t n_chunks = K / chunk_elems;
  float sum = 0.0f;
  for (int64_t c = 0; c < n_chunks; ++c) {
    DequantizeGgufRowCpu(weight_row + c * chunk_bytes, quant_type, chunk_elems, buf);
    for (int64_t b = 0; b < chunk_elems / 32; ++b) {
      const GgufQ8Block& qb = x_q8[(c * chunk_elems) / 32 + b];
      const float* bp = buf + b * 32;
      int32_t isum = 0;
      for (int j = 0; j < 32; ++j) {
        sum += bp[j] * (qb.d * static_cast<float>(qb.qs[j]));
      }
      (void)isum;
    }
  }
  return sum;
}

inline float VecDotGgufRowQ8Cpu(
    int quant_type, const uint8_t* weight_row, const GgufQ8Block* x_q8, int64_t K) {
#if ORT_GGUF_CPU_HAS_AVX2_TARGET
  if (CpuSupportsAvx2Fma()) {
    return VecDotGgufRowQ8Avx2(quant_type, weight_row, x_q8, K);
  }
#endif
  return VecDotGgufRowQ8Scalar(quant_type, weight_row, x_q8, K);
}

inline void VecDotGgufRow4xQ8Cpu(
    int quant_type,
    const uint8_t* row0,
    const uint8_t* row1,
    const uint8_t* row2,
    const uint8_t* row3,
    const GgufQ8Block* x_q8,
    int64_t K,
    float* out4) {
#if ORT_GGUF_CPU_HAS_AVX2_TARGET
  if (CpuSupportsAvx2Fma()) {
    VecDotGgufRow4xQ8Avx2(quant_type, row0, row1, row2, row3, x_q8, K, out4);
    return;
  }
#endif
  out4[0] = VecDotGgufRowQ8Scalar(quant_type, row0, x_q8, K);
  out4[1] = VecDotGgufRowQ8Scalar(quant_type, row1, x_q8, K);
  out4[2] = VecDotGgufRowQ8Scalar(quant_type, row2, x_q8, K);
  out4[3] = VecDotGgufRowQ8Scalar(quant_type, row3, x_q8, K);
}

inline void VecDotGgufGateUpRowQ8Cpu(
    int gate_type,
    const uint8_t* gate_row,
    int up_type,
    const uint8_t* up_row,
    const GgufQ8Block* x_q8,
    int64_t K,
    float* out_gate,
    float* out_up) {
  *out_gate = VecDotGgufRowQ8Cpu(gate_type, gate_row, x_q8, K);
  *out_up = VecDotGgufRowQ8Cpu(up_type, up_row, x_q8, K);
}

inline void VecDotGgufGateUpRow2xQ8Cpu(
    int gate_type,
    const uint8_t* gate_row0,
    const uint8_t* gate_row1,
    int up_type,
    const uint8_t* up_row0,
    const uint8_t* up_row1,
    const GgufQ8Block* x_q8,
    int64_t K,
    float* out_gate2,
    float* out_up2) {
  if (gate_type == up_type) {
    float out4[4];
    VecDotGgufRow4xQ8Cpu(gate_type, gate_row0, gate_row1, up_row0, up_row1, x_q8, K, out4);
    out_gate2[0] = out4[0];
    out_gate2[1] = out4[1];
    out_up2[0] = out4[2];
    out_up2[1] = out4[3];
    return;
  }
  out_gate2[0] = VecDotGgufRowQ8Cpu(gate_type, gate_row0, x_q8, K);
  out_gate2[1] = VecDotGgufRowQ8Cpu(gate_type, gate_row1, x_q8, K);
  out_up2[0] = VecDotGgufRowQ8Cpu(up_type, up_row0, x_q8, K);
  out_up2[1] = VecDotGgufRowQ8Cpu(up_type, up_row1, x_q8, K);
}

inline void VecDotGgufRow4xQ8Batch2Cpu(
    int quant_type,
    const uint8_t* row0,
    const uint8_t* row1,
    const uint8_t* row2,
    const uint8_t* row3,
    const GgufQ8Block* x_qa,
    const GgufQ8Block* x_qb,
    int64_t K,
    float* out4_a,
    float* out4_b) {
#if ORT_GGUF_CPU_HAS_AVX2_TARGET
  if (CpuSupportsAvx2Fma()) {
    VecDotGgufRow4xQ8Batch2Avx2(quant_type, row0, row1, row2, row3, x_qa, x_qb, K, out4_a, out4_b);
    return;
  }
#endif
  VecDotGgufRow4xQ8Cpu(quant_type, row0, row1, row2, row3, x_qa, K, out4_a);
  VecDotGgufRow4xQ8Cpu(quant_type, row0, row1, row2, row3, x_qb, K, out4_b);
}

inline void VecDotGgufGateUpRow2xQ8Batch2Cpu(
    int gate_type,
    const uint8_t* gate_row0,
    const uint8_t* gate_row1,
    int up_type,
    const uint8_t* up_row0,
    const uint8_t* up_row1,
    const GgufQ8Block* x_qa,
    const GgufQ8Block* x_qb,
    int64_t K,
    float* out_gate2_a,
    float* out_up2_a,
    float* out_gate2_b,
    float* out_up2_b) {
  if (gate_type == up_type) {
    float out4_a[4];
    float out4_b[4];
    VecDotGgufRow4xQ8Batch2Cpu(gate_type, gate_row0, gate_row1, up_row0, up_row1, x_qa, x_qb, K, out4_a, out4_b);
    out_gate2_a[0] = out4_a[0];
    out_gate2_a[1] = out4_a[1];
    out_up2_a[0] = out4_a[2];
    out_up2_a[1] = out4_a[3];
    out_gate2_b[0] = out4_b[0];
    out_gate2_b[1] = out4_b[1];
    out_up2_b[0] = out4_b[2];
    out_up2_b[1] = out4_b[3];
    return;
  }
  VecDotGgufGateUpRow2xQ8Cpu(gate_type, gate_row0, gate_row1, up_type, up_row0, up_row1, x_qa, K, out_gate2_a, out_up2_a);
  VecDotGgufGateUpRow2xQ8Cpu(gate_type, gate_row0, gate_row1, up_type, up_row0, up_row1, x_qb, K, out_gate2_b, out_up2_b);
}

inline void VecDotGgufRow4xQ8Batch4Cpu(
    int quant_type,
    const uint8_t* row0,
    const uint8_t* row1,
    const uint8_t* row2,
    const uint8_t* row3,
    const GgufQ8Block* x_qa,
    const GgufQ8Block* x_qb,
    const GgufQ8Block* x_qc,
    const GgufQ8Block* x_qd,
    int64_t K,
    float* out4_a,
    float* out4_b,
    float* out4_c,
    float* out4_d) {
#if ORT_GGUF_CPU_HAS_AVX2_TARGET
  if (CpuSupportsAvx2Fma()) {
    VecDotGgufRow4xQ8Batch4Avx2(quant_type, row0, row1, row2, row3, x_qa, x_qb, x_qc, x_qd, K,
                                out4_a, out4_b, out4_c, out4_d);
    return;
  }
#endif
  VecDotGgufRow4xQ8Batch2Cpu(quant_type, row0, row1, row2, row3, x_qa, x_qb, K, out4_a, out4_b);
  VecDotGgufRow4xQ8Batch2Cpu(quant_type, row0, row1, row2, row3, x_qc, x_qd, K, out4_c, out4_d);
}

inline void VecDotGgufGateUpRow2xQ8Batch4Cpu(
    int gate_type,
    const uint8_t* gate_row0,
    const uint8_t* gate_row1,
    int up_type,
    const uint8_t* up_row0,
    const uint8_t* up_row1,
    const GgufQ8Block* x_qa,
    const GgufQ8Block* x_qb,
    const GgufQ8Block* x_qc,
    const GgufQ8Block* x_qd,
    int64_t K,
    float* out_gate2_a,
    float* out_up2_a,
    float* out_gate2_b,
    float* out_up2_b,
    float* out_gate2_c,
    float* out_up2_c,
    float* out_gate2_d,
    float* out_up2_d) {
  if (gate_type == up_type) {
    float out4_a[4], out4_b[4], out4_c[4], out4_d[4];
    VecDotGgufRow4xQ8Batch4Cpu(gate_type, gate_row0, gate_row1, up_row0, up_row1,
                               x_qa, x_qb, x_qc, x_qd, K, out4_a, out4_b, out4_c, out4_d);
    out_gate2_a[0] = out4_a[0];
    out_gate2_a[1] = out4_a[1];
    out_up2_a[0] = out4_a[2];
    out_up2_a[1] = out4_a[3];
    out_gate2_b[0] = out4_b[0];
    out_gate2_b[1] = out4_b[1];
    out_up2_b[0] = out4_b[2];
    out_up2_b[1] = out4_b[3];
    out_gate2_c[0] = out4_c[0];
    out_gate2_c[1] = out4_c[1];
    out_up2_c[0] = out4_c[2];
    out_up2_c[1] = out4_c[3];
    out_gate2_d[0] = out4_d[0];
    out_gate2_d[1] = out4_d[1];
    out_up2_d[0] = out4_d[2];
    out_up2_d[1] = out4_d[3];
    return;
  }
  VecDotGgufGateUpRow2xQ8Batch2Cpu(gate_type, gate_row0, gate_row1, up_type, up_row0, up_row1,
                                   x_qa, x_qb, K, out_gate2_a, out_up2_a, out_gate2_b, out_up2_b);
  VecDotGgufGateUpRow2xQ8Batch2Cpu(gate_type, gate_row0, gate_row1, up_type, up_row0, up_row1,
                                   x_qc, x_qd, K, out_gate2_c, out_up2_c, out_gate2_d, out_up2_d);
}

template <typename T>
inline float ActTypeToFloatCpu(T v) noexcept {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else if constexpr (std::is_same_v<T, MLFloat16>) {
    return v.ToFloat();
  } else {
    return v.ToFloat();
  }
}

template <typename T>
inline T FloatToActTypeCpu(float v) noexcept {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else if constexpr (std::is_same_v<T, MLFloat16>) {
    return MLFloat16(v);
  } else {
    return BFloat16(v);
  }
}

inline constexpr bool IsUnquantizedGgmlType(int ggml_type) noexcept {
  return ggml_type == kGgmlTypeF32 || ggml_type == kGgmlTypeF16 || ggml_type == kGgmlTypeBF16;
}

inline constexpr bool SupportsGgufQ8VecDotCpu(int ggml_type) noexcept {
  switch (ggml_type) {
    case kGgmlTypeF32:
    case kGgmlTypeF16:
    case kGgmlTypeBF16:
    case kGgmlTypeQ4_0:
    case kGgmlTypeQ5_0:
    case kGgmlTypeQ8_0:
    case kGgmlTypeQ5_K:
    case kGgmlTypeQ6_K:
    case kGgmlTypeIQ2_XXS:
    case kGgmlTypeIQ2_XS:
    case kGgmlTypeIQ3_XXS:
    case kGgmlTypeIQ4_NL:
    case kGgmlTypeIQ2_S:
    case kGgmlTypeIQ4_XS:
    case kGgmlTypeQ2_0:
      return true;
    default:
      return false;
  }
}

struct MoEPackedCpuColdScratch {
  std::vector<int> route_to_local_tok;
  std::vector<int> unique_toks;
  std::vector<GgufQ8Block> x_q8;
  std::vector<GgufQ8Block> inter_q8;
  std::vector<uint16_t> cold_fc2_u16;
};

template <typename T>
inline float RoundToActFloatCpu(float v) noexcept {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else {
    return ActTypeToFloatCpu<T>(FloatToActTypeCpu<T>(v));
  }
}

template <typename T>
inline void ComputeMoEPackedSoftmaxTopKCpu(
    const T* logits,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    float* topk_scales,
    int* topk_indices) {
  for (int row = 0; row < num_rows; ++row) {
    const T* row_logits = logits + static_cast<size_t>(row) * num_experts;
    float max_val = -std::numeric_limits<float>::infinity();
    for (int e = 0; e < num_experts; ++e) {
      float v = ActTypeToFloatCpu<T>(row_logits[e]);
      if (v == 0.0f) v = 0.0f;
      if (v > max_val) max_val = v;
    }
    float sum_exp = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
      float v = ActTypeToFloatCpu<T>(row_logits[e]);
      if (v == 0.0f) v = 0.0f;
      sum_exp += std::exp(v - max_val);
    }
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    int* out_idx = topk_indices + static_cast<size_t>(row) * k;
    float* out_sc = topk_scales + static_cast<size_t>(row) * k;
    float scale_sum = 0.0f;
    for (int r = 0; r < k; ++r) {
      int best_e = -1;
      float best_v = -std::numeric_limits<float>::infinity();
      for (int e = 0; e < num_experts; ++e) {
        bool already = false;
        for (int prev = 0; prev < r; ++prev) {
          if (out_idx[prev] == e) {
            already = true;
            break;
          }
        }
        if (already) continue;
        float v = ActTypeToFloatCpu<T>(row_logits[e]);
        if (v == 0.0f) v = 0.0f;
        if (best_e < 0 || v > best_v) {
          best_v = v;
          best_e = e;
        }
      }
      out_idx[r] = best_e;
      const float p = std::exp(best_v - max_val) * inv_sum;
      out_sc[r] = p;
      scale_sum += p;
    }
    if (normalize_scales && scale_sum > 0.0f) {
      const float inv_norm = 1.0f / scale_sum;
      for (int r = 0; r < k; ++r) {
        out_sc[r] *= inv_norm;
      }
    }
  }
}

template <typename T, typename ParallelForFn>
inline void ExecuteMoEPackedColdRoutesCpuWithScratch(
    ParallelForFn&& parallel_for,
    const T* input,
    const uint8_t* gate_weights,
    const uint8_t* up_weights,
    const uint8_t* down_weights,
    const T* gate_bias,
    const T* up_bias,
    const T* down_bias,
    int hidden_size,
    int inter_size,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int64_t gate_row_bytes,
    int64_t up_row_bytes,
    int64_t down_row_bytes,
    const int* topk_indices,
    const int* cold_route_ids,
    int num_cold_routes,
    float* x_f32,
    float* inter_f32,
    int* route_to_local_tok,
    int* unique_toks,
    T* cold_fc2_out,
    GgufQ8Block* x_q8_scratch = nullptr,
    GgufQ8Block* inter_q8_scratch = nullptr) {
  if (num_cold_routes <= 0) return;

  const int q8_blocks_per_hidden = hidden_size / 32;
  const int q8_blocks_per_inter = inter_size / 32;
  const bool gate_unquant = IsUnquantizedGgmlType(gate_type);
  const bool up_unquant = IsUnquantizedGgmlType(up_type);
  const bool down_unquant = IsUnquantizedGgmlType(down_type);
  const bool need_x_q8 = !gate_unquant || !up_unquant;
  const bool both_fc1_quant = !gate_unquant && !up_unquant;
  const bool need_inter_q8 = !down_unquant;
  std::vector<GgufQ8Block> local_x_q8;
  std::vector<GgufQ8Block> local_inter_q8;

  // Map unique token rows referenced by cold_route_ids in O(num_cold_routes) time via a direct lookup table.
  int max_tok = 0;
  for (int c = 0; c < num_cold_routes; ++c) {
    const int tok = cold_route_ids[c] / k;
    if (tok > max_tok) max_tok = tok;
  }
  int tok_to_u_stack[256];
  std::vector<int> tok_to_u_vec;
  int* tok_to_u = tok_to_u_stack;
  if (max_tok < 256) {
    std::fill_n(tok_to_u_stack, max_tok + 1, -1);
  } else {
    tok_to_u_vec.assign(static_cast<size_t>(SafeInt<size_t>(max_tok) + 1), -1);
    tok_to_u = tok_to_u_vec.data();
  }
  int num_unique_toks = 0;
  for (int c = 0; c < num_cold_routes; ++c) {
    const int tok = cold_route_ids[c] / k;
    int u = tok_to_u[tok];
    if (u < 0) {
      u = num_unique_toks++;
      tok_to_u[tok] = u;
      unique_toks[u] = tok;
    }
    route_to_local_tok[c] = u;
  }

  GgufQ8Block* x_q8 = x_q8_scratch;
  if (need_x_q8 && x_q8 == nullptr) {
    local_x_q8.resize(static_cast<size_t>(SafeInt<size_t>(num_unique_toks) * q8_blocks_per_hidden));
    x_q8 = local_x_q8.data();
  }
  GgufQ8Block* inter_q8 = inter_q8_scratch;
  if (need_inter_q8 && inter_q8 == nullptr) {
    local_inter_q8.resize(static_cast<size_t>(SafeInt<size_t>(num_cold_routes) * q8_blocks_per_inter));
    inter_q8 = local_inter_q8.data();
  }

  auto quant_in_fn = [&](std::ptrdiff_t u_idx) {
    const int u = static_cast<int>(u_idx);
    const T* src_row = input + static_cast<size_t>(unique_toks[u]) * hidden_size;
    float* dst_row = x_f32 + static_cast<size_t>(u) * hidden_size;
    for (int j = 0; j < hidden_size; ++j) {
      dst_row[j] = ActTypeToFloatCpu<T>(src_row[j]);
    }
    if (need_x_q8) {
      QuantizeRowQ8Cpu(dst_row, hidden_size, x_q8 + static_cast<size_t>(u) * q8_blocks_per_hidden);
    }
  };
  if (num_unique_toks >= 2) {
    parallel_for(static_cast<std::ptrdiff_t>(num_unique_toks), quant_in_fn);
  } else {
    for (int u = 0; u < num_unique_toks; ++u) {
      quant_in_fn(u);
    }
  }

  auto eval_gate_up_single = [&](const uint8_t* g_row, const uint8_t* u_row,
                                 const float* x_row, const GgufQ8Block* x_q8_row,
                                 float* out_g, float* out_u) {
    if (both_fc1_quant) {
      VecDotGgufGateUpRowQ8Cpu(gate_type, g_row, up_type, u_row, x_q8_row, hidden_size, out_g, out_u);
    } else if (gate_unquant && up_unquant) {
      VecDotGgufGateUpRowCpu(gate_type, g_row, up_type, u_row, x_row, hidden_size, out_g, out_u);
    } else {
      *out_g = gate_unquant ? VecDotGgufRowCpu(gate_type, g_row, x_row, hidden_size)
                            : VecDotGgufRowQ8Cpu(gate_type, g_row, x_q8_row, hidden_size);
      *out_u = up_unquant ? VecDotGgufRowCpu(up_type, u_row, x_row, hidden_size)
                          : VecDotGgufRowQ8Cpu(up_type, u_row, x_q8_row, hidden_size);
    }
  };

  auto eval_gate_up_pair = [&](const uint8_t* g_row0, const uint8_t* g_row1,
                               const uint8_t* u_row0, const uint8_t* u_row1,
                               const float* x_row, const GgufQ8Block* x_q8_row,
                               float* g_raw, float* u_raw) {
    if (both_fc1_quant) {
      VecDotGgufGateUpRow2xQ8Cpu(gate_type, g_row0, g_row1, up_type, u_row0, u_row1, x_q8_row, hidden_size, g_raw, u_raw);
    } else {
      eval_gate_up_single(g_row0, u_row0, x_row, x_q8_row, &g_raw[0], &u_raw[0]);
      eval_gate_up_single(g_row1, u_row1, x_row, x_q8_row, &g_raw[1], &u_raw[1]);
    }
  };

  constexpr int kFc1Tile = 32;
  constexpr int kFc2Tile = 64;
  const int fc1_tiles_per_route = (inter_size + kFc1Tile - 1) / kFc1Tile;
  const int fc2_tiles_per_route = (hidden_size + kFc2Tile - 1) / kFc2Tile;

  // Single-token fast path (M == 1 decode: all k routes of a token target distinct experts, so B == 1).
  if (num_unique_toks <= 1) {
    const std::ptrdiff_t total_fc1_tasks = static_cast<std::ptrdiff_t>(num_cold_routes) * fc1_tiles_per_route;
    auto fc1_fn = [&](std::ptrdiff_t task_idx) {
      const int c = static_cast<int>(task_idx) / fc1_tiles_per_route;
      const int tile_idx = static_cast<int>(task_idx) % fc1_tiles_per_route;
      const int route_id = cold_route_ids[c];
      const int expert = topk_indices[route_id];
      const size_t u_idx = static_cast<size_t>(route_to_local_tok[c]);
      const float* x_row = x_f32 + u_idx * hidden_size;
      const GgufQ8Block* x_q8_row = need_x_q8 ? (x_q8 + u_idx * q8_blocks_per_hidden) : nullptr;
      float* inter_row = inter_f32 + static_cast<size_t>(c) * inter_size;
      const int j_start = tile_idx * kFc1Tile;
      const int j_end = std::min(inter_size, j_start + kFc1Tile);
      int j = j_start;
      for (; j + 1 < j_end; j += 2) {
        const int64_t exp_row_idx0 = static_cast<int64_t>(expert) * inter_size + j;
        const int64_t exp_row_idx1 = exp_row_idx0 + 1;
        float g_raw[2];
        float u_raw[2];
        eval_gate_up_pair(
            gate_weights + exp_row_idx0 * gate_row_bytes,
            gate_weights + exp_row_idx1 * gate_row_bytes,
            up_weights + exp_row_idx0 * up_row_bytes,
            up_weights + exp_row_idx1 * up_row_bytes,
            x_row, x_q8_row, g_raw, u_raw);
        for (int m = 0; m < 2; ++m) {
          const int64_t r_idx = exp_row_idx0 + m;
          float gm = g_raw[m];
          float um = u_raw[m];
          if (gate_bias != nullptr) {
            gm += ActTypeToFloatCpu<T>(gate_bias[r_idx]);
          }
          if (up_bias != nullptr) {
            um += ActTypeToFloatCpu<T>(up_bias[r_idx]);
          }
          const float g_val = RoundToActFloatCpu<T>(gm);
          const float u_val = RoundToActFloatCpu<T>(um);
          const float swiglu = u_val * (g_val / (1.0f + std::exp(-g_val)));
          inter_row[j + m] = RoundToActFloatCpu<T>(swiglu);
        }
      }
      for (; j < j_end; ++j) {
        const int64_t exp_row_idx = static_cast<int64_t>(expert) * inter_size + j;
        float g_raw = 0.0f;
        float u_raw = 0.0f;
        eval_gate_up_single(
            gate_weights + exp_row_idx * gate_row_bytes,
            up_weights + exp_row_idx * up_row_bytes,
            x_row, x_q8_row, &g_raw, &u_raw);
        if (gate_bias != nullptr) {
          g_raw += ActTypeToFloatCpu<T>(gate_bias[exp_row_idx]);
        }
        if (up_bias != nullptr) {
          u_raw += ActTypeToFloatCpu<T>(up_bias[exp_row_idx]);
        }
        const float g_val = RoundToActFloatCpu<T>(g_raw);
        const float u_val = RoundToActFloatCpu<T>(u_raw);
        const float swiglu = u_val * (g_val / (1.0f + std::exp(-g_val)));
        inter_row[j] = RoundToActFloatCpu<T>(swiglu);
      }
      if (need_inter_q8) {
        QuantizeBlockQ8Cpu(
            inter_row + j_start,
            &inter_q8[static_cast<size_t>(c) * q8_blocks_per_inter + tile_idx]);
      }
    };
    parallel_for(total_fc1_tasks, fc1_fn);

    const std::ptrdiff_t total_fc2_tasks = static_cast<std::ptrdiff_t>(num_cold_routes) * fc2_tiles_per_route;
    auto fc2_fn = [&](std::ptrdiff_t task_idx) {
      const int c = static_cast<int>(task_idx) / fc2_tiles_per_route;
      const int tile_idx = static_cast<int>(task_idx) % fc2_tiles_per_route;
      const int route_id = cold_route_ids[c];
      const int expert = topk_indices[route_id];
      const float* inter_row = inter_f32 + static_cast<size_t>(c) * inter_size;
      const GgufQ8Block* inter_q8_row = need_inter_q8 ? (inter_q8 + static_cast<size_t>(c) * q8_blocks_per_inter) : nullptr;
      T* dst_row = cold_fc2_out + static_cast<size_t>(c) * hidden_size;
      const int n_start = tile_idx * kFc2Tile;
      const int n_end = std::min(hidden_size, n_start + kFc2Tile);
      int n = n_start;
      for (; n + 3 < n_end; n += 4) {
        const int64_t exp_row_idx = static_cast<int64_t>(expert) * hidden_size + n;
        const uint8_t* r0 = down_weights + (exp_row_idx + 0) * down_row_bytes;
        const uint8_t* r1 = down_weights + (exp_row_idx + 1) * down_row_bytes;
        const uint8_t* r2 = down_weights + (exp_row_idx + 2) * down_row_bytes;
        const uint8_t* r3 = down_weights + (exp_row_idx + 3) * down_row_bytes;
        float d4[4];
        if (down_unquant) {
          VecDotGgufRow4xCpu(down_type, r0, r1, r2, r3, inter_row, inter_size, d4);
        } else {
          VecDotGgufRow4xQ8Cpu(down_type, r0, r1, r2, r3, inter_q8_row, inter_size, d4);
        }
        if (down_bias != nullptr) {
          for (int m = 0; m < 4; ++m) {
            d4[m] += ActTypeToFloatCpu<T>(down_bias[exp_row_idx + m]);
          }
        }
        for (int m = 0; m < 4; ++m) {
          dst_row[n + m] = FloatToActTypeCpu<T>(d4[m]);
        }
      }
      for (; n < n_end; ++n) {
        const int64_t exp_row_idx = static_cast<int64_t>(expert) * hidden_size + n;
        const uint8_t* r0 = down_weights + exp_row_idx * down_row_bytes;
        float d_raw = down_unquant ? VecDotGgufRowCpu(down_type, r0, inter_row, inter_size)
                                   : VecDotGgufRowQ8Cpu(down_type, r0, inter_q8_row, inter_size);
        if (down_bias != nullptr) {
          d_raw += ActTypeToFloatCpu<T>(down_bias[exp_row_idx]);
        }
        dst_row[n] = FloatToActTypeCpu<T>(d_raw);
      }
    };
    parallel_for(total_fc2_tasks, fc2_fn);
    return;
  }

  // Multi-token prefill (M > 1): group cold routes by expert_id so all B tokens routed to the same
  // cold expert share each L1/L2-cached GGUF weight row pair/quad instead of re-streaming from DRAM.
  struct ColdExpertGroup {
    int expert;
    int start;
    int count;
  };
  int sorted_c_stack[128];
  ColdExpertGroup groups_stack[128];
  std::vector<int> sorted_c_vec;
  std::vector<ColdExpertGroup> groups_vec;
  int* sorted_c = sorted_c_stack;
  ColdExpertGroup* groups = groups_stack;
  if (num_cold_routes > 128) {
    sorted_c_vec.resize(static_cast<size_t>(num_cold_routes));
    groups_vec.resize(static_cast<size_t>(num_cold_routes));
    sorted_c = sorted_c_vec.data();
    groups = groups_vec.data();
  }
  for (int c = 0; c < num_cold_routes; ++c) {
    sorted_c[c] = c;
  }
  std::sort(sorted_c, sorted_c + num_cold_routes, [&](int a, int b) {
    const int ea = topk_indices[cold_route_ids[a]];
    const int eb = topk_indices[cold_route_ids[b]];
    return (ea != eb) ? (ea < eb) : (a < b);
  });

  int num_groups = 0;
  for (int idx = 0; idx < num_cold_routes;) {
    const int e = topk_indices[cold_route_ids[sorted_c[idx]]];
    int end = idx + 1;
    while (end < num_cold_routes &&
           topk_indices[cold_route_ids[sorted_c[end]]] == e) {
      ++end;
    }
    groups[num_groups++] = ColdExpertGroup{e, idx, end - idx};
    idx = end;
  }

  // Phase 1 (Expert-Grouped): Fused Gate + Up + SwiGLU + Q8 intermediate quantization.
  const std::ptrdiff_t total_fc1_tasks = static_cast<std::ptrdiff_t>(num_groups) * fc1_tiles_per_route;
  auto fc1_grouped_fn = [&](std::ptrdiff_t task_idx) {
    const int g_idx = static_cast<int>(task_idx) / fc1_tiles_per_route;
    const int tile_idx = static_cast<int>(task_idx) % fc1_tiles_per_route;
    const ColdExpertGroup grp = groups[g_idx];
    const int expert = grp.expert;
    const int j_start = tile_idx * kFc1Tile;
    const int j_end = std::min(inter_size, j_start + kFc1Tile);
    int j = j_start;
    for (; j + 1 < j_end; j += 2) {
      const int64_t exp_row_idx0 = static_cast<int64_t>(expert) * inter_size + j;
      const int64_t exp_row_idx1 = exp_row_idx0 + 1;
      const uint8_t* g_row0 = gate_weights + exp_row_idx0 * gate_row_bytes;
      const uint8_t* g_row1 = gate_weights + exp_row_idx1 * gate_row_bytes;
      const uint8_t* u_row0 = up_weights + exp_row_idx0 * up_row_bytes;
      const uint8_t* u_row1 = up_weights + exp_row_idx1 * up_row_bytes;
      const float gb0 = (gate_bias != nullptr) ? ActTypeToFloatCpu<T>(gate_bias[exp_row_idx0]) : 0.0f;
      const float gb1 = (gate_bias != nullptr) ? ActTypeToFloatCpu<T>(gate_bias[exp_row_idx1]) : 0.0f;
      const float ub0 = (up_bias != nullptr) ? ActTypeToFloatCpu<T>(up_bias[exp_row_idx0]) : 0.0f;
      const float ub1 = (up_bias != nullptr) ? ActTypeToFloatCpu<T>(up_bias[exp_row_idx1]) : 0.0f;

      int b = 0;
      if (both_fc1_quant) {
        for (; b + 3 < grp.count; b += 4) {
          const int ca = sorted_c[grp.start + b];
          const int cb = sorted_c[grp.start + b + 1];
          const int cc = sorted_c[grp.start + b + 2];
          const int cd = sorted_c[grp.start + b + 3];
          const GgufQ8Block* x_qa = x_q8 + static_cast<size_t>(route_to_local_tok[ca]) * q8_blocks_per_hidden;
          const GgufQ8Block* x_qb = x_q8 + static_cast<size_t>(route_to_local_tok[cb]) * q8_blocks_per_hidden;
          const GgufQ8Block* x_qc = x_q8 + static_cast<size_t>(route_to_local_tok[cc]) * q8_blocks_per_hidden;
          const GgufQ8Block* x_qd = x_q8 + static_cast<size_t>(route_to_local_tok[cd]) * q8_blocks_per_hidden;
          float ga[2], ua[2], gb[2], ub[2], gc[2], uc[2], gd[2], ud[2];
          VecDotGgufGateUpRow2xQ8Batch4Cpu(
              gate_type, g_row0, g_row1, up_type, u_row0, u_row1,
              x_qa, x_qb, x_qc, x_qd, hidden_size,
              ga, ua, gb, ub, gc, uc, gd, ud);
          const int c_arr[4] = {ca, cb, cc, cd};
          const float* g_arr[4] = {ga, gb, gc, gd};
          const float* u_arr[4] = {ua, ub, uc, ud};
          for (int sub = 0; sub < 4; ++sub) {
            float* inter_s = inter_f32 + static_cast<size_t>(c_arr[sub]) * inter_size;
            const float gs0 = RoundToActFloatCpu<T>(g_arr[sub][0] + gb0);
            const float us0 = RoundToActFloatCpu<T>(u_arr[sub][0] + ub0);
            const float gs1 = RoundToActFloatCpu<T>(g_arr[sub][1] + gb1);
            const float us1 = RoundToActFloatCpu<T>(u_arr[sub][1] + ub1);
            inter_s[j + 0] = RoundToActFloatCpu<T>(us0 * (gs0 / (1.0f + std::exp(-gs0))));
            inter_s[j + 1] = RoundToActFloatCpu<T>(us1 * (gs1 / (1.0f + std::exp(-gs1))));
          }
        }
        for (; b + 1 < grp.count; b += 2) {
          const int ca = sorted_c[grp.start + b];
          const int cb = sorted_c[grp.start + b + 1];
          const GgufQ8Block* x_qa = x_q8 + static_cast<size_t>(route_to_local_tok[ca]) * q8_blocks_per_hidden;
          const GgufQ8Block* x_qb = x_q8 + static_cast<size_t>(route_to_local_tok[cb]) * q8_blocks_per_hidden;
          float* inter_a = inter_f32 + static_cast<size_t>(ca) * inter_size;
          float* inter_b = inter_f32 + static_cast<size_t>(cb) * inter_size;
          float ga[2], ua[2], gb[2], ub[2];
          VecDotGgufGateUpRow2xQ8Batch2Cpu(
              gate_type, g_row0, g_row1, up_type, u_row0, u_row1, x_qa, x_qb, hidden_size, ga, ua, gb, ub);
          const float ga0 = RoundToActFloatCpu<T>(ga[0] + gb0);
          const float ua0 = RoundToActFloatCpu<T>(ua[0] + ub0);
          const float ga1 = RoundToActFloatCpu<T>(ga[1] + gb1);
          const float ua1 = RoundToActFloatCpu<T>(ua[1] + ub1);
          inter_a[j + 0] = RoundToActFloatCpu<T>(ua0 * (ga0 / (1.0f + std::exp(-ga0))));
          inter_a[j + 1] = RoundToActFloatCpu<T>(ua1 * (ga1 / (1.0f + std::exp(-ga1))));
          const float gb_0 = RoundToActFloatCpu<T>(gb[0] + gb0);
          const float ub_0 = RoundToActFloatCpu<T>(ub[0] + ub0);
          const float gb_1 = RoundToActFloatCpu<T>(gb[1] + gb1);
          const float ub_1 = RoundToActFloatCpu<T>(ub[1] + ub1);
          inter_b[j + 0] = RoundToActFloatCpu<T>(ub_0 * (gb_0 / (1.0f + std::exp(-gb_0))));
          inter_b[j + 1] = RoundToActFloatCpu<T>(ub_1 * (gb_1 / (1.0f + std::exp(-gb_1))));
        }
      }
      for (; b < grp.count; ++b) {
        const int c = sorted_c[grp.start + b];
        const size_t u_idx = static_cast<size_t>(route_to_local_tok[c]);
        const float* x_row = x_f32 + u_idx * hidden_size;
        const GgufQ8Block* x_q8_row = need_x_q8 ? (x_q8 + u_idx * q8_blocks_per_hidden) : nullptr;
        float* inter_row = inter_f32 + static_cast<size_t>(c) * inter_size;
        float g_raw[2];
        float u_raw[2];
        eval_gate_up_pair(g_row0, g_row1, u_row0, u_row1, x_row, x_q8_row, g_raw, u_raw);
        const float g0 = RoundToActFloatCpu<T>(g_raw[0] + gb0);
        const float u0 = RoundToActFloatCpu<T>(u_raw[0] + ub0);
        const float g1 = RoundToActFloatCpu<T>(g_raw[1] + gb1);
        const float u1 = RoundToActFloatCpu<T>(u_raw[1] + ub1);
        inter_row[j + 0] = RoundToActFloatCpu<T>(u0 * (g0 / (1.0f + std::exp(-g0))));
        inter_row[j + 1] = RoundToActFloatCpu<T>(u1 * (g1 / (1.0f + std::exp(-g1))));
      }
    }
    for (; j < j_end; ++j) {
      const int64_t exp_row_idx = static_cast<int64_t>(expert) * inter_size + j;
      const uint8_t* g_row = gate_weights + exp_row_idx * gate_row_bytes;
      const uint8_t* u_row = up_weights + exp_row_idx * up_row_bytes;
      const float gb = (gate_bias != nullptr) ? ActTypeToFloatCpu<T>(gate_bias[exp_row_idx]) : 0.0f;
      const float ub = (up_bias != nullptr) ? ActTypeToFloatCpu<T>(up_bias[exp_row_idx]) : 0.0f;
      for (int b = 0; b < grp.count; ++b) {
        const int c = sorted_c[grp.start + b];
        const size_t u_idx = static_cast<size_t>(route_to_local_tok[c]);
        const float* x_row = x_f32 + u_idx * hidden_size;
        const GgufQ8Block* x_q8_row = need_x_q8 ? (x_q8 + u_idx * q8_blocks_per_hidden) : nullptr;
        float* inter_row = inter_f32 + static_cast<size_t>(c) * inter_size;
        float g_raw = 0.0f;
        float u_raw = 0.0f;
        eval_gate_up_single(g_row, u_row, x_row, x_q8_row, &g_raw, &u_raw);
        const float g_val = RoundToActFloatCpu<T>(g_raw + gb);
        const float u_val = RoundToActFloatCpu<T>(u_raw + ub);
        inter_row[j] = RoundToActFloatCpu<T>(u_val * (g_val / (1.0f + std::exp(-g_val))));
      }
    }
    if (need_inter_q8) {
      for (int b = 0; b < grp.count; ++b) {
        const int c = sorted_c[grp.start + b];
        float* inter_row = inter_f32 + static_cast<size_t>(c) * inter_size;
        QuantizeBlockQ8Cpu(
            inter_row + j_start,
            &inter_q8[static_cast<size_t>(c) * q8_blocks_per_inter + tile_idx]);
      }
    }
  };
  parallel_for(total_fc1_tasks, fc1_grouped_fn);

  // Phase 2 (Expert-Grouped): Down projection reusing 4 weight rows in L1D across all B tokens in the group.
  const std::ptrdiff_t total_fc2_tasks = static_cast<std::ptrdiff_t>(num_groups) * fc2_tiles_per_route;
  auto fc2_grouped_fn = [&](std::ptrdiff_t task_idx) {
    const int g_idx = static_cast<int>(task_idx) / fc2_tiles_per_route;
    const int tile_idx = static_cast<int>(task_idx) % fc2_tiles_per_route;
    const ColdExpertGroup grp = groups[g_idx];
    const int expert = grp.expert;
    const int n_start = tile_idx * kFc2Tile;
    const int n_end = std::min(hidden_size, n_start + kFc2Tile);
    int n = n_start;
    for (; n + 3 < n_end; n += 4) {
      const int64_t exp_row_idx = static_cast<int64_t>(expert) * hidden_size + n;
      const uint8_t* r0 = down_weights + (exp_row_idx + 0) * down_row_bytes;
      const uint8_t* r1 = down_weights + (exp_row_idx + 1) * down_row_bytes;
      const uint8_t* r2 = down_weights + (exp_row_idx + 2) * down_row_bytes;
      const uint8_t* r3 = down_weights + (exp_row_idx + 3) * down_row_bytes;
      float db4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (down_bias != nullptr) {
        for (int m = 0; m < 4; ++m) {
          db4[m] = ActTypeToFloatCpu<T>(down_bias[exp_row_idx + m]);
        }
      }
      int b = 0;
      if (!down_unquant) {
        for (; b + 3 < grp.count; b += 4) {
          const int ca = sorted_c[grp.start + b];
          const int cb = sorted_c[grp.start + b + 1];
          const int cc = sorted_c[grp.start + b + 2];
          const int cd = sorted_c[grp.start + b + 3];
          const GgufQ8Block* qa = inter_q8 + static_cast<size_t>(ca) * q8_blocks_per_inter;
          const GgufQ8Block* qb = inter_q8 + static_cast<size_t>(cb) * q8_blocks_per_inter;
          const GgufQ8Block* qc = inter_q8 + static_cast<size_t>(cc) * q8_blocks_per_inter;
          const GgufQ8Block* qd = inter_q8 + static_cast<size_t>(cd) * q8_blocks_per_inter;
          float d4_a[4], d4_b[4], d4_c[4], d4_d[4];
          VecDotGgufRow4xQ8Batch4Cpu(down_type, r0, r1, r2, r3, qa, qb, qc, qd, inter_size, d4_a, d4_b, d4_c, d4_d);
          const int c_arr[4] = {ca, cb, cc, cd};
          const float* d4_arr[4] = {d4_a, d4_b, d4_c, d4_d};
          for (int sub = 0; sub < 4; ++sub) {
            T* dst_s = cold_fc2_out + static_cast<size_t>(c_arr[sub]) * hidden_size;
            for (int m = 0; m < 4; ++m) {
              dst_s[n + m] = FloatToActTypeCpu<T>(d4_arr[sub][m] + db4[m]);
            }
          }
        }
        for (; b + 1 < grp.count; b += 2) {
          const int ca = sorted_c[grp.start + b];
          const int cb = sorted_c[grp.start + b + 1];
          const GgufQ8Block* qa = inter_q8 + static_cast<size_t>(ca) * q8_blocks_per_inter;
          const GgufQ8Block* qb = inter_q8 + static_cast<size_t>(cb) * q8_blocks_per_inter;
          T* dst_a = cold_fc2_out + static_cast<size_t>(ca) * hidden_size;
          T* dst_b = cold_fc2_out + static_cast<size_t>(cb) * hidden_size;
          float d4_a[4], d4_b[4];
          VecDotGgufRow4xQ8Batch2Cpu(down_type, r0, r1, r2, r3, qa, qb, inter_size, d4_a, d4_b);
          for (int m = 0; m < 4; ++m) {
            dst_a[n + m] = FloatToActTypeCpu<T>(d4_a[m] + db4[m]);
            dst_b[n + m] = FloatToActTypeCpu<T>(d4_b[m] + db4[m]);
          }
        }
      }
      for (; b < grp.count; ++b) {
        const int c = sorted_c[grp.start + b];
        const float* inter_row = inter_f32 + static_cast<size_t>(c) * inter_size;
        const GgufQ8Block* inter_q8_row = need_inter_q8 ? (inter_q8 + static_cast<size_t>(c) * q8_blocks_per_inter) : nullptr;
        T* dst_row = cold_fc2_out + static_cast<size_t>(c) * hidden_size;
        float d4[4];
        if (down_unquant) {
          VecDotGgufRow4xCpu(down_type, r0, r1, r2, r3, inter_row, inter_size, d4);
        } else {
          VecDotGgufRow4xQ8Cpu(down_type, r0, r1, r2, r3, inter_q8_row, inter_size, d4);
        }
        for (int m = 0; m < 4; ++m) {
          dst_row[n + m] = FloatToActTypeCpu<T>(d4[m] + db4[m]);
        }
      }
    }
    for (; n < n_end; ++n) {
      const int64_t exp_row_idx = static_cast<int64_t>(expert) * hidden_size + n;
      const uint8_t* r0 = down_weights + exp_row_idx * down_row_bytes;
      const float db = (down_bias != nullptr) ? ActTypeToFloatCpu<T>(down_bias[exp_row_idx]) : 0.0f;
      for (int b = 0; b < grp.count; ++b) {
        const int c = sorted_c[grp.start + b];
        const float* inter_row = inter_f32 + static_cast<size_t>(c) * inter_size;
        const GgufQ8Block* inter_q8_row = need_inter_q8 ? (inter_q8 + static_cast<size_t>(c) * q8_blocks_per_inter) : nullptr;
        T* dst_row = cold_fc2_out + static_cast<size_t>(c) * hidden_size;
        const float d_raw = down_unquant ? VecDotGgufRowCpu(down_type, r0, inter_row, inter_size)
                                         : VecDotGgufRowQ8Cpu(down_type, r0, inter_q8_row, inter_size);
        dst_row[n] = FloatToActTypeCpu<T>(d_raw + db);
      }
    }
  };
  parallel_for(total_fc2_tasks, fc2_grouped_fn);
}

template <typename T, typename ParallelForFn>
inline void ExecuteMoEPackedColdRoutesCpu(
    ParallelForFn&& parallel_for,
    const T* input,
    const uint8_t* gate_weights,
    const uint8_t* up_weights,
    const uint8_t* down_weights,
    const T* gate_bias,
    const T* up_bias,
    const T* down_bias,
    int hidden_size,
    int inter_size,
    int k,
    int gate_type,
    int up_type,
    int down_type,
    int64_t gate_row_bytes,
    int64_t up_row_bytes,
    int64_t down_row_bytes,
    const int* topk_indices,
    const int* cold_route_ids,
    int num_cold_routes,
    T* cold_fc2_out) {
  if (num_cold_routes <= 0) return;
  int max_tok = 0;
  for (int c = 0; c < num_cold_routes; ++c) {
    const int tok = cold_route_ids[c] / k;
    if (tok > max_tok) max_tok = tok;
  }
  int tok_to_u_stack[256];
  std::vector<int> tok_to_u_vec;
  int* tok_to_u = tok_to_u_stack;
  if (max_tok < 256) {
    std::fill_n(tok_to_u_stack, max_tok + 1, -1);
  } else {
    tok_to_u_vec.assign(static_cast<size_t>(SafeInt<size_t>(max_tok) + 1), -1);
    tok_to_u = tok_to_u_vec.data();
  }
  int num_unique_toks = 0;
  for (int c = 0; c < num_cold_routes; ++c) {
    const int tok = cold_route_ids[c] / k;
    if (tok_to_u[tok] < 0) {
      tok_to_u[tok] = num_unique_toks++;
    }
  }
  int route_to_local_tok_stack[128];
  int unique_toks_stack[128];
  std::vector<int> route_to_local_tok_vec;
  std::vector<int> unique_toks_vec;
  int* route_to_local_tok = route_to_local_tok_stack;
  int* unique_toks = unique_toks_stack;
  if (num_cold_routes > 128) {
    route_to_local_tok_vec.resize(static_cast<size_t>(num_cold_routes));
    unique_toks_vec.resize(static_cast<size_t>(num_unique_toks));
    route_to_local_tok = route_to_local_tok_vec.data();
    unique_toks = unique_toks_vec.data();
  }
  std::vector<float> x_f32(static_cast<size_t>(SafeInt<size_t>(num_unique_toks) * hidden_size));
  std::vector<float> inter_f32(static_cast<size_t>(SafeInt<size_t>(num_cold_routes) * inter_size));
  ExecuteMoEPackedColdRoutesCpuWithScratch<T>(
      std::forward<ParallelForFn>(parallel_for),
      input, gate_weights, up_weights, down_weights,
      gate_bias, up_bias, down_bias,
      hidden_size, inter_size, k,
      gate_type, up_type, down_type,
      gate_row_bytes, up_row_bytes, down_row_bytes,
      topk_indices, cold_route_ids, num_cold_routes,
      x_f32.data(), inter_f32.data(),
      route_to_local_tok, unique_toks,
      cold_fc2_out);
}

#endif  // !defined(__CUDA_ARCH__)

}  // namespace gguf_quant
#if !defined(__CUDA_ARCH__)
using gguf_quant::MoEPackedCpuColdScratch;
#endif
}  // namespace contrib
}  // namespace onnxruntime
