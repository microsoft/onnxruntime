// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AVX-512 CPU NVFP4 (block-16 E4M3) and MXFP4 (block-32 E8M0) expert kernels for QMoE.

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
#include <atomic>
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

namespace {

alignas(64) constexpr float kAvx512Fp4E2M1Table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float Avx512Fp4HalfBitsToFloat(uint16_t h) noexcept {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

inline __m512 Avx512Fp4RoundBf16Vec(__m512 v) noexcept {
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

inline __m512 Avx512Fp4RoundFp16Vec(__m512 v) noexcept {
  return _mm512_cvtph_ps(_mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
}

std::atomic<int> g_fp4_group8_mask_override{-1};

const float* GetAvx512E4M3Table() noexcept {
  static const auto table = []() {
    std::array<float, 256> t{};
    for (int i = 0; i < 256; ++i) t[i] = DecodeFloat8E4M3FN(static_cast<uint8_t>(i));
    return t;
  }();
  return table.data();
}

const float* GetAvx512E8M0Table() noexcept {
  static const auto table = []() {
    std::array<float, 256> t{};
    for (int i = 0; i < 256; ++i) t[i] = DecodeFloat8E8M0(static_cast<uint8_t>(i));
    return t;
  }();
  return table.data();
}

struct ScaleFp4LutCacheEntry {
  uint64_t key = ~uint64_t{0};
  alignas(64) __m512 lut_2d[256];
};

struct ScaleFp4LutTlsCache {
  static constexpr size_t kCacheWays = 16;
  void* raw_ways[kCacheWays]{};
  ScaleFp4LutCacheEntry* ways[kCacheWays]{};

  ~ScaleFp4LutTlsCache() noexcept {
    for (size_t i = 0; i < kCacheWays; ++i) {
      std::free(raw_ways[i]);
    }
  }

  ScaleFp4LutCacheEntry* GetOrAllocateWay(size_t slot) noexcept {
    if (ways[slot] != nullptr) return ways[slot];
    void* raw = std::malloc(sizeof(ScaleFp4LutCacheEntry) + 64);
    if (raw == nullptr) return nullptr;
    const uintptr_t addr = (reinterpret_cast<uintptr_t>(raw) + 63u) & ~uintptr_t{63u};
    auto* entry = new (reinterpret_cast<void*>(addr)) ScaleFp4LutCacheEntry();
    raw_ways[slot] = raw;
    ways[slot] = entry;
    return entry;
  }
};

struct ScaleFp4LutTlsHandle {
  ScaleFp4LutTlsCache* ptr = nullptr;
  ~ScaleFp4LutTlsHandle() noexcept { delete ptr; }
};

inline ScaleFp4LutTlsCache* GetScaleFp4TlsCache() noexcept {
  thread_local ScaleFp4LutTlsHandle handle{};
  if (handle.ptr == nullptr) {
    handle.ptr = new (std::nothrow) ScaleFp4LutTlsCache();
  }
  return handle.ptr;
}

inline void PopulateScaleFp4Lut2D(const float* base_lut, float global_scale, bool round_scale_bf16,
                                  __m512 (&lut_2d)[256]) noexcept {
  const __m512 fp4_v = _mm512_load_ps(kAvx512Fp4E2M1Table);
  const __m512 gs_v = _mm512_set1_ps(global_scale);
  alignas(64) float rounded_scales[256];
  for (int i = 0; i < 256; i += 16) {
    const __m512 raw_v = _mm512_mul_ps(_mm512_loadu_ps(base_lut + i), gs_v);
    const __m512 rnd_v = round_scale_bf16 ? Avx512Fp4RoundBf16Vec(raw_v) : Avx512Fp4RoundFp16Vec(raw_v);
    _mm512_store_ps(rounded_scales + i, rnd_v);
  }
  if (round_scale_bf16) {
    for (int i = 0; i < 256; ++i) {
      lut_2d[i] = Avx512Fp4RoundBf16Vec(_mm512_mul_ps(fp4_v, _mm512_set1_ps(rounded_scales[i])));
    }
  } else {
    for (int i = 0; i < 256; ++i) {
      lut_2d[i] = Avx512Fp4RoundFp16Vec(_mm512_mul_ps(fp4_v, _mm512_set1_ps(rounded_scales[i])));
    }
  }
}

template <bool IsMxfp4>
const __m512 (*GetScaleFp4Lut2D(float global_scale, bool round_scale_bf16) noexcept)[256] {
  uint32_t gs_bits = 0;
  std::memcpy(&gs_bits, &global_scale, sizeof(gs_bits));
  const uint64_t key = (static_cast<uint64_t>(gs_bits) << 2) |
                       (static_cast<uint64_t>(round_scale_bf16 ? 1 : 0) << 1) |
                       static_cast<uint64_t>(IsMxfp4 ? 1 : 0);
  const size_t slot = ((gs_bits >> 5) ^ (gs_bits >> 13) ^ (gs_bits >> 21) ^
                       (round_scale_bf16 ? 0x1Fu : 0u) ^ (IsMxfp4 ? 0x07u : 0u)) &
                      (ScaleFp4LutTlsCache::kCacheWays - 1);
  const float* base_lut = IsMxfp4 ? GetAvx512E8M0Table() : GetAvx512E4M3Table();
  if (ScaleFp4LutTlsCache* cache = GetScaleFp4TlsCache()) {
    if (ScaleFp4LutCacheEntry* entry = cache->GetOrAllocateWay(slot)) {
      if (entry->key != key) {
        PopulateScaleFp4Lut2D(base_lut, global_scale, round_scale_bf16, entry->lut_2d);
        entry->key = key;
      }
      return &entry->lut_2d;
    }
  }
  return nullptr;
}

template <int RT, bool IsMxfp4>
void Fp4Group8Avx512(const uint8_t* w_base, const uint8_t* s_base, const __m512 (&lut_2d)[256], int64_t k,
                     const float* a_reordered, float (&result)[8][RT]) {
  const int64_t k_pairs = k / 32;
  const int64_t w_stride = k / 2;
  const int64_t s_stride = IsMxfp4 ? (k / 32) : (k / 16);
  const __m128i mask128 = _mm_set1_epi8(0x0F);
  const __m128i hi_blk = _mm_set_epi64x(0x1010101010101010ULL, 0ULL);

  __m512 acc0_0 = _mm512_setzero_ps(), acc0_1 = _mm512_setzero_ps(), acc0_2 = _mm512_setzero_ps(), acc0_3 = _mm512_setzero_ps();
  __m512 acc1_0 = _mm512_setzero_ps(), acc1_1 = _mm512_setzero_ps(), acc1_2 = _mm512_setzero_ps(), acc1_3 = _mm512_setzero_ps();
  __m512 acc2_0 = _mm512_setzero_ps(), acc2_1 = _mm512_setzero_ps(), acc2_2 = _mm512_setzero_ps(), acc2_3 = _mm512_setzero_ps();
  __m512 acc3_0 = _mm512_setzero_ps(), acc3_1 = _mm512_setzero_ps(), acc3_2 = _mm512_setzero_ps(), acc3_3 = _mm512_setzero_ps();
  __m512 acc4_0 = _mm512_setzero_ps(), acc4_1 = _mm512_setzero_ps(), acc4_2 = _mm512_setzero_ps(), acc4_3 = _mm512_setzero_ps();
  __m512 acc5_0 = _mm512_setzero_ps(), acc5_1 = _mm512_setzero_ps(), acc5_2 = _mm512_setzero_ps(), acc5_3 = _mm512_setzero_ps();
  __m512 acc6_0 = _mm512_setzero_ps(), acc6_1 = _mm512_setzero_ps(), acc6_2 = _mm512_setzero_ps(), acc6_3 = _mm512_setzero_ps();
  __m512 acc7_0 = _mm512_setzero_ps(), acc7_1 = _mm512_setzero_ps(), acc7_2 = _mm512_setzero_ps(), acc7_3 = _mm512_setzero_ps();

  const uint8_t* w_rows[8];
  const uint8_t* s_rows[8];
#if defined(__GNUC__)
#pragma GCC unroll 8
#endif
  for (int c = 0; c < 8; ++c) {
    w_rows[c] = w_base + c * w_stride;
    s_rows[c] = s_base + c * s_stride;
  }

  for (int64_t kp = 0; kp < k_pairs; ++kp) {
    const __m512 al0 = _mm512_loadu_ps(a_reordered + 0 * k + kp * 32);
    const __m512 ah0 = _mm512_loadu_ps(a_reordered + 0 * k + kp * 32 + 16);
    __m512 al1, ah1, al2, ah2, al3, ah3;
    if constexpr (RT >= 2) {
      al1 = _mm512_loadu_ps(a_reordered + 1 * k + kp * 32);
      ah1 = _mm512_loadu_ps(a_reordered + 1 * k + kp * 32 + 16);
    }
    if constexpr (RT >= 3) {
      al2 = _mm512_loadu_ps(a_reordered + 2 * k + kp * 32);
      ah2 = _mm512_loadu_ps(a_reordered + 2 * k + kp * 32 + 16);
    }
    if constexpr (RT >= 4) {
      al3 = _mm512_loadu_ps(a_reordered + 3 * k + kp * 32);
      ah3 = _mm512_loadu_ps(a_reordered + 3 * k + kp * 32 + 16);
    }
    auto step_col = [&](int c, __m512& a0, __m512& a1, __m512& a2, __m512& a3) QMOE_LAMBDA_INLINE {
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_rows[c] + kp * 16));
      __m512 wls, whs;
      if constexpr (IsMxfp4) {
        const __m128i lo = _mm_and_si128(x, mask128);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), mask128);
        const __m512 lut = lut_2d[s_rows[c][kp]];
        wls = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(lo), lut);
        whs = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(hi), lut);
      } else {
        const __m128i lo = _mm_or_si128(_mm_and_si128(x, mask128), hi_blk);
        const __m128i hi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(x, 4), mask128), hi_blk);
        const __m512 lut0 = lut_2d[s_rows[c][kp * 2]];
        const __m512 lut1 = lut_2d[s_rows[c][kp * 2 + 1]];
        wls = _mm512_permutex2var_ps(lut0, _mm512_cvtepu8_epi32(lo), lut1);
        whs = _mm512_permutex2var_ps(lut0, _mm512_cvtepu8_epi32(hi), lut1);
      }
      a0 = _mm512_fmadd_ps(wls, al0, _mm512_fmadd_ps(whs, ah0, a0));
      if constexpr (RT >= 2) a1 = _mm512_fmadd_ps(wls, al1, _mm512_fmadd_ps(whs, ah1, a1));
      if constexpr (RT >= 3) a2 = _mm512_fmadd_ps(wls, al2, _mm512_fmadd_ps(whs, ah2, a2));
      if constexpr (RT >= 4) a3 = _mm512_fmadd_ps(wls, al3, _mm512_fmadd_ps(whs, ah3, a3));
    };
    step_col(0, acc0_0, acc0_1, acc0_2, acc0_3);
    step_col(1, acc1_0, acc1_1, acc1_2, acc1_3);
    step_col(2, acc2_0, acc2_1, acc2_2, acc2_3);
    step_col(3, acc3_0, acc3_1, acc3_2, acc3_3);
    step_col(4, acc4_0, acc4_1, acc4_2, acc4_3);
    step_col(5, acc5_0, acc5_1, acc5_2, acc5_3);
    step_col(6, acc6_0, acc6_1, acc6_2, acc6_3);
    step_col(7, acc7_0, acc7_1, acc7_2, acc7_3);
  }
  auto store_col = [&](int c, __m512 a0, __m512 a1, __m512 a2, __m512 a3) QMOE_LAMBDA_INLINE {
    result[c][0] = _mm512_reduce_add_ps(a0);
    if constexpr (RT >= 2) result[c][1] = _mm512_reduce_add_ps(a1);
    if constexpr (RT >= 3) result[c][2] = _mm512_reduce_add_ps(a2);
    if constexpr (RT >= 4) result[c][3] = _mm512_reduce_add_ps(a3);
  };
  store_col(0, acc0_0, acc0_1, acc0_2, acc0_3);
  store_col(1, acc1_0, acc1_1, acc1_2, acc1_3);
  store_col(2, acc2_0, acc2_1, acc2_2, acc2_3);
  store_col(3, acc3_0, acc3_1, acc3_2, acc3_3);
  store_col(4, acc4_0, acc4_1, acc4_2, acc4_3);
  store_col(5, acc5_0, acc5_1, acc5_2, acc5_3);
  store_col(6, acc6_0, acc6_1, acc6_2, acc6_3);
  store_col(7, acc7_0, acc7_1, acc7_2, acc7_3);
}

template <int RT, bool IsMxfp4>
void Fp4Group4Avx512(const uint8_t* w_base, const uint8_t* next_w_base, const uint8_t* s_base,
                     const __m512 (&lut_2d)[256], int64_t k, const float* a_reordered, float (&result)[4][RT]) {
  const int64_t k_pairs = k / 32;
  const int64_t w_stride = k / 2;
  const int64_t s_stride = IsMxfp4 ? (k / 32) : (k / 16);
  const __m128i mask128 = _mm_set1_epi8(0x0F);
  const __m128i hi_blk = _mm_set_epi64x(0x1010101010101010ULL, 0ULL);

  __m512 acc0_0 = _mm512_setzero_ps(), acc0_1 = _mm512_setzero_ps(), acc0_2 = _mm512_setzero_ps(), acc0_3 = _mm512_setzero_ps();
  __m512 acc1_0 = _mm512_setzero_ps(), acc1_1 = _mm512_setzero_ps(), acc1_2 = _mm512_setzero_ps(), acc1_3 = _mm512_setzero_ps();
  __m512 acc2_0 = _mm512_setzero_ps(), acc2_1 = _mm512_setzero_ps(), acc2_2 = _mm512_setzero_ps(), acc2_3 = _mm512_setzero_ps();
  __m512 acc3_0 = _mm512_setzero_ps(), acc3_1 = _mm512_setzero_ps(), acc3_2 = _mm512_setzero_ps(), acc3_3 = _mm512_setzero_ps();

  const uint8_t* w_rows[4] = {w_base, w_base + w_stride, w_base + 2 * w_stride, w_base + 3 * w_stride};
  const uint8_t* s_rows[4] = {s_base, s_base + s_stride, s_base + 2 * s_stride, s_base + 3 * s_stride};

  for (int64_t kp = 0; kp < k_pairs; ++kp) {
    if (next_w_base != nullptr && (kp & 3) == 0) {
      const int64_t byte_off = kp * 16;
      _mm_prefetch(reinterpret_cast<const char*>(next_w_base + 0 * w_stride + byte_off), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(next_w_base + 1 * w_stride + byte_off), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(next_w_base + 2 * w_stride + byte_off), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(next_w_base + 3 * w_stride + byte_off), _MM_HINT_T0);
    }
    const __m512 al0 = _mm512_loadu_ps(a_reordered + 0 * k + kp * 32);
    const __m512 ah0 = _mm512_loadu_ps(a_reordered + 0 * k + kp * 32 + 16);
    __m512 al1, ah1, al2, ah2, al3, ah3;
    if constexpr (RT >= 2) {
      al1 = _mm512_loadu_ps(a_reordered + 1 * k + kp * 32);
      ah1 = _mm512_loadu_ps(a_reordered + 1 * k + kp * 32 + 16);
    }
    if constexpr (RT >= 3) {
      al2 = _mm512_loadu_ps(a_reordered + 2 * k + kp * 32);
      ah2 = _mm512_loadu_ps(a_reordered + 2 * k + kp * 32 + 16);
    }
    if constexpr (RT >= 4) {
      al3 = _mm512_loadu_ps(a_reordered + 3 * k + kp * 32);
      ah3 = _mm512_loadu_ps(a_reordered + 3 * k + kp * 32 + 16);
    }
    auto step_col = [&](int c, __m512& a0, __m512& a1, __m512& a2, __m512& a3) QMOE_LAMBDA_INLINE {
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_rows[c] + kp * 16));
      __m512 wls, whs;
      if constexpr (IsMxfp4) {
        const __m128i lo = _mm_and_si128(x, mask128);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), mask128);
        const __m512 lut = lut_2d[s_rows[c][kp]];
        wls = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(lo), lut);
        whs = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(hi), lut);
      } else {
        const __m128i lo = _mm_or_si128(_mm_and_si128(x, mask128), hi_blk);
        const __m128i hi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(x, 4), mask128), hi_blk);
        const __m512 lut0 = lut_2d[s_rows[c][kp * 2]];
        const __m512 lut1 = lut_2d[s_rows[c][kp * 2 + 1]];
        wls = _mm512_permutex2var_ps(lut0, _mm512_cvtepu8_epi32(lo), lut1);
        whs = _mm512_permutex2var_ps(lut0, _mm512_cvtepu8_epi32(hi), lut1);
      }
      a0 = _mm512_fmadd_ps(wls, al0, _mm512_fmadd_ps(whs, ah0, a0));
      if constexpr (RT >= 2) a1 = _mm512_fmadd_ps(wls, al1, _mm512_fmadd_ps(whs, ah1, a1));
      if constexpr (RT >= 3) a2 = _mm512_fmadd_ps(wls, al2, _mm512_fmadd_ps(whs, ah2, a2));
      if constexpr (RT >= 4) a3 = _mm512_fmadd_ps(wls, al3, _mm512_fmadd_ps(whs, ah3, a3));
    };
    step_col(0, acc0_0, acc0_1, acc0_2, acc0_3);
    step_col(1, acc1_0, acc1_1, acc1_2, acc1_3);
    step_col(2, acc2_0, acc2_1, acc2_2, acc2_3);
    step_col(3, acc3_0, acc3_1, acc3_2, acc3_3);
  }
  auto store_col = [&](int c, __m512 a0, __m512 a1, __m512 a2, __m512 a3) QMOE_LAMBDA_INLINE {
    result[c][0] = _mm512_reduce_add_ps(a0);
    if constexpr (RT >= 2) result[c][1] = _mm512_reduce_add_ps(a1);
    if constexpr (RT >= 3) result[c][2] = _mm512_reduce_add_ps(a2);
    if constexpr (RT >= 4) result[c][3] = _mm512_reduce_add_ps(a3);
  };
  store_col(0, acc0_0, acc0_1, acc0_2, acc0_3);
  store_col(1, acc1_0, acc1_1, acc1_2, acc1_3);
  store_col(2, acc2_0, acc2_1, acc2_2, acc2_3);
  store_col(3, acc3_0, acc3_1, acc3_2, acc3_3);
}

inline int GetFp4Group8Mask() noexcept {
  const int override_mask = g_fp4_group8_mask_override.load(std::memory_order_relaxed);
  if (override_mask >= 0) {
    return override_mask & 0xF;
  }
  static const int mask = []() {
    const char* env = std::getenv("ORT_QMOE_FP4_GROUP8_MASK");
    if (env != nullptr && *env != '\0') {
      return static_cast<int>(std::strtol(env, nullptr, 0)) & 0xF;
    }
    // Default to Group4 (0x0) across RT=1..4: eliminates 32-ZMM register spills at RT=3,4
    // and keeps 4 weight + 4 scale streams within the L1D hardware prefetcher and 48 KiB L1D cache.
    return 0x0;
  }();
  return mask;
}

template <int RT, bool IsMxfp4>
void Fp4RowTileAvx512(const uint8_t* weights, const uint8_t* block_scales, float global_scale,
                      const __m512 (&lut_2d)[256], const uint16_t* bias, bool bias_is_bf16, int64_t n_total,
                      int64_t k, const float* a_reordered, float* out, int64_t out_stride, int64_t n_begin,
                      int64_t n_end, bool round_scale_bf16) {
  const int64_t w_stride = k / 2;
  const int64_t s_stride = IsMxfp4 ? (k / 32) : (k / 16);
  int64_t n0 = n_begin;
  if constexpr (RT <= 4) {
    if ((GetFp4Group8Mask() & (1 << (RT - 1))) != 0) {
      for (; n0 + 8 <= n_end; n0 += 8) {
        float result8[8][RT];
        Fp4Group8Avx512<RT, IsMxfp4>(weights + n0 * w_stride, block_scales + n0 * s_stride, lut_2d, k, a_reordered,
                                     result8);
        for (int c = 0; c < 8; ++c) {
          const float b = (bias != nullptr)
                              ? (bias_is_bf16 ? BFloat16ToFloat(bias[n0 + c]) : Avx512Fp4HalfBitsToFloat(bias[n0 + c]))
                              : 0.0f;
          for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result8[c][r] + b;
        }
      }
    }
  }
  for (; n0 + 4 <= n_end; n0 += 4) {
    float result[4][RT];
    const uint8_t* next_w = (n0 + 8 <= n_end) ? (weights + (n0 + 4) * w_stride) : nullptr;
    Fp4Group4Avx512<RT, IsMxfp4>(weights + n0 * w_stride, next_w, block_scales + n0 * s_stride, lut_2d, k, a_reordered,
                                 result);
    for (int c = 0; c < 4; ++c) {
      const float b = (bias != nullptr)
                          ? (bias_is_bf16 ? BFloat16ToFloat(bias[n0 + c]) : Avx512Fp4HalfBitsToFloat(bias[n0 + c]))
                          : 0.0f;
      for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result[c][r] + b;
    }
  }
  if (n0 < n_end) {
    if constexpr (IsMxfp4) {
      Mxfp4GemmRange(weights, block_scales, global_scale, bias, bias_is_bf16, n_total, k, a_reordered, RT,
                     out + (n0 - n_begin), out_stride, n0, n_end, /*force_scalar=*/true, round_scale_bf16);
    } else {
      Nvfp4GemmRange(weights, block_scales, global_scale, bias, bias_is_bf16, n_total, k, a_reordered, RT,
                     out + (n0 - n_begin), out_stride, n0, n_end, /*force_scalar=*/true, round_scale_bf16);
    }
  }
}

}  // namespace

void Avx512Nvfp4GemmRange(const uint8_t* weights, const uint8_t* block_scales, float global_scale,
                          const uint16_t* bias, bool bias_is_bf16, int64_t n_total, int64_t k,
                          const float* a_reordered, int64_t m, float* out, int64_t out_stride,
                          int64_t n_begin, int64_t n_end, bool round_scale_bf16) {
  const auto* lut_2d_ptr = GetScaleFp4Lut2D<false>(global_scale, round_scale_bf16);
  if (lut_2d_ptr == nullptr) {
    Nvfp4GemmRange(weights, block_scales, global_scale, bias, bias_is_bf16, n_total, k,
                   a_reordered, m, out, out_stride, n_begin, n_end, /*force_scalar=*/true, round_scale_bf16);
    return;
  }
  const auto& lut_2d = *lut_2d_ptr;
  int64_t r = 0;
  while (r < m) {
    const int64_t left = m - r;
    const float* ar = a_reordered + r * k;
    float* orow = out + r * out_stride;
    if (left >= 4) {
      Fp4RowTileAvx512<4, false>(weights, block_scales, global_scale, lut_2d, bias, bias_is_bf16, n_total, k, ar,
                                 orow, out_stride, n_begin, n_end, round_scale_bf16);
      r += 4;
    } else if (left == 3) {
      Fp4RowTileAvx512<3, false>(weights, block_scales, global_scale, lut_2d, bias, bias_is_bf16, n_total, k, ar,
                                 orow, out_stride, n_begin, n_end, round_scale_bf16);
      r += 3;
    } else if (left == 2) {
      Fp4RowTileAvx512<2, false>(weights, block_scales, global_scale, lut_2d, bias, bias_is_bf16, n_total, k, ar,
                                 orow, out_stride, n_begin, n_end, round_scale_bf16);
      r += 2;
    } else {
      Fp4RowTileAvx512<1, false>(weights, block_scales, global_scale, lut_2d, bias, bias_is_bf16, n_total, k, ar,
                                 orow, out_stride, n_begin, n_end, round_scale_bf16);
      r += 1;
    }
  }
}

void Avx512Mxfp4GemmRange(const uint8_t* weights, const uint8_t* block_scales, float global_scale,
                          const uint16_t* bias, bool bias_is_bf16, int64_t n_total, int64_t k,
                          const float* a_reordered, int64_t m, float* out, int64_t out_stride,
                          int64_t n_begin, int64_t n_end, bool round_scale_bf16) {
  const auto* lut_2d_ptr = GetScaleFp4Lut2D<true>(global_scale, round_scale_bf16);
  if (lut_2d_ptr == nullptr) {
    Mxfp4GemmRange(weights, block_scales, global_scale, bias, bias_is_bf16, n_total, k,
                   a_reordered, m, out, out_stride, n_begin, n_end, /*force_scalar=*/true, round_scale_bf16);
    return;
  }
  const auto& lut_2d = *lut_2d_ptr;
  int64_t r = 0;
  while (r < m) {
    const int64_t left = m - r;
    const float* ar = a_reordered + r * k;
    float* orow = out + r * out_stride;
    if (left >= 4) {
      Fp4RowTileAvx512<4, true>(weights, block_scales, global_scale, lut_2d, bias, bias_is_bf16, n_total, k, ar,
                                orow, out_stride, n_begin, n_end, round_scale_bf16);
      r += 4;
    } else if (left == 3) {
      Fp4RowTileAvx512<3, true>(weights, block_scales, global_scale, lut_2d, bias, bias_is_bf16, n_total, k, ar,
                                orow, out_stride, n_begin, n_end, round_scale_bf16);
      r += 3;
    } else if (left == 2) {
      Fp4RowTileAvx512<2, true>(weights, block_scales, global_scale, lut_2d, bias, bias_is_bf16, n_total, k, ar,
                                orow, out_stride, n_begin, n_end, round_scale_bf16);
      r += 2;
    } else {
      Fp4RowTileAvx512<1, true>(weights, block_scales, global_scale, lut_2d, bias, bias_is_bf16, n_total, k, ar,
                                orow, out_stride, n_begin, n_end, round_scale_bf16);
      r += 1;
    }
  }
}

void SetFp4Group8MaskOverride(int mask) noexcept {
  g_fp4_group8_mask_override.store(mask, std::memory_order_relaxed);
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
