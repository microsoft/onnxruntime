// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Vectorized e2m1 (NVFP4/MXFP4) -> bf16/fp16 decode for the weight-only GEMV fast path.
//
// Fp4I2FConverter (details.h) decodes one 4-bit code at a time: two `prmt.b32` to look up the
// magnitude's high/low bytes, plus a shift/mask/or to splice the sign in, plus the nibble
// extraction -- about eight ALU-pipe instructions per element. Because prmt's selector is four
// independent nibbles, a single prmt looks up FOUR codes at once, and the packed weight word
// already *is* that selector once the sign bits are masked off. The signs are gathered with
// prmt's msb-replicate mode. Result: ~19 ALU ops per eight codes (~2.4/element) instead of ~8.
//
// The magnitude byte tables below are the same constants Fp4I2FConverter uses, so every decoded
// value is bit-identical to the per-element path. This is verified exhaustively over all 2^32
// packed words for {bf16, fp16} x {linear, pair-interleaved}.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {
namespace fp4_fast {

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Decodes the eight e2m1 codes packed in `w` into four registers of two 16-bit floats each:
// r[t] holds { nibble slot 2t, nibble slot 2t+1 }, i.e. results are in *slot* order. The caller
// maps logical element -> slot with slot_of() below.
template <typename TypeA>
__device__ __forceinline__ void decode_word(uint32_t w, uint32_t* r) {
#ifdef ENABLE_BF16
  constexpr bool kBf16 = std::is_same_v<TypeA, __nv_bfloat16>;
#else
  constexpr bool kBf16 = false;
#endif
  // Magnitude bytes indexed by the 3-bit code {0, .5, 1, 1.5, 2, 3, 4, 6}:
  //   bf16 hi {00,3F,3F,3F,40,40,40,40}  lo {00,00,80,C0,00,40,80,C0}
  //   fp16 hi {00,38,3C,3E,40,42,44,46}  lo all zero
  constexpr uint32_t kHiA = kBf16 ? 0x3F3F3F00u : 0x3E3C3800u;
  constexpr uint32_t kHiB = kBf16 ? 0x40404040u : 0x46444240u;
  constexpr uint32_t kLoA = 0xC0800000u;
  constexpr uint32_t kLoB = 0xC0804000u;

  // Clearing bit 3 of each nibble both isolates the magnitude and keeps prmt out of msb mode.
  const uint32_t m = w & 0x77777777u;
  const uint32_t m2 = m >> 16;

  uint32_t hi_a = prmt(kHiA, kHiB, m);   // [hi(e0), hi(e1), hi(e2), hi(e3)]
  uint32_t hi_b = prmt(kHiA, kHiB, m2);  // [hi(e4), hi(e5), hi(e6), hi(e7)]

  // Sign gather. A prmt selector nibble of 8|j replicates the msb of source byte j across the
  // whole output byte. The byte msbs of w are the odd elements' signs; the byte msbs of w<<4 are
  // the even elements' signs.
  const uint32_t ev = prmt(w << 4, w << 4, 0xBA98u);  // [sgn(e0), sgn(e2), sgn(e4), sgn(e6)]
  const uint32_t od = prmt(w, w, 0xBA98u);            // [sgn(e1), sgn(e3), sgn(e5), sgn(e7)]
  hi_a |= prmt(ev, od, 0x5140u) & 0x80808080u;        // one LOP3 each
  hi_b |= prmt(ev, od, 0x7362u) & 0x80808080u;

  if constexpr (kBf16) {
    const uint32_t lo_a = prmt(kLoA, kLoB, m);
    const uint32_t lo_b = prmt(kLoA, kLoB, m2);
    r[0] = prmt(lo_a, hi_a, 0x5140u);
    r[1] = prmt(lo_a, hi_a, 0x7362u);
    r[2] = prmt(lo_b, hi_b, 0x5140u);
    r[3] = prmt(lo_b, hi_b, 0x7362u);
  } else {
    r[0] = prmt(0u, hi_a, 0x5140u);
    r[1] = prmt(0u, hi_a, 0x7362u);
    r[2] = prmt(0u, hi_b, 0x5140u);
    r[3] = prmt(0u, hi_b, 0x7362u);
  }
}

// Nibble slot holding logical element i of an 8-element group.
//   linear                 : slot(i) = i
//   pair-interleaved (SM80): [e0,e2,e4,e6,e1,e3,e5,e7], i.e. Fp4I2FConverter's kSlot table
__host__ __device__ constexpr int slot_of(int i, bool pair_interleaved) {
  return pair_interleaved ? ((i & 1) * 4 + (i >> 1)) : i;
}
// Logical element p of a 32-element chunk lives in register reg_of(p), half half_of(p).
__host__ __device__ constexpr int reg_of(int p, bool pair) { return (p / 8) * 4 + slot_of(p % 8, pair) / 2; }
__host__ __device__ constexpr int half_of(int p, bool pair) { return slot_of(p % 8, pair) & 1; }

template <typename TypeA>
__device__ __forceinline__ float lo_to_float(uint32_t x) {
#ifdef ENABLE_BF16
  if constexpr (std::is_same_v<TypeA, __nv_bfloat16>) {
    return __uint_as_float(x << 16);  // exact: bf16 widening is a 16-bit left shift
  } else
#endif
  {
    return __half2float(__ushort_as_half(static_cast<uint16_t>(x)));
  }
}

template <typename TypeA>
__device__ __forceinline__ float hi_to_float(uint32_t x) {
#ifdef ENABLE_BF16
  if constexpr (std::is_same_v<TypeA, __nv_bfloat16>) {
    return __uint_as_float(x & 0xFFFF0000u);
  } else
#endif
  {
    return __half2float(__ushort_as_half(static_cast<uint16_t>(x >> 16)));
  }
}

}  // namespace fp4_fast
}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
