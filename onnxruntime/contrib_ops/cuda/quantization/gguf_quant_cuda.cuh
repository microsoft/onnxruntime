// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace gguf_cuda {

using namespace onnxruntime::contrib::gguf_quant;

__device__ __forceinline__ uint16_t ReadU16Dev(const uint8_t* p) {
  if ((reinterpret_cast<uintptr_t>(p) & 1u) == 0) {
    return __ldg(reinterpret_cast<const uint16_t*>(p));
  }
  return static_cast<uint16_t>(__ldg(p)) | (static_cast<uint16_t>(__ldg(p + 1)) << 8);
}

__device__ __forceinline__ uint32_t ReadU32Dev(const uint8_t* p) {
  if ((reinterpret_cast<uintptr_t>(p) & 3u) == 0) {
    return __ldg(reinterpret_cast<const uint32_t*>(p));
  }
  return static_cast<uint32_t>(ReadU16Dev(p)) | (static_cast<uint32_t>(ReadU16Dev(p + 2)) << 16);
}

// Used for GGUF block formats whose stride or 32-bit field offset is 2 (mod 4), where lane addresses
// alternate between 2-byte and 4-byte alignment across a warp.
__device__ __forceinline__ uint32_t ReadU32Unaligned2BDev(const uint8_t* p) {
  if ((reinterpret_cast<uintptr_t>(p) & 1u) == 0) {
    const auto* p16 = reinterpret_cast<const uint16_t*>(p);
    return static_cast<uint32_t>(__ldg(p16)) | (static_cast<uint32_t>(__ldg(p16 + 1)) << 16);
  }
  return static_cast<uint32_t>(__ldg(p)) |
         (static_cast<uint32_t>(__ldg(p + 1)) << 8) |
         (static_cast<uint32_t>(__ldg(p + 2)) << 16) |
         (static_cast<uint32_t>(__ldg(p + 3)) << 24);
}

// Tensor rows may be element-aligned without being aligned for a 128-bit load.
__device__ __forceinline__ uint4 LoadU4Dev(const void* p) {
  if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) {
    return __ldg(reinterpret_cast<const uint4*>(p));
  }
  const auto* bytes = static_cast<const uint8_t*>(p);
  return make_uint4(ReadU32Dev(bytes), ReadU32Dev(bytes + 4), ReadU32Dev(bytes + 8), ReadU32Dev(bytes + 12));
}

__device__ __forceinline__ float Fp16ToFloatDev(uint16_t bits) {
  return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ float Bf16ToFloatDev(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ void GetScaleMinK4Dev(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = __ldg(q + j) & 63;
    m = __ldg(q + j + 4) & 63;
  } else {
    d = (__ldg(q + j + 4) & 0xF) | ((__ldg(q + j - 4) >> 6) << 4);
    m = (__ldg(q + j + 4) >> 4) | ((__ldg(q + j - 0) >> 6) << 4);
  }
}

__device__ __forceinline__ void MmaBf16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ uint4 PackFloat8ToBf16x2(const float (&v)[8]) {
  uint4 res;
  *reinterpret_cast<__nv_bfloat162*>(&res.x) = __floats2bfloat162_rn(v[0], v[1]);
  *reinterpret_cast<__nv_bfloat162*>(&res.y) = __floats2bfloat162_rn(v[2], v[3]);
  *reinterpret_cast<__nv_bfloat162*>(&res.z) = __floats2bfloat162_rn(v[4], v[5]);
  *reinterpret_cast<__nv_bfloat162*>(&res.w) = __floats2bfloat162_rn(v[6], v[7]);
  return res;
}

// Register-only computation of ksigns_iq2xs[x7] (7-bit index -> 8-bit even-parity sign mask).
__device__ __forceinline__ uint8_t Iq2xsSignsDev(uint32_t x7) {
  const uint32_t s = x7 & 127u;
  return static_cast<uint8_t>(s | ((__popc(s) & 1u) << 7));
}

// Register-only lookup of kvalues_iq4nl[n & 0xF] via hardware prmt.b32 (__byte_perm).
__device__ __forceinline__ float Iq4NlValueDev(uint32_t n4) {
  const uint32_t n = n4 & 0xFu;
  const uint32_t lo = __byte_perm(0xbfad9881u, 0xf6eaddcfu, n);
  const uint32_t hi = __byte_perm(0x26190d01u, 0x71594535u, n);
  return static_cast<float>(static_cast<int8_t>((n & 8u) ? hi : lo));
}

// Decodes 8 contiguous elements [sb32_idx * 32 + l * 8 .. + 7] (where l in 0..3) of a GGUF weight row.
template <int kCompileType>
__device__ __forceinline__ void DecodeGgufSubBlock8Impl(
    const uint8_t* row_ptr, int sb32_idx, int l, float (&out)[8]) {
  if constexpr (kCompileType == kGgmlTypeF32) {
    const uint8_t* p = row_ptr + (static_cast<int64_t>(sb32_idx) * 32 + l * 8) * 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = __uint_as_float(ReadU32Dev(p + j * 4));
  } else if constexpr (kCompileType == kGgmlTypeF16) {
    const uint8_t* p = row_ptr + (static_cast<int64_t>(sb32_idx) * 32 + l * 8) * 2;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = Fp16ToFloatDev(ReadU16Dev(p + j * 2));
  } else if constexpr (kCompileType == kGgmlTypeBF16) {
    const uint8_t* p = row_ptr + (static_cast<int64_t>(sb32_idx) * 32 + l * 8) * 2;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = Bf16ToFloatDev(ReadU16Dev(p + j * 2));
  } else if constexpr (kCompileType == kGgmlTypeQ8_0) {
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb32_idx) * 34;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint32_t q0 = ReadU32Unaligned2BDev(blk + 2 + l * 8);
    const uint32_t q1 = ReadU32Unaligned2BDev(blk + 6 + l * 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[j] = d * static_cast<float>(static_cast<int8_t>((q0 >> (8 * j)) & 0xFFu));
      out[j + 4] = d * static_cast<float>(static_cast<int8_t>((q1 >> (8 * j)) & 0xFFu));
    }
  } else if constexpr (kCompileType == kGgmlTypeNVFP4) {
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb32_idx >> 1) * 36;
    const int group = 2 * (sb32_idx & 1) + (l >> 1);
    const uint8_t scale = __ldg(blk + group);
    // The packed-format contract requires bit 7 clear; 0x7f is the zero sentinel.
    const float d = scale == 127 ? 0.0f : __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(scale), __NV_E4M3)));
    const uint8_t* codes = blk + 4 + group * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int code = (__ldg(codes + j) >> (4 * (l & 1))) & 15;
      out[j] = d * (0.5f * static_cast<float>(kvalues_fp4[code]));
    }
  } else if constexpr (kCompileType == kGgmlTypeQ4_0) {
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb32_idx) * 18;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const int shift = (l >> 1) * 4;
    const int off = (l & 1) * 8;
    const uint32_t q0 = ReadU32Unaligned2BDev(blk + 2 + off);
    const uint32_t q1 = ReadU32Unaligned2BDev(blk + 6 + off);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[j] = static_cast<float>(static_cast<int>((q0 >> (8 * j + shift)) & 0xFu) - 8) * d;
      out[j + 4] = static_cast<float>(static_cast<int>((q1 >> (8 * j + shift)) & 0xFu) - 8) * d;
    }
  } else if constexpr (kCompileType == kGgmlTypeQ5_0) {
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb32_idx) * 22;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint32_t qh = ReadU32Unaligned2BDev(blk + 2) >> (l * 8);
    const int shift = (l >> 1) * 4;
    const int off = (l & 1) * 8;
    const uint32_t q0 = ReadU32Unaligned2BDev(blk + 6 + off);
    const uint32_t q1 = ReadU32Unaligned2BDev(blk + 10 + off);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int q5_lo = static_cast<int>(((q0 >> (8 * j + shift)) & 0xFu) | (((qh >> j) & 1u) << 4));
      const int q5_hi = static_cast<int>(((q1 >> (8 * j + shift)) & 0xFu) | (((qh >> (j + 4)) & 1u) << 4));
      out[j] = static_cast<float>(q5_lo - 16) * d;
      out[j + 4] = static_cast<float>(q5_hi - 16) * d;
    }
  } else if constexpr (kCompileType == kGgmlTypeQ2_0) {
    const int blk64 = sb32_idx >> 1;
    const int sub32 = sb32_idx & 1;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(blk64) * 18;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint16_t q16 = ReadU16Dev(blk + 2 + sub32 * 8 + l * 2);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int q = static_cast<int>((q16 >> (2 * j)) & 0x3u);
      out[j] = static_cast<float>(q - 1) * d;
    }
  } else if constexpr (kCompileType == kGgmlTypeIQ4_NL) {
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb32_idx) * 18;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const int shift = (l >> 1) * 4;
    const int off = (l & 1) * 8;
    const uint32_t q0 = ReadU32Unaligned2BDev(blk + 2 + off);
    const uint32_t q1 = ReadU32Unaligned2BDev(blk + 6 + off);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[j] = d * Iq4NlValueDev(q0 >> (8 * j + shift));
      out[j + 4] = d * Iq4NlValueDev(q1 >> (8 * j + shift));
    }
  } else if constexpr (kCompileType == kGgmlTypeIQ4_XS) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 136;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint16_t scales_h = ReadU16Dev(blk + 2);
    const uint8_t sc_l = __ldg(blk + 4 + (ib >> 1));
    const int ls = ((sc_l >> (4 * (ib & 1))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
    const float dl = d * static_cast<float>(ls - 32);
    const int shift = (l >> 1) * 4;
    const int off = (l & 1) * 8;
    const uint32_t q0 = ReadU32Dev(blk + 8 + ib * 16 + off);
    const uint32_t q1 = ReadU32Dev(blk + 12 + ib * 16 + off);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[j] = dl * Iq4NlValueDev(q0 >> (8 * j + shift));
      out[j + 4] = dl * Iq4NlValueDev(q1 >> (8 * j + shift));
    }
  } else if constexpr (kCompileType == kGgmlTypeQ5_K) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const int half16 = l >> 1;
    const int off = (l & 1) * 8;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 176;
    const float dall = Fp16ToFloatDev(ReadU16Dev(blk));
    const float dmin = Fp16ToFloatDev(ReadU16Dev(blk + 2));
    uint8_t sc, m;
    GetScaleMinK4Dev(ib, blk + 4, sc, m);
    const float d_sub = dall * static_cast<float>(sc);
    const float m_sub = dmin * static_cast<float>(m);
    const uint32_t qh0 = ReadU32Dev(blk + 16 + half16 * 16 + off);
    const uint32_t qh1 = ReadU32Dev(blk + 20 + half16 * 16 + off);
    const uint32_t ql0 = ReadU32Dev(blk + 48 + (ib >> 1) * 32 + half16 * 16 + off);
    const uint32_t ql1 = ReadU32Dev(blk + 52 + (ib >> 1) * 32 + half16 * 16 + off);
    const int shift = (ib & 1) * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int q0 = static_cast<int>(((ql0 >> (8 * j + shift)) & 0xFu) | (((qh0 >> (8 * j + ib)) & 1u) << 4));
      const int q1 = static_cast<int>(((ql1 >> (8 * j + shift)) & 0xFu) | (((qh1 >> (8 * j + ib)) & 1u) << 4));
      out[j] = d_sub * static_cast<float>(q0) - m_sub;
      out[j + 4] = d_sub * static_cast<float>(q1) - m_sub;
    }
  } else if constexpr (kCompileType == kGgmlTypeQ6_K) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const int half16 = l >> 1;
    const int off = (l & 1) * 8;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 210;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk + 208));
    const int ip = ib >> 2;
    const int q_grp = ib & 3;
    const int8_t sc = static_cast<int8_t>(__ldg(blk + 192 + 8 * ip + 2 * q_grp + half16));
    const float dl = d * static_cast<float>(sc);
    const uint32_t ql0 = ReadU32Unaligned2BDev(blk + 64 * ip + 32 * (q_grp & 1) + half16 * 16 + off);
    const uint32_t ql1 = ReadU32Unaligned2BDev(blk + 64 * ip + 32 * (q_grp & 1) + half16 * 16 + off + 4);
    const uint32_t qh0 = ReadU32Unaligned2BDev(blk + 128 + 32 * ip + half16 * 16 + off);
    const uint32_t qh1 = ReadU32Unaligned2BDev(blk + 128 + 32 * ip + half16 * 16 + off + 4);
    const int ql_shift = (q_grp >> 1) * 4;
    const int qh_shift = q_grp * 2;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int q0 =
          static_cast<int>(((ql0 >> (8 * j + ql_shift)) & 0xFu) | (((qh0 >> (8 * j + qh_shift)) & 3u) << 4)) - 32;
      const int q1 =
          static_cast<int>(((ql1 >> (8 * j + ql_shift)) & 0xFu) | (((qh1 >> (8 * j + qh_shift)) & 3u) << 4)) - 32;
      out[j] = dl * static_cast<float>(q0);
      out[j + 4] = dl * static_cast<float>(q1);
    }
  } else if constexpr (kCompileType == kGgmlTypeIQ2_XXS) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 66;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint8_t* aux8 = blk + 2 + 8 * ib;
    const uint32_t aux32 = ReadU32Unaligned2BDev(aux8 + 4);
    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.25f;
    const uint64_t g64 = __ldg(iq2xxs_grid + __ldg(aux8 + l));
    const uint8_t signs = Iq2xsSignsDev(aux32 >> (7 * l));
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      out[j] = db * static_cast<float>(static_cast<uint8_t>(g64 >> (8 * j))) *
               ((signs & (1u << j)) ? -1.0f : 1.0f);
    }
  } else if constexpr (kCompileType == kGgmlTypeIQ2_XS) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 74;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint8_t sc_byte = __ldg(blk + 66 + ib);
    const float db = d * (0.5f + static_cast<float>((sc_byte >> (4 * (l >> 1))) & 0xF)) * 0.25f;
    const uint16_t q2 = ReadU16Dev(blk + 2 + (4 * ib + l) * 2);
    const uint64_t g64 = __ldg(iq2xs_grid + (q2 & 511));
    const uint8_t signs = Iq2xsSignsDev(q2 >> 9);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      out[j] = db * static_cast<float>(static_cast<uint8_t>(g64 >> (8 * j))) *
               ((signs & (1u << j)) ? -1.0f : 1.0f);
    }
  } else if constexpr (kCompileType == kGgmlTypeIQ2_S) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 82;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint8_t sc_byte = __ldg(blk + 74 + ib);
    const float db = d * (0.5f + static_cast<float>((sc_byte >> (4 * (l >> 1))) & 0xF)) * 0.25f;
    const uint8_t qh_ib = __ldg(blk + 66 + ib);
    const int g_idx =
        static_cast<int>(__ldg(blk + 2 + 4 * ib + l)) | ((static_cast<int>(qh_ib) << (8 - 2 * l)) & 0x300);
    const uint64_t g64 = __ldg(iq2s_grid + g_idx);
    const uint8_t signs = __ldg(blk + 34 + 4 * ib + l);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      out[j] = db * static_cast<float>(static_cast<uint8_t>(g64 >> (8 * j))) *
               ((signs & (1u << j)) ? -1.0f : 1.0f);
    }
  } else if constexpr (kCompileType == kGgmlTypeIQ3_XXS) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 98;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint32_t aux32 = ReadU32Unaligned2BDev(blk + 66 + 4 * ib);
    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
    const uint8_t signs = Iq2xsSignsDev(aux32 >> (7 * l));
    const uint16_t q2 = ReadU16Dev(blk + 2 + 8 * ib + 2 * l);
    const uint32_t g1 = __ldg(iq3xxs_grid + (q2 & 0xFFu));
    const uint32_t g2 = __ldg(iq3xxs_grid + (q2 >> 8));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[j] = db * static_cast<float>(static_cast<uint8_t>(g1 >> (8 * j))) *
               ((signs & (1u << j)) ? -1.0f : 1.0f);
      out[j + 4] = db * static_cast<float>(static_cast<uint8_t>(g2 >> (8 * j))) *
                   ((signs & (1u << (j + 4))) ? -1.0f : 1.0f);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = 0.0f;
  }
}

__device__ __forceinline__ void DecodeGgufSubBlock8(
    const uint8_t* row_ptr, int ggml_type, int sb32_idx, int l, float (&out)[8]) {
  switch (ggml_type) {
    case kGgmlTypeF32:
      DecodeGgufSubBlock8Impl<kGgmlTypeF32>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeF16:
      DecodeGgufSubBlock8Impl<kGgmlTypeF16>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeBF16:
      DecodeGgufSubBlock8Impl<kGgmlTypeBF16>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeQ8_0:
      DecodeGgufSubBlock8Impl<kGgmlTypeQ8_0>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeNVFP4:
      DecodeGgufSubBlock8Impl<kGgmlTypeNVFP4>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeQ4_0:
      DecodeGgufSubBlock8Impl<kGgmlTypeQ4_0>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeQ5_0:
      DecodeGgufSubBlock8Impl<kGgmlTypeQ5_0>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeQ2_0:
      DecodeGgufSubBlock8Impl<kGgmlTypeQ2_0>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeIQ4_NL:
      DecodeGgufSubBlock8Impl<kGgmlTypeIQ4_NL>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeIQ4_XS:
      DecodeGgufSubBlock8Impl<kGgmlTypeIQ4_XS>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeQ5_K:
      DecodeGgufSubBlock8Impl<kGgmlTypeQ5_K>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeQ6_K:
      DecodeGgufSubBlock8Impl<kGgmlTypeQ6_K>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeIQ2_XXS:
      DecodeGgufSubBlock8Impl<kGgmlTypeIQ2_XXS>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeIQ2_XS:
      DecodeGgufSubBlock8Impl<kGgmlTypeIQ2_XS>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeIQ2_S:
      DecodeGgufSubBlock8Impl<kGgmlTypeIQ2_S>(row_ptr, sb32_idx, l, out);
      return;
    case kGgmlTypeIQ3_XXS:
      DecodeGgufSubBlock8Impl<kGgmlTypeIQ3_XXS>(row_ptr, sb32_idx, l, out);
      return;
    default:
#pragma unroll
      for (int j = 0; j < 8; ++j) out[j] = 0.0f;
      return;
  }
}

template <int kCompileType>
__device__ __forceinline__ void DecodeGgufSubBlock8Tpl(
    const uint8_t* row_ptr, int runtime_type, int sb32_idx, int l, float (&out)[8]) {
  if constexpr (kCompileType >= 0) {
    DecodeGgufSubBlock8Impl<kCompileType>(row_ptr, sb32_idx, l, out);
  } else {
    DecodeGgufSubBlock8(row_ptr, runtime_type, sb32_idx, l, out);
  }
}

template <int kCompileType>
__device__ __forceinline__ void DecodeGgufSubBlock8SmemLutTpl(
    const uint8_t* row_ptr,
    int runtime_type,
    int sb32_idx,
    int l,
    const void* s_lut,
    float (&out)[8]) {
  if constexpr (kCompileType == kGgmlTypeIQ2_XXS) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 66;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint8_t* aux8 = blk + 2 + 8 * ib;
    const uint32_t aux32 = ReadU32Unaligned2BDev(aux8 + 4);
    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.25f;
    const uint64_t g64 = static_cast<const uint64_t*>(s_lut)[__ldg(aux8 + l)];
    const uint8_t signs = Iq2xsSignsDev(aux32 >> (7 * l));
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      out[j] = ((signs & (1u << j)) ? -db : db) * static_cast<float>(static_cast<uint8_t>(g64 >> (8 * j)));
    }
  } else if constexpr (kCompileType == kGgmlTypeIQ2_XS) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 74;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint8_t sc_byte = __ldg(blk + 66 + ib);
    const float db = d * (0.5f + static_cast<float>((sc_byte >> (4 * (l >> 1))) & 0xF)) * 0.25f;
    const uint16_t q2 = ReadU16Dev(blk + 2 + (4 * ib + l) * 2);
    const uint64_t g64 = static_cast<const uint64_t*>(s_lut)[q2 & 511];
    const uint8_t signs = Iq2xsSignsDev(q2 >> 9);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      out[j] = ((signs & (1u << j)) ? -db : db) * static_cast<float>(static_cast<uint8_t>(g64 >> (8 * j)));
    }
  } else if constexpr (kCompileType == kGgmlTypeIQ2_S) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 82;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint8_t sc_byte = __ldg(blk + 74 + ib);
    const float db = d * (0.5f + static_cast<float>((sc_byte >> (4 * (l >> 1))) & 0xF)) * 0.25f;
    const uint8_t qh_ib = __ldg(blk + 66 + ib);
    const int g_idx =
        static_cast<int>(__ldg(blk + 2 + 4 * ib + l)) | ((static_cast<int>(qh_ib) << (8 - 2 * l)) & 0x300);
    const uint64_t g64 = static_cast<const uint64_t*>(s_lut)[g_idx];
    const uint8_t signs = __ldg(blk + 34 + 4 * ib + l);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      out[j] = ((signs & (1u << j)) ? -db : db) * static_cast<float>(static_cast<uint8_t>(g64 >> (8 * j)));
    }
  } else if constexpr (kCompileType == kGgmlTypeIQ3_XXS) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 98;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint32_t aux32 = ReadU32Unaligned2BDev(blk + 66 + 4 * ib);
    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
    const uint8_t signs = Iq2xsSignsDev(aux32 >> (7 * l));
    const uint16_t q2 = ReadU16Dev(blk + 2 + 8 * ib + 2 * l);
    const uint32_t g1 = static_cast<const uint32_t*>(s_lut)[q2 & 0xFFu];
    const uint32_t g2 = static_cast<const uint32_t*>(s_lut)[q2 >> 8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[j] = ((signs & (1u << j)) ? -db : db) * static_cast<float>(static_cast<uint8_t>(g1 >> (8 * j)));
      out[j + 4] = ((signs & (1u << (j + 4))) ? -db : db) *
                   static_cast<float>(static_cast<uint8_t>(g2 >> (8 * j)));
    }
  } else {
    DecodeGgufSubBlock8Tpl<kCompileType>(row_ptr, runtime_type, sb32_idx, l, out);
  }
}

// Decodes 16 contiguous elements [sb32_idx * 32 + half16 * 16 .. + 15] of a GGUF weight row
// with a single scale computation per 16 elements and vectorized 32-bit quant loads.
template <int kCompileType>
__device__ __forceinline__ void DecodeGgufSubBlock16Impl(
    const uint8_t* row_ptr, int sb32_idx, int half16, float (&out)[16]) {
  if constexpr (kCompileType == kGgmlTypeF32) {
    const uint8_t* p = row_ptr + (static_cast<int64_t>(sb32_idx) * 32 + half16 * 16) * 4;
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = __uint_as_float(ReadU32Dev(p + j * 4));
  } else if constexpr (kCompileType == kGgmlTypeF16) {
    const uint8_t* p = row_ptr + (static_cast<int64_t>(sb32_idx) * 32 + half16 * 16) * 2;
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = Fp16ToFloatDev(ReadU16Dev(p + j * 2));
  } else if constexpr (kCompileType == kGgmlTypeBF16) {
    const uint8_t* p = row_ptr + (static_cast<int64_t>(sb32_idx) * 32 + half16 * 16) * 2;
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = Bf16ToFloatDev(ReadU16Dev(p + j * 2));
  } else if constexpr (kCompileType == kGgmlTypeQ8_0) {
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb32_idx) * 34;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint8_t* qs = blk + 2 + half16 * 16;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t qw = ReadU32Unaligned2BDev(qs + w * 4);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        out[w * 4 + b] = d * static_cast<float>(static_cast<int8_t>((qw >> (8 * b)) & 0xFFu));
      }
    }
  } else if constexpr (kCompileType == kGgmlTypeQ4_0) {
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb32_idx) * 18;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const int shift = half16 * 4;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t qw = ReadU32Unaligned2BDev(blk + 2 + w * 4);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        out[w * 4 + b] = static_cast<float>(static_cast<int>((qw >> (8 * b + shift)) & 0xFu) - 8) * d;
      }
    }
  } else if constexpr (kCompileType == kGgmlTypeQ5_0) {
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb32_idx) * 22;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint32_t qh = ReadU32Unaligned2BDev(blk + 2) >> (half16 * 16);
    const int shift = half16 * 4;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t qw = ReadU32Unaligned2BDev(blk + 6 + w * 4);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int idx = w * 4 + b;
        const int q5 = static_cast<int>(((qw >> (8 * b + shift)) & 0xFu) | (((qh >> idx) & 1u) << 4));
        out[idx] = static_cast<float>(q5 - 16) * d;
      }
    }
  } else if constexpr (kCompileType == kGgmlTypeQ2_0) {
    const int blk64 = sb32_idx >> 1;
    const int sub32 = sb32_idx & 1;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(blk64) * 18;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint32_t q32 = ReadU32Unaligned2BDev(blk + 2 + sub32 * 8 + half16 * 4);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      out[j] = static_cast<float>(static_cast<int>((q32 >> (2 * j)) & 0x3u) - 1) * d;
    }
  } else if constexpr (kCompileType == kGgmlTypeIQ4_NL) {
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb32_idx) * 18;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const int shift = half16 * 4;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t qw = ReadU32Unaligned2BDev(blk + 2 + w * 4);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        out[w * 4 + b] = d * Iq4NlValueDev(qw >> (8 * b + shift));
      }
    }
  } else if constexpr (kCompileType == kGgmlTypeIQ4_XS) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 136;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk));
    const uint16_t scales_h = ReadU16Dev(blk + 2);
    const uint8_t sc_l = __ldg(blk + 4 + (ib >> 1));
    const int ls = ((sc_l >> (4 * (ib & 1))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
    const float dl = d * static_cast<float>(ls - 32);
    const int shift = half16 * 4;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t qw = ReadU32Dev(blk + 8 + ib * 16 + w * 4);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        out[w * 4 + b] = dl * Iq4NlValueDev(qw >> (8 * b + shift));
      }
    }
  } else if constexpr (kCompileType == kGgmlTypeQ5_K) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 176;
    const float dall = Fp16ToFloatDev(ReadU16Dev(blk));
    const float dmin = Fp16ToFloatDev(ReadU16Dev(blk + 2));
    uint8_t sc, m;
    GetScaleMinK4Dev(ib, blk + 4, sc, m);
    const float d_sub = dall * static_cast<float>(sc);
    const float m_sub = dmin * static_cast<float>(m);
    const uint8_t* qh = blk + 16 + half16 * 16;
    const uint8_t* ql = blk + 48 + (ib >> 1) * 32 + half16 * 16;
    const int shift = (ib & 1) * 4;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t qlw = ReadU32Dev(ql + w * 4);
      const uint32_t qhw = ReadU32Dev(qh + w * 4);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int q = static_cast<int>(((qlw >> (8 * b + shift)) & 0xFu) | (((qhw >> (8 * b + ib)) & 1u) << 4));
        out[w * 4 + b] = d_sub * static_cast<float>(q) - m_sub;
      }
    }
  } else if constexpr (kCompileType == kGgmlTypeQ6_K) {
    const int sb256 = sb32_idx >> 3;
    const int ib = sb32_idx & 7;
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb256) * 210;
    const float d = Fp16ToFloatDev(ReadU16Dev(blk + 208));
    const int ip = ib >> 2;
    const int q_grp = ib & 3;
    const int8_t sc = static_cast<int8_t>(__ldg(blk + 192 + 8 * ip + 2 * q_grp + half16));
    const float dl = d * static_cast<float>(sc);
    const uint8_t* ql = blk + 64 * ip + 32 * (q_grp & 1) + half16 * 16;
    const uint8_t* qh = blk + 128 + 32 * ip + half16 * 16;
    const int ql_shift = (q_grp >> 1) * 4;
    const int qh_shift = q_grp * 2;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t qlw = ReadU32Unaligned2BDev(ql + w * 4);
      const uint32_t qhw = ReadU32Unaligned2BDev(qh + w * 4);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int q =
            static_cast<int>(((qlw >> (8 * b + ql_shift)) & 0xFu) | (((qhw >> (8 * b + qh_shift)) & 3u) << 4)) - 32;
        out[w * 4 + b] = dl * static_cast<float>(q);
      }
    }
  } else {
    float lo[8];
    float hi[8];
    DecodeGgufSubBlock8Impl<kCompileType>(row_ptr, sb32_idx, 2 * half16 + 0, lo);
    DecodeGgufSubBlock8Impl<kCompileType>(row_ptr, sb32_idx, 2 * half16 + 1, hi);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      out[j] = lo[j];
      out[j + 8] = hi[j];
    }
  }
}

// Decodes 16 contiguous elements [sb32_idx * 32 + half16 * 16 .. + 15] of a GGUF weight row.
__device__ __forceinline__ void DecodeGgufSubBlock32(
    const uint8_t* row_ptr, int ggml_type, int sb32_idx, int half16, float (&out)[16]) {
  switch (ggml_type) {
    case kGgmlTypeF32:
      DecodeGgufSubBlock16Impl<kGgmlTypeF32>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeF16:
      DecodeGgufSubBlock16Impl<kGgmlTypeF16>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeBF16:
      DecodeGgufSubBlock16Impl<kGgmlTypeBF16>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeQ8_0:
      DecodeGgufSubBlock16Impl<kGgmlTypeQ8_0>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeQ4_0:
      DecodeGgufSubBlock16Impl<kGgmlTypeQ4_0>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeQ5_0:
      DecodeGgufSubBlock16Impl<kGgmlTypeQ5_0>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeQ2_0:
      DecodeGgufSubBlock16Impl<kGgmlTypeQ2_0>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeIQ4_NL:
      DecodeGgufSubBlock16Impl<kGgmlTypeIQ4_NL>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeIQ4_XS:
      DecodeGgufSubBlock16Impl<kGgmlTypeIQ4_XS>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeQ5_K:
      DecodeGgufSubBlock16Impl<kGgmlTypeQ5_K>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeQ6_K:
      DecodeGgufSubBlock16Impl<kGgmlTypeQ6_K>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeIQ2_XXS:
      DecodeGgufSubBlock16Impl<kGgmlTypeIQ2_XXS>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeIQ2_XS:
      DecodeGgufSubBlock16Impl<kGgmlTypeIQ2_XS>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeIQ2_S:
      DecodeGgufSubBlock16Impl<kGgmlTypeIQ2_S>(row_ptr, sb32_idx, half16, out);
      return;
    case kGgmlTypeIQ3_XXS:
      DecodeGgufSubBlock16Impl<kGgmlTypeIQ3_XXS>(row_ptr, sb32_idx, half16, out);
      return;
    default:
#pragma unroll
      for (int j = 0; j < 16; ++j) out[j] = 0.0f;
      return;
  }
}

template <typename CudaT>
__device__ __forceinline__ CudaT FloatToCudaType(float v);

template <>
__device__ __forceinline__ float FloatToCudaType<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ half FloatToCudaType<half>(float v) {
  return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 FloatToCudaType<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

template <typename CudaT>
__device__ __forceinline__ float CudaTypeToFloat(CudaT v);

template <>
__device__ __forceinline__ float CudaTypeToFloat<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ float CudaTypeToFloat<half>(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float CudaTypeToFloat<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

}  // namespace gguf_cuda
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
