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

// Both the Q8 row base and its byte stride must be 16-byte aligned. The windows
// can overlap adjacent records, but never leave the row, including its final half-block.
__device__ __forceinline__ void LoadQ8SubBlock16AlignedStage(
    const uint8_t* row_ptr, int sb32_idx, int half16, uint4& raw, float& d) {
  const uint8_t* block = row_ptr + static_cast<int64_t>(sb32_idx) * 34;
  d = Fp16ToFloatDev(__ldg(reinterpret_cast<const uint16_t*>(block)));
  const uint8_t* codes = block + 2 + half16 * 16;
  const unsigned int shift = static_cast<unsigned int>(reinterpret_cast<uintptr_t>(codes) & 15u);
  const auto* aligned = reinterpret_cast<const uint4*>(codes - shift);
  const uint4 low = __ldg(aligned);
  const uint4 high = shift ? __ldg(aligned + 1) : make_uint4(0u, 0u, 0u, 0u);
  uint32_t last;
  switch (shift >> 2) {
    case 0:
      raw = low;
      last = high.x;
      break;
    case 1:
      raw = make_uint4(low.y, low.z, low.w, high.x);
      last = high.y;
      break;
    case 2:
      raw = make_uint4(low.z, low.w, high.x, high.y);
      last = high.z;
      break;
    default:
      raw = make_uint4(low.w, high.x, high.y, high.z);
      last = high.w;
      break;
  }
  if (shift & 2u) {
    raw = make_uint4(__byte_perm(raw.x, raw.y, 0x5432),
                     __byte_perm(raw.y, raw.z, 0x5432),
                     __byte_perm(raw.z, raw.w, 0x5432),
                     __byte_perm(raw.w, last, 0x5432));
  }
}

__device__ __forceinline__ void DecodeQ8StageToBf16(
    const uint4& raw, float d, uint4& low, uint4& high) {
  auto* lo = reinterpret_cast<__nv_bfloat162*>(&low);
  auto* hi = reinterpret_cast<__nv_bfloat162*>(&high);
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int shift = 8 * s;
    const float e0 = d * static_cast<float>(static_cast<int8_t>((raw.x >> shift) & 0xFFu));
    const float e1 = d * static_cast<float>(static_cast<int8_t>((raw.y >> shift) & 0xFFu));
    const float h0 = d * static_cast<float>(static_cast<int8_t>((raw.z >> shift) & 0xFFu));
    const float h1 = d * static_cast<float>(static_cast<int8_t>((raw.w >> shift) & 0xFFu));
    lo[s] = __floats2bfloat162_rn(e0, e1);
    hi[s] = __floats2bfloat162_rn(h0, h1);
  }
}

__device__ __forceinline__ void DecodeQ8SubBlock16Aligned(
    const uint8_t* row_ptr, int sb32_idx, int half16, float (&out)[16]) {
  uint4 raw;
  float d;
  LoadQ8SubBlock16AlignedStage(row_ptr, sb32_idx, half16, raw, d);
  const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int w = 0; w < 4; ++w) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      out[w * 4 + b] = d * static_cast<float>(static_cast<int8_t>((words[w] >> (8 * b)) & 0xFFu));
    }
  }
}

// Register-only computation of ksigns_iq2xs[x7] (7-bit index -> 8-bit even-parity sign mask).
__device__ __forceinline__ uint8_t Iq2xsSignsDev(uint32_t x7) {
  const uint32_t s = x7 & 127u;
  return static_cast<uint8_t>(s | ((__popc(s) & 1u) << 7));
}

// Only integer magnitudes are encoded here; block scales are never rounded by
// the codebook. Every unsigned-byte magnitude is exactly representable in BF16.
constexpr uint32_t Iq2MagnitudeBits(uint32_t value) {
  if (value == 0) return 0;
  uint32_t exponent = 0;
  for (uint32_t remaining = value; remaining > 1; remaining >>= 1) ++exponent;
  return ((127u + exponent) << 23) | ((value << (23u - exponent)) & 0x7fffffu);
}

constexpr uint32_t Iq2GridPair(uint64_t grid, int pair) {
  const auto first = static_cast<uint32_t>((grid >> (16 * pair)) & 255u);
  const auto second = static_cast<uint32_t>((grid >> (16 * pair + 8)) & 255u);
  return (Iq2MagnitudeBits(first) >> 16) | (Iq2MagnitudeBits(second) & 0xffff0000u);
}

constexpr uint4 Iq2GridEntry(uint64_t grid) {
  return {Iq2GridPair(grid, 0), Iq2GridPair(grid, 1), Iq2GridPair(grid, 2), Iq2GridPair(grid, 3)};
}

struct Iq2Codebook {
  uint4 values[256];
};

constexpr Iq2Codebook Iq2CodebookBuild() {
  // Canonical IQ2_XXS byte codes from gguf_quant_tables.h.
  constexpr uint64_t codes[256] = {
      0x0808080808080808ULL,
      0x080808080808082bULL,
      0x0808080808081919ULL,
      0x0808080808082b08ULL,
      0x0808080808082b2bULL,
      0x0808080808190819ULL,
      0x0808080808191908ULL,
      0x08080808082b0808ULL,
      0x08080808082b082bULL,
      0x08080808082b2b08ULL,
      0x08080808082b2b2bULL,
      0x0808080819080819ULL,
      0x0808080819081908ULL,
      0x0808080819190808ULL,
      0x0808080819192b08ULL,
      0x08080808192b0819ULL,
      0x08080808192b1908ULL,
      0x080808082b080808ULL,
      0x080808082b08082bULL,
      0x080808082b082b2bULL,
      0x080808082b2b082bULL,
      0x0808081908080819ULL,
      0x0808081908081908ULL,
      0x0808081908190808ULL,
      0x0808081908191919ULL,
      0x0808081919080808ULL,
      0x080808192b081908ULL,
      0x080808192b192b08ULL,
      0x0808082b08080808ULL,
      0x0808082b0808082bULL,
      0x0808082b082b082bULL,
      0x0808082b2b08082bULL,
      0x0808190808080819ULL,
      0x0808190808081908ULL,
      0x0808190808190808ULL,
      0x08081908082b0819ULL,
      0x08081908082b1908ULL,
      0x0808190819080808ULL,
      0x080819081908082bULL,
      0x0808190819082b08ULL,
      0x08081908192b0808ULL,
      0x080819082b080819ULL,
      0x080819082b081908ULL,
      0x080819082b190808ULL,
      0x080819082b2b1908ULL,
      0x0808191908080808ULL,
      0x080819190808082bULL,
      0x0808191908082b08ULL,
      0x08081919082b0808ULL,
      0x080819191908192bULL,
      0x08081919192b2b19ULL,
      0x080819192b080808ULL,
      0x080819192b190819ULL,
      0x0808192b08082b19ULL,
      0x0808192b08190808ULL,
      0x0808192b19080808ULL,
      0x0808192b2b081908ULL,
      0x0808192b2b2b1908ULL,
      0x08082b0808080808ULL,
      0x08082b0808081919ULL,
      0x08082b0808082b08ULL,
      0x08082b0808191908ULL,
      0x08082b08082b2b08ULL,
      0x08082b0819080819ULL,
      0x08082b0819081908ULL,
      0x08082b0819190808ULL,
      0x08082b081919082bULL,
      0x08082b082b082b08ULL,
      0x08082b1908081908ULL,
      0x08082b1919080808ULL,
      0x08082b2b0808082bULL,
      0x08082b2b08191908ULL,
      0x0819080808080819ULL,
      0x0819080808081908ULL,
      0x0819080808190808ULL,
      0x08190808082b0819ULL,
      0x0819080819080808ULL,
      0x08190808192b0808ULL,
      0x081908082b081908ULL,
      0x081908082b190808ULL,
      0x081908082b191919ULL,
      0x0819081908080808ULL,
      0x0819081908082b08ULL,
      0x08190819082b0808ULL,
      0x0819081919190808ULL,
      0x0819081919192b2bULL,
      0x081908192b080808ULL,
      0x0819082b082b1908ULL,
      0x0819082b19081919ULL,
      0x0819190808080808ULL,
      0x0819190808082b08ULL,
      0x08191908082b0808ULL,
      0x08191908082b1919ULL,
      0x0819190819082b19ULL,
      0x081919082b080808ULL,
      0x0819191908192b08ULL,
      0x08191919192b082bULL,
      0x0819192b08080808ULL,
      0x0819192b0819192bULL,
      0x08192b0808080819ULL,
      0x08192b0808081908ULL,
      0x08192b0808190808ULL,
      0x08192b0819080808ULL,
      0x08192b082b080819ULL,
      0x08192b1908080808ULL,
      0x08192b1908081919ULL,
      0x08192b192b2b0808ULL,
      0x08192b2b19190819ULL,
      0x082b080808080808ULL,
      0x082b08080808082bULL,
      0x082b080808082b2bULL,
      0x082b080819081908ULL,
      0x082b0808192b0819ULL,
      0x082b08082b080808ULL,
      0x082b08082b08082bULL,
      0x082b0819082b2b19ULL,
      0x082b081919082b08ULL,
      0x082b082b08080808ULL,
      0x082b082b0808082bULL,
      0x082b190808080819ULL,
      0x082b190808081908ULL,
      0x082b190808190808ULL,
      0x082b190819080808ULL,
      0x082b19081919192bULL,
      0x082b191908080808ULL,
      0x082b191919080819ULL,
      0x082b1919192b1908ULL,
      0x082b192b2b190808ULL,
      0x082b2b0808082b08ULL,
      0x082b2b08082b0808ULL,
      0x082b2b082b191908ULL,
      0x082b2b2b19081908ULL,
      0x1908080808080819ULL,
      0x1908080808081908ULL,
      0x1908080808190808ULL,
      0x1908080808192b08ULL,
      0x19080808082b0819ULL,
      0x19080808082b1908ULL,
      0x1908080819080808ULL,
      0x1908080819082b08ULL,
      0x190808081919192bULL,
      0x19080808192b0808ULL,
      0x190808082b080819ULL,
      0x190808082b081908ULL,
      0x190808082b190808ULL,
      0x1908081908080808ULL,
      0x19080819082b0808ULL,
      0x19080819192b0819ULL,
      0x190808192b080808ULL,
      0x190808192b081919ULL,
      0x1908082b08080819ULL,
      0x1908082b08190808ULL,
      0x1908082b19082b08ULL,
      0x1908082b1919192bULL,
      0x1908082b192b2b08ULL,
      0x1908190808080808ULL,
      0x1908190808082b08ULL,
      0x19081908082b0808ULL,
      0x190819082b080808ULL,
      0x190819082b192b19ULL,
      0x190819190819082bULL,
      0x19081919082b1908ULL,
      0x1908192b08080808ULL,
      0x19082b0808080819ULL,
      0x19082b0808081908ULL,
      0x19082b0808190808ULL,
      0x19082b0819080808ULL,
      0x19082b0819081919ULL,
      0x19082b1908080808ULL,
      0x19082b1919192b08ULL,
      0x19082b19192b0819ULL,
      0x19082b192b08082bULL,
      0x19082b2b19081919ULL,
      0x19082b2b2b190808ULL,
      0x1919080808080808ULL,
      0x1919080808082b08ULL,
      0x1919080808190819ULL,
      0x1919080808192b19ULL,
      0x19190808082b0808ULL,
      0x191908082b080808ULL,
      0x191908082b082b08ULL,
      0x1919081908081908ULL,
      0x191908191908082bULL,
      0x191908192b2b1908ULL,
      0x1919082b2b190819ULL,
      0x191919082b190808ULL,
      0x191919082b19082bULL,
      0x1919191908082b2bULL,
      0x1919192b08080819ULL,
      0x1919192b19191908ULL,
      0x19192b0808080808ULL,
      0x19192b0808190819ULL,
      0x19192b0808192b19ULL,
      0x19192b08192b1908ULL,
      0x19192b1919080808ULL,
      0x19192b2b08082b08ULL,
      0x192b080808081908ULL,
      0x192b080808190808ULL,
      0x192b080819080808ULL,
      0x192b0808192b2b08ULL,
      0x192b081908080808ULL,
      0x192b081919191919ULL,
      0x192b082b08192b08ULL,
      0x192b082b192b0808ULL,
      0x192b190808080808ULL,
      0x192b190808081919ULL,
      0x192b191908190808ULL,
      0x192b19190819082bULL,
      0x192b19192b081908ULL,
      0x192b2b081908082bULL,
      0x2b08080808080808ULL,
      0x2b0808080808082bULL,
      0x2b08080808082b2bULL,
      0x2b08080819080819ULL,
      0x2b0808082b08082bULL,
      0x2b08081908081908ULL,
      0x2b08081908192b08ULL,
      0x2b08081919080808ULL,
      0x2b08082b08190819ULL,
      0x2b08190808080819ULL,
      0x2b08190808081908ULL,
      0x2b08190808190808ULL,
      0x2b08190808191919ULL,
      0x2b08190819080808ULL,
      0x2b081908192b0808ULL,
      0x2b08191908080808ULL,
      0x2b0819191908192bULL,
      0x2b0819192b191908ULL,
      0x2b08192b08082b19ULL,
      0x2b08192b19080808ULL,
      0x2b08192b192b0808ULL,
      0x2b082b080808082bULL,
      0x2b082b1908081908ULL,
      0x2b082b2b08190819ULL,
      0x2b19080808081908ULL,
      0x2b19080808190808ULL,
      0x2b190808082b1908ULL,
      0x2b19080819080808ULL,
      0x2b1908082b2b0819ULL,
      0x2b1908190819192bULL,
      0x2b1908192b080808ULL,
      0x2b19082b19081919ULL,
      0x2b19190808080808ULL,
      0x2b191908082b082bULL,
      0x2b19190819081908ULL,
      0x2b19191919190819ULL,
      0x2b192b082b080819ULL,
      0x2b192b19082b0808ULL,
      0x2b2b08080808082bULL,
      0x2b2b080819190808ULL,
      0x2b2b08082b081919ULL,
      0x2b2b081908082b19ULL,
      0x2b2b082b08080808ULL,
      0x2b2b190808192b08ULL,
      0x2b2b2b0819190808ULL,
      0x2b2b2b1908081908ULL,
  };
  Iq2Codebook packed{};
  for (int code = 0; code < 256; ++code) packed.values[code] = Iq2GridEntry(codes[code]);
  return packed;
}

// uint4 supplies the alignment required by vector magnitude loads.
static __device__ const Iq2Codebook kIq2xxsGridBf16 = Iq2CodebookBuild();

template <bool kAlignedU16 = false>
__device__ __forceinline__ void LoadIq2SubBlock16Stage(
    const uint8_t* row, int sb32_idx, int half16, uint32_t& packed, float& scale) {
  const int offset = sb32_idx * 32 + half16 * 16;
  const uint8_t* block = row + static_cast<int64_t>(offset / 256) * 66;
  const uint8_t* record = block + 2 + ((offset / 32) & 7) * 8;
  uint16_t d_bits, g2;
  uint32_t signs_scale;
  if constexpr (kAlignedU16) {
    d_bits = __ldg(reinterpret_cast<const uint16_t*>(block));
    g2 = __ldg(reinterpret_cast<const uint16_t*>(record + 2 * half16));
    const uint16_t sg_lo = __ldg(reinterpret_cast<const uint16_t*>(record + 4));
    const uint16_t sg_hi = __ldg(reinterpret_cast<const uint16_t*>(record + 6));
    signs_scale = static_cast<uint32_t>(sg_lo) | (static_cast<uint32_t>(sg_hi) << 16);
  } else {
    d_bits = ReadU16Dev(block);
    g2 = ReadU16Dev(record + 2 * half16);
    signs_scale = ReadU32Dev(record + 4);
  }
  packed = static_cast<uint32_t>(g2) | (((signs_scale >> (14 * half16)) & 0x3FFFu) << 16);
  scale = Fp16ToFloatDev(d_bits) * (0.5f + static_cast<float>(signs_scale >> 28)) * 0.25f;
}

__device__ __forceinline__ void DecodeIq2StageToBf16(
    uint32_t packed, float scale, uint4& low, uint4& high) {
  auto* low_words = reinterpret_cast<uint32_t*>(&low);
  auto* high_words = reinterpret_cast<uint32_t*>(&high);
#pragma unroll
  for (int group = 0; group < 2; ++group) {
    const uint32_t code = (packed >> (8 * group)) & 255u;
    const uint4 grid = __ldg(kIq2xxsGridBf16.values + code);
    const uint32_t words[4] = {grid.x, grid.y, grid.z, grid.w};
    const uint32_t signs = Iq2xsSignsDev(packed >> (16 + 7 * group));
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const uint32_t sign_mask =
          (signs * ((1u << (15 - 2 * pair)) + (1u << (30 - 2 * pair)))) & 0x80008000u;
      const uint32_t signed_words = words[pair] | sign_mask;
      const float a = scale * __uint_as_float(signed_words << 16);
      const float b = scale * __uint_as_float(signed_words & 0xffff0000u);
      const auto p = __floats2bfloat162_rn(a, b);
      const uint32_t bits = *reinterpret_cast<const uint32_t*>(&p);
      if (pair & 1) {
        high_words[2 * group + pair / 2] = bits;
      } else {
        low_words[2 * group + pair / 2] = bits;
      }
    }
  }
}

// Adjacent BF16 pairs match the chunk64 IQ2 MMA layout. Reconstruct the complete
// signed operand in FP32 before its only BF16 rounding boundary.
template <bool kCodesWord = false, bool kSignBits = false, bool kAlignedRecordWindow = false>
__device__ __forceinline__ void DecodeIq2SubBlock16Bf16(
    const uint8_t* row, int sb32_idx, int half16, uint4& low, uint4& high) {
  uint32_t packed;
  float scale;
  LoadIq2SubBlock16Stage<kAlignedRecordWindow>(row, sb32_idx, half16, packed, scale);
  DecodeIq2StageToBf16(packed, scale, low, high);
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
  } else if constexpr (kCompileType == kGgmlTypeNVFP4) {
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb32_idx >> 1) * 36;
    const int group = 2 * (sb32_idx & 1) + half16;
    const uint8_t scale = __ldg(blk + group);
    const float d = scale == 127
                        ? 0.0f
                        : __half2float(__half(__nv_cvt_fp8_to_halfraw(
                              static_cast<__nv_fp8_storage_t>(scale), __NV_E4M3)));
    const uint8_t* codes = blk + 4 + group * 8;
    const uint32_t words[2] = {ReadU32Dev(codes), ReadU32Dev(codes + 4)};
    // The low/high nibbles share one FP8 scale and encode two 8-value halves.
    // Keep FP32 reconstruction before the caller's complete-operand BF16 cast.
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t code = words[j / 4] >> (8 * (j & 3));
      out[j] = d * (0.5f * static_cast<float>(kvalues_fp4[code & 15u]));
      out[j + 8] = d * (0.5f * static_cast<float>(kvalues_fp4[(code >> 4) & 15u]));
    }
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

// FP4 magnitudes encoded as BF16(value * 2^-126), in the low/high 16-bit fields.
__device__ __forceinline__ uint32_t Nvfp4ScaledBf16PairBits(uint32_t codes) {
  const uint32_t magnitude = codes & 0x00070007u;
  const uint32_t nonzero_mask = ((magnitude + 0x00070007u) & 0x00080008u) * 0x1fffu;
  // The GGUF value table maps code 8 to positive zero, not negative zero.
  return (magnitude << 6) | (((codes & 0x00080008u) << 12) & nonzero_mask);
}

__device__ __forceinline__ void DecodeNvfp4WordsBf16(
    uint2 raw, uint8_t scale_byte, uint4& low, uint4& high) {
  const uint32_t q0 = raw.x, q1 = raw.y;
  const float d = scale_byte == 127
                      ? 0.0f
                      : __half2float(__half(__nv_cvt_fp8_to_halfraw(
                            static_cast<__nv_fp8_storage_t>(scale_byte), __NV_E4M3)));
  const __nv_bfloat162 scale = __bfloat162bfloat162(__float2bfloat16_rn(d * 0x1p110f));
  const __nv_bfloat162 restore = __floats2bfloat162_rn(0x1p16f, 0x1p16f);
  auto* lo = reinterpret_cast<uint32_t*>(&low);
  auto* hi = reinterpret_cast<uint32_t*>(&high);
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const uint32_t pair = __byte_perm(q0, q1, 0x4400u + 0x1111u * s);
    const uint32_t l = Nvfp4ScaledBf16PairBits(pair);
    const uint32_t h = Nvfp4ScaledBf16PairBits(pair >> 4);
    // FP8*2^110 and each finite scaled product are exact BF16 values.
    // Restore before MMA: its operands and reduction must not be rescaled.
    const auto pl = __hmul2(__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&l), scale), restore);
    const auto ph = __hmul2(__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&h), scale), restore);
    lo[s] = *reinterpret_cast<const uint32_t*>(&pl);
    hi[s] = *reinterpret_cast<const uint32_t*>(&ph);
  }
}

template <bool kAlignedRows>
__device__ __forceinline__ void DecodeNvfp4SubBlock16Bf16(
    const uint8_t* row_ptr, int sb32_idx, int half16, uint4& low, uint4& high) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
  const uint8_t* blk = row_ptr + static_cast<int64_t>(sb32_idx >> 1) * 36;
  const int group = 2 * (sb32_idx & 1) + half16;
  const uint8_t scale_byte = __ldg(blk + group);
  const uint8_t* codes = blk + 4 + group * 8;
  uint2 raw;
  if constexpr (kAlignedRows) {
    raw.x = __ldg(reinterpret_cast<const uint32_t*>(codes));
    raw.y = __ldg(reinterpret_cast<const uint32_t*>(codes + 4));
  } else {
    raw.x = ReadU32Dev(codes);
    raw.y = ReadU32Dev(codes + 4);
  }
  DecodeNvfp4WordsBf16(raw, scale_byte, low, high);
#else
  // Keep the established FP32 decoder on architectures not qualified for this
  // subnormal BF16 input encoding.
  float weights[16], l[8], h[8];
  DecodeGgufSubBlock16Impl<kGgmlTypeNVFP4>(row_ptr, sb32_idx, half16, weights);
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    l[2 * s] = weights[s];
    l[2 * s + 1] = weights[s + 4];
    h[2 * s] = weights[s + 8];
    h[2 * s + 1] = weights[s + 12];
  }
  low = PackFloat8ToBf16x2(l);
  high = PackFloat8ToBf16x2(h);
#endif
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
