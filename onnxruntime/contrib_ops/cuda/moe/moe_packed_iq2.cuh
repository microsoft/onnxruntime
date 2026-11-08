// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/quantization/gguf_quant_cuda.cuh"

namespace onnxruntime::contrib::cuda {

// Encode the canonical unsigned-byte magnitudes exactly. Every 8-bit integer is
// representable in BF16; this table never contains or rounds a block scale.
constexpr uint32_t MoEPackedIq2MagnitudeBits(uint32_t value) {
  if (value == 0) return 0;
  uint32_t exponent = 0;
  for (uint32_t remaining = value; remaining > 1; remaining >>= 1) ++exponent;
  return ((127u + exponent) << 23) | ((value << (23u - exponent)) & 0x7fffffu);
}

constexpr uint32_t MoEPackedIq2GridPair(uint64_t grid, int pair) {
  const auto first = static_cast<uint32_t>((grid >> (16 * pair)) & 255u);
  const auto second = static_cast<uint32_t>((grid >> (16 * pair + 8)) & 255u);
  return (MoEPackedIq2MagnitudeBits(first) >> 16) |
         (MoEPackedIq2MagnitudeBits(second) & 0xffff0000u);
}

constexpr uint4 MoEPackedIq2GridEntry(uint64_t grid) {
  return {MoEPackedIq2GridPair(grid, 0), MoEPackedIq2GridPair(grid, 1),
          MoEPackedIq2GridPair(grid, 2), MoEPackedIq2GridPair(grid, 3)};
}

struct MoEPackedIq2Codebook {
  uint4 values[256];
};

constexpr MoEPackedIq2Codebook MoEPackedIq2CodebookBuild() {
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
  MoEPackedIq2Codebook packed{};
  for (int code = 0; code < 256; ++code) packed.values[code] = MoEPackedIq2GridEntry(codes[code]);
  return packed;
}

// uint4 supplies the 16-byte alignment required by vector magnitude loads.
static __device__ const MoEPackedIq2Codebook kMoEPackedIq2Grid = MoEPackedIq2CodebookBuild();

// Reconstruct both eight-element grids from one GGML record. Keep the complete
// FP32 scale*grid reconstruction before rounding each operand to BF16.
template <bool kCodesWord = true>
__device__ __forceinline__ void MoEPackedIq2Fragment(
    const uint8_t* row, int offset, uint32_t (&low)[4], uint32_t (&high)[4]) {
  const uint8_t* block = row + static_cast<int64_t>(offset / 256) * 66;
  const uint8_t* record = block + 2 + ((offset / 32) & 7) * 8;
  const uint32_t codes = kCodesWord ? gguf_cuda::ReadU32Dev(record) : 0u;
  const uint32_t signs_scale = gguf_cuda::ReadU32Dev(record + 4);
  const float scale = gguf_cuda::Fp16ToFloatDev(gguf_cuda::ReadU16Dev(block)) *
                      (0.5f + static_cast<float>(signs_scale >> 28)) * 0.25f;
  const int half = (offset / 16) & 1;
#pragma unroll
  for (int group = 0; group < 2; ++group) {
    const int grid_index = 2 * half + group;
    const uint32_t code = kCodesWord ? ((codes >> (8 * grid_index)) & 255u) : __ldg(record + grid_index);
    const uint4 grid = __ldg(kMoEPackedIq2Grid.values + code);
    const uint32_t words[4] = {grid.x, grid.y, grid.z, grid.w};
    const uint8_t signs = gguf_cuda::Iq2xsSignsDev(signs_scale >> (7 * grid_index));
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const int j = 2 * pair;
      const float a = scale * __uint_as_float(words[pair] << 16) *
                      ((signs & (1u << j)) ? -1.f : 1.f);
      const float b = scale * __uint_as_float(words[pair] & 0xffff0000u) *
                      ((signs & (1u << (j + 1))) ? -1.f : 1.f);
      const auto packed = __floats2bfloat162_rn(a, b);
      const uint32_t bits = *reinterpret_cast<const uint32_t*>(&packed);
      if (pair & 1) {
        high[2 * group + pair / 2] = bits;
      } else {
        low[2 * group + pair / 2] = bits;
      }
    }
  }
}

}  // namespace onnxruntime::contrib::cuda
