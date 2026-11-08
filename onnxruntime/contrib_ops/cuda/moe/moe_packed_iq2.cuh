// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/quantization/gguf_quant_cuda.cuh"

namespace onnxruntime::contrib::cuda {

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
    const uint64_t grid = __ldg(gguf_quant::iq2xxs_grid + code);
    const uint8_t signs = gguf_cuda::Iq2xsSignsDev(signs_scale >> (7 * grid_index));
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const int j = 2 * pair;
      const float a = scale * static_cast<float>((grid >> (8 * j)) & 255u) *
                      ((signs & (1u << j)) ? -1.f : 1.f);
      const float b = scale * static_cast<float>((grid >> (8 * (j + 1))) & 255u) *
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
