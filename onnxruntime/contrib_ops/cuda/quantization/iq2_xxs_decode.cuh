// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/quantization/iq2_xxs_grid.h"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime::contrib::cuda::iq2_xxs {

// All 32 lanes participate. Lane zero receives the FP32 dot product; complete
// reconstructed weights round to T before multiplication, not their block scales.
template <typename T>
__device__ __forceinline__ float Dot(const T* a, const uint8_t* b, int64_t k_blocks) {
  using namespace onnxruntime::cuda;
  const int lane = threadIdx.x % 32;
  const int vector = lane % 4;
  const int group = lane / 4;
  float sum = 0.0f;
  for (int64_t block = 0; block < k_blocks; ++block) {
    const uint8_t* packed = b + block * 66;
    // GGML's 66-byte block stride does not guarantee uint32 alignment.
    const uint16_t scale_bits = static_cast<uint16_t>(packed[0]) |
                                (static_cast<uint16_t>(packed[1]) << 8);
    const float d = __half2float(__ushort_as_half(scale_bits));
    const uint8_t* record = packed + 2 + group * 8;
    const uint32_t signs_scale = static_cast<uint32_t>(record[4]) |
                                 (static_cast<uint32_t>(record[5]) << 8) |
                                 (static_cast<uint32_t>(record[6]) << 16) |
                                 (static_cast<uint32_t>(record[7]) << 24);
    const unsigned sign_index = (signs_scale >> (7 * vector)) & 127;
    const unsigned signs = sign_index | ((__popc(sign_index) & 1) << 7);
    const float scale = d * (0.5f + static_cast<float>(signs_scale >> 28)) * 0.25f;
    const uint64_t grid = kIQ2XXSGrid[record[vector]];
    const int64_t a_base = block * 256 + group * 32 + vector * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float magnitude = static_cast<float>((grid >> (8 * j)) & 255);
      const float weight = to_float<T>(
          from_float<T>(scale * magnitude * ((signs & (1u << j)) ? -1.0f : 1.0f)));
      sum = fmaf(to_float<T>(a[a_base + j]), weight, sum);
    }
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  return sum;
}

}  // namespace onnxruntime::contrib::cuda::iq2_xxs
