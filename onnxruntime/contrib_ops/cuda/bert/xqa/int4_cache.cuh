#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename Element>
__device__ inline uint4 DequantizeInt4CacheGrain(uint32_t packed, float scale) {
  union {
    Element elements[8];
    uint4 storage;
  } result;
#pragma unroll
  for (uint32_t channel = 0; channel < 8; ++channel) {
    const int code = static_cast<int>((packed >> (channel * 4)) & 15) - 8;
    result.elements[channel] = static_cast<Element>(static_cast<float>(code) * scale);
  }
  return result.storage;
}