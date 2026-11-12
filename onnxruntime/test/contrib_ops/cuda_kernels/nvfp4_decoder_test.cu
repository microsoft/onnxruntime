// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/gguf_quant_cuda.cuh"

namespace onnxruntime::test {
namespace {

__global__ void DecodeNvfp4OracleKernel(uint16_t* output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 128 * 16) return;
  const int scale = index / 16, first_code = index % 16;
  uint2 raw = make_uint2(0, 0);
  for (int position = 0; position < 16; ++position) {
    const uint32_t code = (first_code + position) % 16;
    const int shift = 8 * (position % 4) + 4 * (position / 8);
    if (position % 8 < 4)
      raw.x |= code << shift;
    else
      raw.y |= code << shift;
  }
  uint4 low, high;
  contrib::cuda::gguf_cuda::DecodeNvfp4WordsBf16(raw, static_cast<uint8_t>(scale), low, high);
  const auto* lo = reinterpret_cast<const uint32_t*>(&low);
  const auto* hi = reinterpret_cast<const uint32_t*>(&high);
  for (int s = 0; s < 4; ++s) {
    output[index * 16 + s] = static_cast<uint16_t>(lo[s]);
    output[index * 16 + s + 4] = static_cast<uint16_t>(lo[s] >> 16);
    output[index * 16 + s + 8] = static_cast<uint16_t>(hi[s]);
    output[index * 16 + s + 12] = static_cast<uint16_t>(hi[s] >> 16);
  }
#endif
}

}  // namespace

cudaError_t LaunchNvfp4DecoderOracle(uint16_t* output) {
  DecodeNvfp4OracleKernel<<<8, 256, 0, nullptr>>>(output);
  return cudaGetLastError();
}

}  // namespace onnxruntime::test
