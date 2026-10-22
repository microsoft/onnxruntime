// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/gated_rms_norm.h"

#include <algorithm>
#include <cuda_fp16.h>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {
namespace {

__global__ void GatedRMSNormKernel(const half* input, const half* gate,
                                   const float* scale, half* output, int rows, float epsilon) {
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int lane = threadIdx.x;
    const int64_t offset = static_cast<int64_t>(row) * 128;
    float sum = 0.0f;
    // Match cuWelfordMuSigma2<float,float,true>: four adjacent values per lane.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float value = __half2float(input[offset + 4 * lane + i]);
      sum = sum + value * value;
    }
#pragma unroll
    for (int stride = 16; stride > 0; stride /= 2) {
      sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, stride));
    }
    const float variance = __shfl_sync(0xffffffff, sum / 128.0f, 0);
    const float inverse = rsqrtf(variance + epsilon);
#pragma unroll
    for (int i = lane; i < 128; i += 32) {
      const float value = __half2float(input[offset + i]);
      const float normalized = __fmul_rn(inverse, value);
      const float weighted = __fmul_rn(scale[i], normalized);
      const float g = __half2float(gate[offset + i]);
      // Optimized Swish is QuickGelu(alpha=1), with an FP32 sigmoid/product.
      float activated;
      if (g < -87.0f) {
        const double wide = static_cast<double>(g);
        const double exponent = exp(wide);
        activated = static_cast<float>(wide * exponent / (1.0 + exponent));
      } else {
        const float exponent = expf(g >= 0.0f ? -g : g);
        const float sigmoid = g >= 0.0f ? 1.0f / (1.0f + exponent) : exponent / (1.0f + exponent);
        activated = __fmul_rn(g, sigmoid);
      }
      output[offset + i] = __float2half_rn(__fmul_rn(weighted, activated));
    }
  }
}

}  // namespace

Status LaunchGatedRMSNorm(cudaStream_t stream, const void* input, const void* gate,
                          const float* scale, void* output, int rows, float epsilon) {
  GatedRMSNormKernel<<<std::min(rows, 65535), 32, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(gate), scale,
      static_cast<half*>(output), rows, epsilon);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda
