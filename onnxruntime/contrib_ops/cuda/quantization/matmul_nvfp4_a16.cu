// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_nvfp4_a16.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

#include <algorithm>

namespace onnxruntime::contrib::cuda {
namespace {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
template <typename T>
__global__ void DequantizeLocalKernel(T* weight, const uint8_t* packed, const uint8_t* scales,
                                      int64_t count, int64_t k) {
  constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const unsigned code = (packed[i / 2] >> (4 * (i % 2))) & 15;
    const float local_scale = e4m3_to_float(scales[(i / k) * (k / 16) + (i % k) / 16]);
    const float value = magnitudes[code & 7] * ((code & 8) ? -1.0f : 1.0f);
    weight[i] = from_float<T>(value * local_scale);
  }
}

template <typename T>
__global__ void EpilogueKernel(T* y, const float* accumulator, const float* global_scale,
                               int64_t count) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const float sum = accumulator == nullptr ? 0.0f : accumulator[i];
    y[i] = from_float<T>(sum * *global_scale);
  }
}

int GridSize(int64_t count) {
  return static_cast<int>(std::min<int64_t>((count - 1) / 256 + 1, 65535));
}
#endif

}  // namespace

Status LaunchDequantizeNVFP4Local(void* weight, const uint8_t* packed, const uint8_t* scales,
                                  int64_t n, int64_t k, bool is_bf16, cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  const int64_t count = n * k;
  if (count == 0) {
    return Status::OK();
  }
  if (is_bf16) {
    DequantizeLocalKernel<<<GridSize(count), 256, 0, stream>>>(
        static_cast<nv_bfloat16*>(weight), packed, scales, count, k);
  } else {
    DequantizeLocalKernel<<<GridSize(count), 256, 0, stream>>>(
        static_cast<half*>(weight), packed, scales, count, k);
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(weight);
  ORT_UNUSED_PARAMETER(packed);
  ORT_UNUSED_PARAMETER(scales);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulNVFP4A16 requires CUDA 12.8 or newer.");
#endif
}

Status LaunchNVFP4A16Epilogue(void* y, const float* accumulator, const float* global_scale,
                              int64_t count, bool is_bf16, cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  if (count == 0) {
    return Status::OK();
  }
  if (is_bf16) {
    EpilogueKernel<<<GridSize(count), 256, 0, stream>>>(
        static_cast<nv_bfloat16*>(y), accumulator, global_scale, count);
  } else {
    EpilogueKernel<<<GridSize(count), 256, 0, stream>>>(
        static_cast<half*>(y), accumulator, global_scale, count);
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(accumulator);
  ORT_UNUSED_PARAMETER(global_scale);
  ORT_UNUSED_PARAMETER(count);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulNVFP4A16 requires CUDA 12.8 or newer.");
#endif
}

}  // namespace onnxruntime::contrib::cuda
