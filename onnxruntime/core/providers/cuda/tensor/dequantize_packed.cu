// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/quantization/dequantize_packed_impl.h"

namespace onnxruntime {
namespace cuda {
namespace {

template <typename T>
__device__ __forceinline__ T ConvertPacked(float value) { return static_cast<T>(value); }

template <>
__device__ __forceinline__ half ConvertPacked(float value) { return __float2half_rn(value); }

template <>
__device__ __forceinline__ __nv_bfloat16 ConvertPacked(float value) { return __float2bfloat16_rn(value); }

template <int Kind, int Elements, int Bytes, typename T>
__global__ void DequantizePackedKernel(const uint8_t* __restrict__ input, T* __restrict__ output, int64_t count) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < count;) {
    const auto* block = input + (j / Elements) * Bytes;
    output[j] = ConvertPacked<T>(dequantize_packed::Decode<Kind>(block, static_cast<int>(j % Elements)));
    if (count - j <= stride) break;
    j += stride;
  }
}

template <typename T>
Status Launch(cudaStream_t stream, const uint8_t* input, void* output, int format, int64_t count) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((count - 1) / threads + 1, 65535));
  const bool supported = dequantize_packed::Dispatch(format, [&](auto kind, auto elements, auto bytes) {
    DequantizePackedKernel<decltype(kind)::value, decltype(elements)::value, decltype(bytes)::value, T>
        <<<blocks, threads, 0, stream>>>(
            input, static_cast<T*>(output), count);
  });
  ORT_RETURN_IF_NOT(supported, "DequantizePacked: unsupported packed format ID ", format,
                    ". Select one of the 27 fixed GGUF formats.");
  return Status::OK();
}

}  // namespace

Status LaunchDequantizePacked(cudaStream_t stream, const uint8_t* input, void* output,
                              int format, int output_dtype, int64_t elements) {
  ORT_RETURN_IF_NOT(elements > 0, "DequantizePacked: launch requires a positive output element count.");
  switch (output_dtype) {
    case 1:
      ORT_RETURN_IF_ERROR(Launch<float>(stream, input, output, format, elements));
      break;
    case 10:
      ORT_RETURN_IF_ERROR(Launch<half>(stream, input, output, format, elements));
      break;
    case 16:
      ORT_RETURN_IF_ERROR(Launch<__nv_bfloat16>(stream, input, output, format, elements));
      break;
    case 11:
      ORT_RETURN_IF_ERROR(Launch<double>(stream, input, output, format, elements));
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "DequantizePacked: select FLOAT, FLOAT16, BFLOAT16, or DOUBLE output_dtype.");
  }
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace onnxruntime
