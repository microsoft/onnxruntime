// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda_runtime.h>

// Convert scale and zero_point from MatMulNBits to the format required by fpA_intB_gemm or fpA_intB_gemv kernels.
namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

template <bool is_zero_point_int4_packed, typename T, typename Z>
void launch_scaled_zero_point_kernel(
    cudaStream_t stream,
    const T* scale,
    const Z* zero_point,
    T* transposed_scale,
    T* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);

// unpack int4 packed transposed weight of shape (n, k/2) to int8 weight of shape (k, n)
void unpack_uint4_transposed_to_int8_cuda(cudaStream_t stream, void* packed_transposed_weight, void* transposed_weight, const void* weight, int n, int k);

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
