// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Convert scale and zero_point from MatMulNBits to the format required by fpA_intB_gemm or fpA_intB_gemv kernels.
namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

template <bool is_zero_point_int4_packed, typename T, typename Z>
void launch_scaled_zero_point_kernel(
    cudaStream_t stream,
    const Z* zero_point,
    const T* transposed_scale,
    T* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);

template <typename T>
void launch_transpose_scale_kernel(
    cudaStream_t stream,
    const T* scale,
    T* transposed_scale,
    int n, int k_blocks);

// Transpose uint4 weight matrix and add default zero points then pack as int8.
void unpack_uint4_transposed_to_int8_direct_cuda(cudaStream_t stream,
                                                 void* packed_transposed_weight,
                                                 const void* packed_weight,
                                                 int n,
                                                 int k);

// Transpose uint8 weight matrix and add default zero points as int8.
void transpose_uint8_matrix_and_convert_to_int8(cudaStream_t stream,
                                                int8_t* output,        // shape: (k, n)
                                                const uint8_t* input,  // shape: (n, k)
                                                int n, int k);

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
