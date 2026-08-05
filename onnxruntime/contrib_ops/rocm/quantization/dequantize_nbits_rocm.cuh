// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Declarations of the ROCm dequantize launchers used by MatMulNBits.
// Definitions live in dequantize_blockwise.cu.

#pragma once

#include <hip/hip_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace rocm {

// Dequantize a column-wise 4-bit packed weight matrix.
// quant_data : [N, k_blocks, block_size/2]  (two nibbles per byte)
// scales     : [N, k_blocks]
// zero_points: [N, ceil(k_blocks/2)]  (packed nibbles, default midpoint=8 if null)
// output     : [N, K_padded]          (K_padded = k_blocks * block_size)
template <typename T>
hipError_t LaunchDequantize4Bits(
    T* output,
    const unsigned char* quant_data,
    const T* scales,
    const unsigned char* zero_points,  // may be nullptr
    int K_padded,
    int N,
    int block_size,
    hipStream_t stream);

// Dequantize a column-wise 8-bit packed weight matrix.
template <typename T>
hipError_t LaunchDequantize8Bits(
    T* output,
    const unsigned char* quant_data,
    const T* scales,
    const unsigned char* zero_points,  // may be nullptr
    int K_padded,
    int N,
    int block_size,
    hipStream_t stream);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
