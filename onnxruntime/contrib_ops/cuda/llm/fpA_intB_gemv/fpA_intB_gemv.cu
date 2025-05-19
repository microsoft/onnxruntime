// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// #include <cuda_runtime_api.h>
// #include <iostream>
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/fpA_intB_gemv.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/details.h"

namespace ort_llm {
namespace kernels {
namespace fpA_intB_gemv {

void kernel_launcher(int arch, Params& params, cudaStream_t s)
{
#define EXEC(KType, A, B, Layout, ConverterInterleave)                                                                 \
    if (params.type == KType)                                                                                          \
    {                                                                                                                  \
        select_gs<kernel_type_traits<KType>::isGroupwise, KernelDetails<A, B, Layout, ConverterInterleave, 64>>(       \
            params, s);                                                                                                \
        return;                                                                                                        \
    }

// This is not used since there is no alpha for MatMulNBits currently.
#define EXEC_W4A8(KType, A, B, Layout, ConverterInterleave)                                                            \
    if (params.type == KType && params.apply_alpha_in_advance)                                                         \
    {                                                                                                                  \
        select_gs<kernel_type_traits<KType>::isGroupwise, KernelDetails<A, B, Layout, ConverterInterleave, 128>>(      \
            params, s);                                                                                                \
        return;                                                                                                        \
    }

    if (arch >= 75 && arch < 80)
    {
        EXEC(KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
        EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    }
    else if (arch >= 80 && arch < 90 || arch >= 100)
    {
        // if (arch == 89 || arch >= 120)
        // {
        //     EXEC_W4A8(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
        //     EXEC_W4A8(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
        // }
        EXEC(KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
        EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);

        // EXEC(KernelType::BF16Int8Groupwise, BF16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
        // EXEC(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    }
    else if (arch >= 90)
    {
        // Dispatchers for W4A8 groupwise
        // EXEC_W4A8(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);
        // EXEC_W4A8(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);

        EXEC(KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleavedForHopper, true);
        EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);

        // EXEC(KernelType::BF16Int8Groupwise, BF16DetailsA, Int8DetailsW, ColumnMajorInterleavedForHopper, true);
        // EXEC(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);
    }
#undef EXEC_W4A8
#undef EXEC
}

bool is_supported(int arch, KernelType kernel_type)
{
#define SUPPORT(Type)                                                                                                  \
    if (kernel_type == Type)                                                                                           \
        return true;

    if (arch >= 75 && arch < 80)
    {
        SUPPORT(KernelType::FP16Int8Groupwise);
        SUPPORT(KernelType::FP16Int4Groupwise);
    }
    else if (arch >= 80)
    {
        SUPPORT(KernelType::FP16Int8Groupwise);
        SUPPORT(KernelType::FP16Int4Groupwise);
        // SUPPORT(KernelType::BF16Int8Groupwise);
        // SUPPORT(KernelType::BF16Int4Groupwise);
    }
    return false;
#undef SUPPORT
}


// CUDA kernel to compute -scale * zero_point and transpose
// Each thread computes one element of the OUTPUT matrix (shape [k_blocks, n])
template <bool is_zero_point_int4_packed, typename T, typename Z>
__global__ void computeScaledZeroPointAndTransposeKernel(
    const T* scale,        // Input scale matrix [n, k_blocks]
    const Z* zero_point,   // Input zero_point matrix [n, k_blocks]
    T* transposed_scale,   // transposed scale [k_blocks, n]
    T* scaled_zero_point,  // Output matrix [k_blocks, n]
    int n,                 // Rows of input matrices
    int k_blocks,          // Columns of input matrices
    float default_zero_point) {
  // Calculate the output matrix coordinates [row, col] for this thread
  // The output matrix has dimensions [k_blocks, n]
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check bounds to ensure we are within the output matrix dimensions [k_blocks, n]
  if (out_row < k_blocks && out_col < n) {
    int in_row = out_col;
    int in_col = out_row;
    int64_t input_offset = static_cast<int64_t>(in_row) * k_blocks + in_col;
    int64_t output_offset = static_cast<int64_t>(out_row) * n + out_col;

    // Perform the computation: scaled_zero_point[out_row, out_col] = -scale[in_row, in_col] * zero_point[in_row, in_col]
    T scale_val = scale[input_offset];
    float zero_point_val;
    if (zero_point != nullptr) {
      if constexpr (is_zero_point_int4_packed) {  // zero point is 4 bit, and two elements are packed into one byte.
        int64_t packed_zp_offset = static_cast<int64_t>(in_row) * k_blocks + in_col / 2;
        uint8_t packed_zp = zero_point[packed_zp_offset];
        zero_point_val = static_cast<float>((in_col & 0x01) ? (packed_zp >> 4) : (packed_zp & 0x0f));
      } else {
        zero_point_val = static_cast<float>(zero_point[input_offset]);
      }
    } else {
      zero_point_val = default_zero_point;
    }

    float result = static_cast<float>(scale_val) * (-zero_point_val + default_zero_point);
    scaled_zero_point[output_offset] = static_cast<T>(result);
    transposed_scale[output_offset] = scale_val;
  }
}

template<bool is_zero_point_int4_packed, typename T, typename Z>
void launch_scaled_zero_point_kernel(
    cudaStream_t stream,
    const T* scale,
    const Z* zero_point,
    T* transposed_scale,
    T* scaled_zero_point,
    int n, int k_blocks, float default_zero_point)
{
    constexpr int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (n + blockDim.x - 1) / blockDim.x, // Grid size in x covers output columns (n)
        (k_blocks + blockDim.y - 1) / blockDim.y  // Grid size in y covers output rows (k_blocks)
    );

    computeScaledZeroPointAndTransposeKernel<is_zero_point_int4_packed, T, Z><<<gridDim, blockDim, 0, stream>>>(
        scale,
        zero_point,
        transposed_scale,
        scaled_zero_point,
        n,
        k_blocks,
        default_zero_point
    );
}

// Explicit instantiations:
template
void launch_scaled_zero_point_kernel<false, float, float>(
    cudaStream_t stream,
    const float* scale,
    const float* zero_point,
    float* transposed_scale,
    float* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);

template
void launch_scaled_zero_point_kernel<false, float, uint8_t>(
    cudaStream_t stream,
    const float* scale,
    const uint8_t* zero_point,
    float* transposed_scale,
    float* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);


template
void launch_scaled_zero_point_kernel<false, half, half>(
    cudaStream_t stream,
    const half* scale,
    const half* zero_point,
    half* transposed_scale,
    half* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);

template
void launch_scaled_zero_point_kernel<false, half, uint8_t>(
    cudaStream_t stream,
    const half* scale,
    const uint8_t* zero_point,
    half* transposed_scale,
    half* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);

// zero point is 4 bits packed.
template
void launch_scaled_zero_point_kernel<true, half, uint8_t>(
    cudaStream_t stream,
    const half* scale,
    const uint8_t* zero_point,
    half* transposed_scale,
    half* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);

// zero point is 4 bits packed.
template
void launch_scaled_zero_point_kernel<true, float, uint8_t>(
    cudaStream_t stream,
    const float* scale,
    const uint8_t* zero_point,
    float* transposed_scale,
    float* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);

// CUDA kernel to unpack int4 to int8 and transposed tensor
__global__ void unpack_uint4_transposed_to_int8_kernel(
    const unsigned char* __restrict__ packed_weight,
    signed char* __restrict__ transposed_weight,
    signed char* __restrict__ packed_transposed_weight,
    int total_packed_elements,
    int n,
    int k) {
  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (flat_idx < total_packed_elements) {
    unsigned char packed_data = packed_weight[flat_idx];

    // Extract the two int4 values
    constexpr signed char default_zero_point = 8;
    signed char elt_0 = (signed char)(packed_data & 0x0f) - default_zero_point;  // Sign extension for lower 4 bits
    signed char elt_1 = (signed char)(packed_data >> 4) - default_zero_point;    // Sign extension for upper 4 bits

    // Calculate the corresponding indices in the input (n, k/2) tensor
    // flat_idx maps to (c, r) in (n, k/2), where c is the row, r is the column
    const int input_cols = k / 2;   // Number of columns in the input tensor
    int c = flat_idx / input_cols;  // Corresponds to 'n' dimension
    int r = flat_idx % input_cols;  // Corresponds to 'k/2' dimension

    int out_row_0 = 2 * r;
    int out_row_1 = 2 * r + 1;
    int out_col = c;

    // Calculate the corresponding indices in the output (k, n) tensor
    int flat_out_idx_0 = out_row_0 * n + out_col;
    int flat_out_idx_1 = out_row_1 * n + out_col;

    transposed_weight[flat_out_idx_0] = elt_0;
    transposed_weight[flat_out_idx_1] = elt_1;

    int out_idx_0 = flat_out_idx_0 / 2;
    int out_idx_1 = flat_out_idx_1 / 2;


    if (flat_out_idx_0 % 2 == 0) {
      packed_transposed_weight[out_idx_0] += elt_0;
    } else {
      packed_transposed_weight[out_idx_0] += elt_0 * 16;
    }

    if (flat_out_idx_1 % 2 == 0) { // lower 4 bits
      packed_transposed_weight[out_idx_1] += elt_1;
    } else { // higher 4 bits
      packed_transposed_weight[out_idx_1] += elt_1 * 16;
    }
  }
}


// CUDA kernel to unpack tensor to int8
__global__ void unpack_uint4_transposed_to_int8_kernel(
    const unsigned char* __restrict__ packed_weight,
    signed char* __restrict__ transposed_weight,
    signed char* __restrict__ packed_transposed_weight,
    int n,
    int k) {
  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (flat_idx < n * k / 2) {
    unsigned char packed_data = packed_weight[flat_idx];

    // Extract the two int4 values
    constexpr signed char default_zero_point = 8;
    signed char elt_0 = (signed char)(packed_data & 0x0f) - default_zero_point;  // Sign extension for lower 4 bits
    signed char elt_1 = (signed char)(packed_data >> 4) - default_zero_point;    // Sign extension for upper 4 bits

    // Calculate the corresponding indices in the input (n, k/2) tensor
    // flat_idx maps to (c, r) in (n, k/2), where c is the row, r is the column
    const int input_cols = k / 2;   // Number of columns in the input tensor
    int c = flat_idx / input_cols;  // Corresponds to 'n' dimension
    int r = flat_idx % input_cols;  // Corresponds to 'k/2' dimension

    int out_row_0 = 2 * r;
    int out_row_1 = 2 * r + 1;
    int out_col = c;

    // Calculate the corresponding indices in the output (k, n) tensor
    int flat_out_idx_0 = out_row_0 * n + out_col;
    int flat_out_idx_1 = out_row_1 * n + out_col;

    transposed_weight[flat_out_idx_0] = elt_0;
    transposed_weight[flat_out_idx_1] = elt_1;
 }
}

// CUDA kernel to unpack tensor to int8
__global__ void pack_uint4_to_int8_kernel(
    signed char* __restrict__ transposed_weight,
    signed char* __restrict__ packed_transposed_weight,
    int n,
    int k) {
  int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (out_idx < n * k / 2) {
    int low_idx = 2 * out_idx;
    int high_idx = low_idx + 1;
    signed char elt_0 = transposed_weight[low_idx];
    signed char elt_1 = transposed_weight[high_idx];
    packed_transposed_weight[out_idx] = (unsigned char)((elt_0 & 0x0f) | ((elt_1 & 0x0f) << 4));
  }
}



void unpack_uint4_transposed_to_int8_cuda(
    cudaStream_t stream, void* packed_transposed_weight, void* transposed_weight, const void* weight, int n, int k) {
    int threads_per_block = 256;
    int num_blocks = (n * k / 2 + threads_per_block - 1) / threads_per_block;

    unpack_uint4_transposed_to_int8_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        (const unsigned char*)weight,
        (signed char*)transposed_weight,
        (signed char*)packed_transposed_weight,
        n,
        k);

    pack_uint4_to_int8_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        (signed char*) transposed_weight,
        (signed char*) packed_transposed_weight,
        n,
        k);
}


}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace ort_llm
