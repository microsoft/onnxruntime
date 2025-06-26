// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/fpA_intB_gemm_adaptor.h"
#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

template <typename T>
__global__ void transposeScaleKernel(
    const T* scale,
    T* transposed_scale,
    int n, int k_blocks) {
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
    T scale_val = scale[input_offset];
    transposed_scale[output_offset] = scale_val;
  }
}

template <typename T>
void launch_transpose_scale_kernel(
    cudaStream_t stream,
    const T* scale,
    T* transposed_scale,
    int n, int k_blocks) {
  constexpr int BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
      (n + blockDim.x - 1) / blockDim.x,        // Grid size in x covers output columns (n)
      (k_blocks + blockDim.y - 1) / blockDim.y  // Grid size in y covers output rows (k_blocks)
  );

  transposeScaleKernel<T><<<gridDim, blockDim, 0, stream>>>(
      scale,
      transposed_scale,
      n,
      k_blocks);
}

// CUDA kernel to compute -scale * zero_point and transpose
// Each thread computes one element of the OUTPUT matrix (shape [k_blocks, n])
template <bool is_zero_point_int4_packed, typename T, typename Z>
__global__ void computeScaledZeroPointAndTransposeKernel(
    const Z* zero_point,        // Input zero_point matrix [n, k_blocks]  or [n, (k_blocks + 1) / 2] if packed int4
    const T* transposed_scale,  // transposed scale [k_blocks, n]
    T* scaled_zero_point,       // Output matrix [k_blocks, n]
    int n,                      // Rows of input matrices
    int k_blocks,               // Columns of input matrices
    float default_zero_point) {
  // Calculate the output matrix coordinates [row, col] for this thread
  // The output matrix has dimensions [k_blocks, n]
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check bounds to ensure we are within the output matrix dimensions [k_blocks, n]
  if (out_row < k_blocks && out_col < n) {
    int in_row = out_col;
    int in_col = out_row;
    int64_t output_offset = static_cast<int64_t>(out_row) * n + out_col;

    // Perform the computation: scaled_zero_point[out_row, out_col] = -scale[in_row, in_col] * zero_point[in_row, in_col]
    T scale_val = transposed_scale[output_offset];
    float zero_point_val;
    if (zero_point != nullptr) {
      if constexpr (is_zero_point_int4_packed) {  // zero point is 4 bit, and two elements are packed into one byte.
        int64_t packed_row_size = (k_blocks + 1) / 2;
        int64_t packed_zp_offset = static_cast<int64_t>(in_row) * packed_row_size + in_col / 2;
        uint8_t packed_zp = zero_point[packed_zp_offset];
        zero_point_val = static_cast<float>((in_col & 0x01) ? (packed_zp >> 4) : (packed_zp & 0x0f));
      } else {
        int64_t input_offset = static_cast<int64_t>(in_row) * k_blocks + in_col;
        zero_point_val = static_cast<float>(zero_point[input_offset]);
      }
    } else {
      zero_point_val = default_zero_point;
    }

    float result = static_cast<float>(scale_val) * (-zero_point_val + default_zero_point);
    scaled_zero_point[output_offset] = static_cast<T>(result);
  }
}

template <bool is_zero_point_int4_packed, typename T, typename Z>
void launch_scaled_zero_point_kernel(
    cudaStream_t stream,
    const Z* zero_point,
    const T* transposed_scale,
    T* scaled_zero_point,
    int n, int k_blocks, float default_zero_point) {
  assert(zero_point != nullptr);
  constexpr int BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
      (n + blockDim.x - 1) / blockDim.x,        // Grid size in x covers output columns (n)
      (k_blocks + blockDim.y - 1) / blockDim.y  // Grid size in y covers output rows (k_blocks)
  );

  computeScaledZeroPointAndTransposeKernel<is_zero_point_int4_packed, T, Z><<<gridDim, blockDim, 0, stream>>>(
      zero_point,
      transposed_scale,
      scaled_zero_point,
      n,
      k_blocks,
      default_zero_point);
}

// Explicit instantiations:
template void launch_transpose_scale_kernel<half>(
    cudaStream_t stream,
    const half* scale,
    half* transposed_scale,
    int n, int k_blocks);

template void launch_scaled_zero_point_kernel<false, half, half>(
    cudaStream_t stream,
    const half* zero_point,
    const half* transposed_scale,
    half* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);

template void launch_scaled_zero_point_kernel<false, half, uint8_t>(
    cudaStream_t stream,
    const uint8_t* zero_point,
    const half* transposed_scale,
    half* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);

// zero point is 4 bits packed.
template void launch_scaled_zero_point_kernel<true, half, uint8_t>(
    cudaStream_t stream,
    const uint8_t* zero_point,
    const half* transposed_scale,
    half* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);

template void launch_transpose_scale_kernel<__nv_bfloat16>(
    cudaStream_t stream,
    const __nv_bfloat16* scale,
    __nv_bfloat16* transposed_scale,
    int n, int k_blocks);

template void launch_scaled_zero_point_kernel<false, __nv_bfloat16, __nv_bfloat16>(
    cudaStream_t stream,
    const __nv_bfloat16* zero_point,
    const __nv_bfloat16* transposed_scale,
    __nv_bfloat16* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);

template void launch_scaled_zero_point_kernel<false, __nv_bfloat16, uint8_t>(
    cudaStream_t stream,
    const uint8_t* zero_point,
    const __nv_bfloat16* transposed_scale,
    __nv_bfloat16* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);

// zero point is 4 bits packed.
template void launch_scaled_zero_point_kernel<true, __nv_bfloat16, uint8_t>(
    cudaStream_t stream,
    const uint8_t* zero_point,
    const __nv_bfloat16* transposed_scale,
    __nv_bfloat16* scaled_zero_point,
    int n, int k_blocks, float default_zero_point);


// CUDA kernel to unpack uint4, transpose, and pack into int8 directly
__global__ void unpack_transpose_pack_uint4_to_int8_kernel_v2(
    const unsigned char* __restrict__ packed_weight,
    signed char* __restrict__ packed_transposed_weight,
    int n, // original matrix rows
    int k) // original matrix columns
{
  // The output 'packed_transposed_weight' has dimensions k x (n/2) bytes.
  // Each thread processes one byte in the output.
  int out_flat_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Total number of bytes in the output packed_transposed_weight matrix
  int total_output_bytes = k * (n / 2);

  if (out_flat_idx < total_output_bytes) {
    constexpr signed char default_zero_point = 8;

    // Calculate row and column in the output packed_transposed_weight matrix (k x n/2)
    // out_row_packed: row in the k dimension of the output (0 to k-1)
    // out_col_packed: column in the n/2 dimension of the output (0 to n/2 - 1)
    const int out_row_packed = out_flat_idx / (n / 2);
    const int out_col_packed = out_flat_idx % (n / 2);

    // These two int8 values will form the current output packed byte:
    // val_0: corresponds to original_unpacked[2 * out_col_packed][out_row_packed]
    // val_1: corresponds to original_unpacked[2 * out_col_packed + 1][out_row_packed]

    // --- Retrieve val_0 ---
    // Its original (unpacked) row index was '2 * out_col_packed'
    const int r_orig_0 = 2 * out_col_packed;
    // Its original (unpacked) column index was 'out_row_packed'
    const int c_orig_0 = out_row_packed;

    // Determine the flat index in the input 'packed_weight' (n x k/2) where val_0 resides
    const int packed_weight_idx_0 = r_orig_0 * (k / 2) + c_orig_0 / 2;

    unsigned char packed_data_0 = packed_weight[packed_weight_idx_0];
    signed char val_0;
    if ((c_orig_0 % 2) == 0) { // If original column is even, it's the lower 4 bits
      val_0 = (signed char)(packed_data_0 & 0x0f) - default_zero_point;
    } else { // If original column is odd, it's the upper 4 bits
      val_0 = (signed char)(packed_data_0 >> 4) - default_zero_point;
    }

    // --- Retrieve val_1 ---
    // Its original (unpacked) row index was '2 * out_col_packed + 1'
    const int r_orig_1 = 2 * out_col_packed + 1;
    // Its original (unpacked) column index was 'out_row_packed'
    const int c_orig_1 = out_row_packed;

    // Determine the flat index in the input 'packed_weight' (n x k/2) where val_1 resides
    const int packed_weight_idx_1 = r_orig_1 * (k / 2) + c_orig_1 / 2;

    unsigned char packed_data_1 = packed_weight[packed_weight_idx_1];
    signed char val_1;
    if ((c_orig_1 % 2) == 0) { // If original column is even, it's the lower 4 bits
      val_1 = (signed char)(packed_data_1 & 0x0f) - default_zero_point;
    } else { // If original column is odd, it's the upper 4 bits
      val_1 = (signed char)(packed_data_1 >> 4) - default_zero_point;
    }

    // Pack the two signed char values (now 8-bit, but we only care about their 4 LSBs)
    // back into a single byte for the output.
    packed_transposed_weight[out_flat_idx] = (unsigned char)((val_0 & 0x0f) | ((val_1 & 0x0f) << 4));
  }
}

void unpack_uint4_transposed_to_int8_direct_cuda(
    cudaStream_t stream, void* packed_transposed_weight, const void* packed_weight, int n, int k) {
  int total_output_bytes = k * (n / 2);
  int threads_per_block = 256;
  int num_blocks = (total_output_bytes + threads_per_block - 1) / threads_per_block;

  unpack_transpose_pack_uint4_to_int8_kernel_v2<<<num_blocks, threads_per_block, 0, stream>>>(
      (const unsigned char*)packed_weight,
      (signed char*)packed_transposed_weight,
      n,
      k);
}

__global__ void transpose_uint8_matrix_and_convert_to_int8_kernel(
    const uint8_t* __restrict__ input,  // shape: (n, k)
    int8_t* __restrict__ output,        // shape: (k, n)
    int n, int k) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;  // index in n
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // index in k

  if (row < n && col < k) {
    int input_idx = row * k + col;
    int output_idx = col * n + row;
    output[output_idx] = static_cast<int8_t>(static_cast<int>(input[input_idx]) - 128);
  }
}

void transpose_uint8_matrix_and_convert_to_int8(
    cudaStream_t stream,
    int8_t* output,        // shape: (k, n)
    const uint8_t* input,  // shape: (n, k)
    int n, int k) {

  dim3 blockDim(16, 16);
  dim3 gridDim((k + blockDim.x - 1) / blockDim.x,
               (n + blockDim.y - 1) / blockDim.y);

  transpose_uint8_matrix_and_convert_to_int8_kernel<<<gridDim, blockDim, 0, stream>>>(input, output, n, k);
}


}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
