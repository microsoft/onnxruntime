
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors_impl.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/common/safeint.h"

namespace onnxruntime::llm {
namespace kernels {
namespace weight_only {

/**
 * @brief CUDA kernel to permute rows of a quantized tensor.
 *
 * This kernel reorders rows based on a permutation map. It's designed to work
 * for tensors where elements are packed into 32-bit unsigned integers.
 * Each CUDA block handles a "tile group" of rows to be permuted.
 * - blockIdx.x: Index of the row tile group.
 * - threadIdx.y: Index of the row within the current permutation tile (0 to permutation_tile_size - 1).
 * - threadIdx.x: Cooperatively processes the columns of uint32_t vectors.
 *
 * @param permuted_quantized_tensor Output buffer for the permuted tensor (device memory).
 * @param quantized_tensor Input quantized tensor (device memory).
 * @param num_matrix_rows Total number of rows in the input matrix (K dimension).
 * @param num_matrix_cols Total number of columns (element-wise) in the input matrix (N dimension).
 * @param row_permutation Device pointer to the permutation map array.
 * @param permutation_tile_size The size of the permutation tile (e.g., 16 for W8, 32 for W4).
 * @param num_experts Number of experts (for MoE models, typically 1 if not MoE).
 * @param bits_per_element The number of bits per quantized element (e.g., 4 or 8).
 */
__global__ void permute_rows_kernel(
    uint32_t* permuted_quantized_tensor,
    const uint32_t* quantized_tensor,
    int num_matrix_rows,  // K
    int num_matrix_cols,  // N (element-wise)
    const int* row_permutation,
    int permutation_tile_size,
    int num_experts,
    int bits_per_element) {
  const int elts_in_int32 = 32 / bits_per_element;
  const int num_vec_cols = num_matrix_cols / elts_in_int32;  // Number of uint32_t columns

  // Determine the expert index and the row tile group index within that expert
  int expert_idx = 0;
  int block_idx_val = blockIdx.x;

  if (num_experts > 1) {
    int num_row_tile_groups_per_expert = (num_matrix_rows + permutation_tile_size - 1) / permutation_tile_size;
    expert_idx = blockIdx.x / num_row_tile_groups_per_expert;
    block_idx_val = blockIdx.x % num_row_tile_groups_per_expert;
  }

  if (expert_idx >= num_experts) return;

  // Base row for this block's tile group
  int base_row = block_idx_val * permutation_tile_size;
  if (base_row >= num_matrix_rows) return;

  // Row within the current permutation tile (handled by threadIdx.y)
  int tile_row_idx = threadIdx.y;  // From 0 to blockDim.y - 1

  if (tile_row_idx >= permutation_tile_size) return;

  int write_row = base_row + tile_row_idx;
  if (write_row >= num_matrix_rows) return;  // Boundary check for the last tile

  // Determine the source row within the tile using the permutation map
  // `row_permutation` maps the destination `tile_row_idx` to the source row index *within the tile*.
  int permuted_source_tile_row = row_permutation[tile_row_idx];
  int read_row = base_row + permuted_source_tile_row;

  // Offset for the current expert
  int64_t expert_matrix_offset_vec = (int64_t)expert_idx * num_matrix_rows * num_vec_cols;

  // Threads in blockDim.x iterate over the vector columns
  for (int current_vec_col = threadIdx.x; current_vec_col < num_vec_cols; current_vec_col += blockDim.x) {
    int64_t read_offset_vec = expert_matrix_offset_vec + (int64_t)read_row * num_vec_cols + current_vec_col;
    int64_t write_offset_vec = expert_matrix_offset_vec + (int64_t)write_row * num_vec_cols + current_vec_col;

    permuted_quantized_tensor[write_offset_vec] = quantized_tensor[read_offset_vec];
  }
}

/**
 * @brief Performs row permutation for B matrix in mixed GEMM on the GPU.
 *
 * @param permuted_quantized_tensor_out GPU buffer for the output permuted tensor.
 * @param quantized_tensor_in GPU buffer for the input tensor.
 * @param shape Shape of the input tensor. Can be 2D {K, N} or 3D {num_experts, K, N}.
 * K is rows, N is columns (element-wise).
 * @param quant_type The quantization type (W4_A16 or W8_A16).
 * @param stream CUDA stream for the operation.
 */
void permute_B_rows_on_gpu(
    int8_t* permuted_quantized_tensor_out,
    const int8_t* quantized_tensor_in,
    const int* d_row_permutation,
    const int permutation_tile_size,
    const std::vector<size_t>& shape,
    QuantType quant_type,
    cudaStream_t stream) {
  ORT_ENFORCE(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");

  const int num_experts = shape.size() == 3 ? static_cast<int>(shape[0]) : 1;
  const int k_dim = static_cast<int>(shape.size() == 3 ? shape[1] : shape[0]);  // Rows to permute
  const int n_dim = static_cast<int>(shape.size() == 3 ? shape[2] : shape[1]);  // Columns (element-wise)

  // Kernel launch configuration
  // threadIdx.y will correspond to the row index within the permutation tile.
  // blockDim.y should be permutation_tile_size.
  // threadIdx.x will cooperatively process vector columns.

  int threads_per_block_y = permutation_tile_size;  // 16 or 32, depending on the quantization type
  int threads_per_block_x = 32;                     // Tunable: number of vector columns processed by a warp/blockDim.x threads

  dim3 blockDim(threads_per_block_x, threads_per_block_y, 1);

  int num_row_tile_groups_per_expert = (k_dim + permutation_tile_size - 1) / permutation_tile_size;
  int total_row_tile_groups = num_experts * num_row_tile_groups_per_expert;
  dim3 gridDim(total_row_tile_groups, 1, 1);

  int bits_per_element = get_weight_quant_bits(quant_type);
  permute_rows_kernel<<<gridDim, blockDim, 0, stream>>>(
      reinterpret_cast<uint32_t*>(permuted_quantized_tensor_out),
      reinterpret_cast<const uint32_t*>(quantized_tensor_in),
      k_dim,
      n_dim,
      d_row_permutation,
      permutation_tile_size,
      num_experts,
      bits_per_element);
}

// Constants for the subbyte_transpose_kernel
constexpr int SUBBYTE_TRANSPOSE_TILE_DIM_ELTS = 32;  // Tile dimension in elements (e.g., 32 elements wide and 32 elements high)
constexpr int SUBBYTE_TRANSPOSE_BLOCK_ROWS = 8;      // Affects how many rows of a tile a threadblock y-dimension loads/stores in one pass

template <int BITS_PER_ELT>
__global__ void subbyte_transpose_kernel(int8_t* output, const int8_t* input, int num_rows_in, int num_cols_in) {
  static_assert(BITS_PER_ELT == 8 || BITS_PER_ELT == 4, "BITS_PER_ELT must be 8 or 4");

  constexpr int ELTS_PER_BYTE = 8 / BITS_PER_ELT;

  // Shared memory tile dimensions
  // Tile height is fixed in elements. Tile width is fixed in elements, convert to bytes for smem.
  constexpr int SMEM_TILE_HEIGHT_ELTS = SUBBYTE_TRANSPOSE_TILE_DIM_ELTS;
  constexpr int SMEM_TILE_WIDTH_ELTS = SUBBYTE_TRANSPOSE_TILE_DIM_ELTS;
  constexpr int SMEM_TILE_WIDTH_BYTES = SMEM_TILE_WIDTH_ELTS / ELTS_PER_BYTE;

  // Shared memory tile. Padding +1 can sometimes help with bank conflicts.
  __shared__ uint8_t smem_tile[SMEM_TILE_HEIGHT_ELTS][SMEM_TILE_WIDTH_BYTES];

  // Thread indices
  int tx_smem_byte_col = threadIdx.x;      // Thread's x-index corresponds to byte column in shared memory
  int ty_smem_elt_row_base = threadIdx.y;  // Thread's y-index corresponds to base row in shared memory tile part for this thread

  // Starting global coordinates for the input tile this block is processing
  int block_input_start_col_byte = blockIdx.x * SMEM_TILE_WIDTH_BYTES;
  int block_input_start_row_elt = blockIdx.y * SMEM_TILE_HEIGHT_ELTS;

  // Load data from global input to shared memory tile
  for (int i = 0; i < SMEM_TILE_HEIGHT_ELTS; i += SUBBYTE_TRANSPOSE_BLOCK_ROWS) {
    int current_smem_row_elt = ty_smem_elt_row_base + i;  // Actual row in shared memory this thread instance writes to

    if (current_smem_row_elt < SMEM_TILE_HEIGHT_ELTS) {  // Check smem row bound
      int gmem_load_row_elt = block_input_start_row_elt + current_smem_row_elt;
      int gmem_load_col_byte = block_input_start_col_byte + tx_smem_byte_col;

      // Boundary checks for global memory read from input
      if (gmem_load_row_elt < num_rows_in && gmem_load_col_byte < (num_cols_in / ELTS_PER_BYTE)) {
        smem_tile[current_smem_row_elt][tx_smem_byte_col] =
            input[gmem_load_row_elt * (num_cols_in / ELTS_PER_BYTE) + gmem_load_col_byte];
      } else {
        // Pad with a known value (e.g., 0) if reading out of bounds of the input matrix.
        // This is important if the matrix dimensions are not multiples of tile dimensions.
        smem_tile[current_smem_row_elt][tx_smem_byte_col] = 0;
      }
    }
  }

  __syncthreads();  // Ensure all data is loaded into shared memory

  // Write data from shared memory tile to global output (transposed)
  // Output matrix dimensions: num_cols_in rows, num_rows_in columns (element-wise)
  // Output matrix byte columns: (num_rows_in / ELTS_PER_BYTE)

  // Starting global coordinates for the output tile this block is processing
  // Transposed: blockIdx.x (input col tiles) -> output row tiles
  //             blockIdx.y (input row tiles) -> output col tiles
  int block_output_start_row_elt = blockIdx.x * SMEM_TILE_WIDTH_ELTS;
  int block_output_start_col_elt = blockIdx.y * SMEM_TILE_HEIGHT_ELTS;

  for (int i = 0; i < SMEM_TILE_HEIGHT_ELTS; i += SUBBYTE_TRANSPOSE_BLOCK_ROWS) {
    int smem_read_row_elt = ty_smem_elt_row_base + i;  // This is the row in smem to read from

    if (smem_read_row_elt < SMEM_TILE_HEIGHT_ELTS) {
      // The byte read from shared memory. Its original position was (smem_read_row_elt, tx_smem_byte_col)
      // in the tile structure.
      uint8_t source_byte_from_smem = smem_tile[smem_read_row_elt][tx_smem_byte_col];

      // This byte contains ELTS_PER_BYTE elements. Iterate through them.
      for (int k = 0; k < ELTS_PER_BYTE; ++k) {  // k is the element index within the source_byte_from_smem (0 or 1 for 4-bit)
        // Transposed mapping:
        // Original element in tile: (row=smem_read_row_elt, col_element_in_tile=tx_smem_byte_col*ELTS_PER_BYTE + k)
        // Transposed element in tile: (row=tx_smem_byte_col*ELTS_PER_BYTE + k, col=smem_read_row_elt)

        int gmem_write_row_elt = block_output_start_row_elt + (tx_smem_byte_col * ELTS_PER_BYTE + k);
        int gmem_write_col_elt = block_output_start_col_elt + smem_read_row_elt;

        // Boundary check for global memory write to output
        if (gmem_write_row_elt < num_cols_in && gmem_write_col_elt < num_rows_in) {
          if constexpr (BITS_PER_ELT == 8) {
            // Direct byte write for 8-bit elements.
            // Output has num_cols_in rows. Output byte stride is num_rows_in bytes.
            output[gmem_write_row_elt * num_rows_in + gmem_write_col_elt] = source_byte_from_smem;
          } else if constexpr (BITS_PER_ELT == 4) {
            uint8_t nibble = (source_byte_from_smem >> (k * 4)) & 0x0F;

            // Calculate precise byte and nibble index in the output byte
            int output_matrix_num_byte_cols = num_rows_in / ELTS_PER_BYTE;
            int gmem_dest_col_byte = gmem_write_col_elt / ELTS_PER_BYTE;
            int gmem_dest_col_nibble_idx = gmem_write_col_elt % ELTS_PER_BYTE;

            int8_t* p_target_byte = &output[gmem_write_row_elt * output_matrix_num_byte_cols + gmem_dest_col_byte];

            // Ensure atomicOr operates on a 4-byte aligned address.
            uintptr_t addr_val = (uintptr_t)p_target_byte;
            uint32_t* p_aligned_word = (uint32_t*)(addr_val & ~3ULL);  // Align address down to nearest 4-byte boundary
            uint32_t byte_offset_in_word = addr_val & 3ULL;            // Find byte's offset within this 4-byte word (0,1,2,3)

            // Calculate the shift for the nibble within the 4-byte aligned word
            uint32_t shift_in_aligned_word = (byte_offset_in_word * 8) + (gmem_dest_col_nibble_idx * 4);
            uint32_t value_to_or = ((uint32_t)nibble) << shift_in_aligned_word;

            atomicOr(p_aligned_word, value_to_or);
          }
        }
      }
    }
  }
}

void subbyte_transpose_cuda(
    int8_t* transposed_quantized_tensor_out,  // Output buffer
    const int8_t* quantized_tensor_in,        // Input buffer
    const std::vector<size_t>& shape,         // Shape of the input tensor {num_rows_in, num_cols_in}
    QuantType quant_type,
    cudaStream_t stream) {
  ORT_ENFORCE(shape.size() == 2, "CUDA subbyte_transpose currently only supports 2D shapes for simplicity in this example.");
  const int num_rows_in = static_cast<int>(shape[0]);  // K
  const int num_cols_in = static_cast<int>(shape[1]);  // N (element-wise columns)

  const int BITS_PER_ELT = get_weight_quant_bits(quant_type);
  const int ELTS_PER_BYTE_HOST = 8 / BITS_PER_ELT;

  // blockDim.x should correspond to the width of the shared memory tile IN BYTES.
  const int SMEM_TILE_WIDTH_BYTES_CONST = SUBBYTE_TRANSPOSE_TILE_DIM_ELTS / ELTS_PER_BYTE_HOST;
  dim3 blockDim(SMEM_TILE_WIDTH_BYTES_CONST, SUBBYTE_TRANSPOSE_BLOCK_ROWS);

  // Grid dimensions are based on how many tiles are needed to cover the input matrix
  dim3 gridDim(
      // Number of tiles needed for input columns (in bytes)
      ((num_cols_in / ELTS_PER_BYTE_HOST) + SMEM_TILE_WIDTH_BYTES_CONST - 1) / SMEM_TILE_WIDTH_BYTES_CONST,
      // Number of tiles needed for input rows (in elements)
      (num_rows_in + SUBBYTE_TRANSPOSE_TILE_DIM_ELTS - 1) / SUBBYTE_TRANSPOSE_TILE_DIM_ELTS);

  // IMPORTANT: For atomicOr to work correctly by combining nibbles,
  // the output buffer must be zero-initialized before launching the kernel.
  if (BITS_PER_ELT == 4) {
    size_t output_num_bytes = static_cast<size_t>(num_cols_in) * num_rows_in * BITS_PER_ELT / 8;
    cudaMemsetAsync(transposed_quantized_tensor_out, 0, output_num_bytes, stream);
  }

  if (BITS_PER_ELT == 4) {
    subbyte_transpose_kernel<4><<<gridDim, blockDim, 0, stream>>>(
        transposed_quantized_tensor_out, quantized_tensor_in, num_rows_in, num_cols_in);
  } else if (BITS_PER_ELT == 8) {
    subbyte_transpose_kernel<8><<<gridDim, blockDim, 0, stream>>>(
        transposed_quantized_tensor_out, quantized_tensor_in, num_rows_in, num_cols_in);
  } else {
    ORT_THROW("Invalid quant_type for CUDA subbyte_transpose.");
  }
}

/**
 * @brief CUDA kernel to interleave a column-major tensor.
 *
 * This kernel rearranges the elements of a tensor from a standard column-major
 * layout to an interleaved layout, as required for certain optimized GEMM operations.
 * Each thread processes a single 32-bit element.
 *
 * @param interleaved_quantized_tensor The output buffer for the interleaved tensor.
 * @param quantized_tensor The input column-major tensor.
 * @param num_rows The number of rows in the tensor.
 * @param num_cols The number of columns in the tensor.
 * @param num_experts The number of experts (for Mixture-of-Experts models).
 * @param rows_per_tile The number of rows in a tile.
 * @param columns_interleaved The number of columns to interleave.
 * @param quant_type The quantization type of the weight.
 */
__global__ void interleave_column_major_tensor_kernel(
    uint32_t* interleaved_quantized_tensor,
    const uint32_t* quantized_tensor,
    int num_rows,
    int num_cols,
    int num_experts,
    int rows_per_tile,
    int columns_interleaved,
    QuantType quant_type) {
  const int BITS_PER_ELT = get_weight_quant_bits(quant_type);
  const int elts_in_int32 = 32 / BITS_PER_ELT;
  const int num_vec_rows = num_rows / elts_in_int32;
  const int vec_rows_per_tile = rows_per_tile / elts_in_int32;

  // Each thread handles one 32-bit element
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_experts * num_vec_rows * num_cols) {
    // Deconstruct the flat index to get expert, column, and row
    const int expert = idx / (num_vec_rows * num_cols);
    const int col_row_idx = idx % (num_vec_rows * num_cols);
    const int read_col = col_row_idx / num_vec_rows;
    const int vec_read_row = col_row_idx % num_vec_rows;

    const int write_col = read_col / columns_interleaved;
    const int base_vec_row = (vec_read_row / vec_rows_per_tile) * vec_rows_per_tile;

    const int vec_write_row = columns_interleaved * base_vec_row +
                              vec_rows_per_tile * (read_col % columns_interleaved) +
                              (vec_read_row % vec_rows_per_tile);

    const int64_t matrix_offset = (int64_t)expert * num_vec_rows * num_cols;
    const int64_t read_offset = matrix_offset + (int64_t)read_col * num_vec_rows + vec_read_row;
    const int64_t write_offset = matrix_offset + (int64_t)write_col * num_vec_rows * columns_interleaved + vec_write_row;

    interleaved_quantized_tensor[write_offset] = quantized_tensor[read_offset];
  }
}

/**
 * @brief Launches the CUDA kernel for column-major tensor interleaving.
 *
 * @param interleaved_quantized_tensor The output buffer on the GPU.
 * @param quantized_tensor The input tensor on the GPU.
 * @param shape The shape of the tensor.
 * @param quant_type The quantization type.
 * @param details The layout details.
 * @param stream The CUDA stream for the operation.
 */
void interleave_column_major_tensor_cuda(
    int8_t* interleaved_quantized_tensor,
    const int8_t* quantized_tensor,
    const std::vector<size_t>& shape,
    QuantType quant_type,
    const LayoutDetails& details,
    cudaStream_t stream) {
  const int num_experts = shape.size() == 2 ? 1 : static_cast<int>(shape[0]);
  const int num_rows = static_cast<int>(shape.size() == 2 ? shape[0] : shape[1]);
  const int num_cols = static_cast<int>(shape.size() == 2 ? shape[1] : shape[2]);

  const int BITS_PER_ELT = get_weight_quant_bits(quant_type);
  const int elts_in_int32 = 32 / BITS_PER_ELT;

  const int total_elements_32bit = SafeInt<int32_t>(num_experts) * (num_rows / elts_in_int32) * num_cols;
  const int threads_per_block = 256;
  const int num_blocks = (total_elements_32bit + threads_per_block - 1) / threads_per_block;

  interleave_column_major_tensor_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      reinterpret_cast<uint32_t*>(interleaved_quantized_tensor),
      reinterpret_cast<const uint32_t*>(quantized_tensor),
      num_rows,
      num_cols,
      num_experts,
      details.rows_per_column_tile,
      details.columns_interleaved,
      quant_type);
}

/**
 * @brief CUDA kernel to add bias and interleave an INT8 tensor in place.
 *
 * Each thread handles a 32-bit segment (4 int8 elements).
 * 1. Adds a bias of 128 to each int8 element.
 * 2. Swaps the middle two elements to match the required register layout.
 * [elt_0, elt_1, elt_2, elt_3] -> [elt_0, elt_2, elt_1, elt_3]
 *
 * @param tensor The int8 tensor to be modified in place, treated as uint32_t*.
 * @param num_elts The total number of int8 elements in the tensor.
 */
__global__ void add_bias_and_interleave_int8s_inplace_kernel(uint32_t* tensor, size_t num_elts) {
  // Each thread processes one 32-bit word (4 elements)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t register_idx = static_cast<size_t>(idx);

  if (register_idx < num_elts / 4) {
    uint32_t current_register = tensor[register_idx];

    // Unpack the 4 int8 elements from the 32-bit register
    int8_t elt0 = (current_register >> 0) & 0xFF;
    int8_t elt1 = (current_register >> 8) & 0xFF;
    int8_t elt2 = (current_register >> 16) & 0xFF;
    int8_t elt3 = (current_register >> 24) & 0xFF;

    // Add default bias of 128 to each element
    uint8_t biased_elt0 = static_cast<uint8_t>(elt0 + 128);
    uint8_t biased_elt1 = static_cast<uint8_t>(elt1 + 128);
    uint8_t biased_elt2 = static_cast<uint8_t>(elt2 + 128);
    uint8_t biased_elt3 = static_cast<uint8_t>(elt3 + 128);

    // Interleave by swapping elements 1 and 2
    uint32_t transformed_register = (static_cast<uint32_t>(biased_elt3) << 24) |
                                    (static_cast<uint32_t>(biased_elt1) << 16) |
                                    (static_cast<uint32_t>(biased_elt2) << 8) |
                                    (static_cast<uint32_t>(biased_elt0) << 0);

    tensor[register_idx] = transformed_register;
  }
}

/**
 * @brief CUDA kernel to add bias and interleave an INT4 tensor in place.
 *
 * Each thread handles a 32-bit segment (8 int4 elements).
 * 1. Unpacks 8 int4 elements.
 * 2. Adds a bias of 8 to each element.
 * 3. Repacks them into an interleaved layout to optimize for GEMM operations.
 *    [e7, e6, e5, e4, e3, e2, e1, e0] -> [e7, e5, e3, e1, e6, e4, e2, e0]
 *
 * @param tensor The packed int4 tensor to be modified, treated as uint32_t*.
 * @param num_elts The total number of int4 elements in the tensor.
 */
__global__ void add_bias_and_interleave_int4s_inplace_kernel(uint32_t* tensor, size_t num_elts) {
  // Each thread processes one 32-bit word (8 int4 elements)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t register_idx = static_cast<size_t>(idx);

  // Each register holds 8 int4 elements.
  if (register_idx < num_elts / 8) {
    uint32_t current_register = tensor[register_idx];
    uint32_t transformed_register = 0;

    // This loop processes each source nibble 'i' from the current_register
    // and places its biased version into the correct destination slot in transformed_register.
    for (int i = 0; i < 8; ++i) {  // i = src_idx of nibble (0 to 7, LSB to MSB within current_register)
      // Extract the i-th 4-bit element (raw nibble) from current_register
      uint8_t raw_nibble = (current_register >> (i * 4)) & 0x0F;

      // Sign extend the 4-bit raw_nibble to int8_t.
      // Assumes raw_nibble 0x0..0x7 are positive, 0x8..0xF are negative.
      // e.g., 0xF is +7, 0x8 is 0, 0x7 is -1, 0x0 is -8 (if zero point is 8)
      int8_t signed_nibble = static_cast<int8_t>(raw_nibble);
      if (signed_nibble & 0x08) {  // If sign bit (MSB of nibble, value 8) is set
        signed_nibble |= 0xF0;     // Sign extend to fill upper bits of int8_t
      }

      // After (int8_t)(raw_nibble << 4) >> 4, we get the signed value. Example:
      // raw_nibble=0xF -> (0xF0)>>4 (signed) = -1.
      // raw_nibble=0x8 -> (0x80)>>4 (signed) = -8.
      // raw_nibble=0x7 -> (0x70)>>4 (signed) = 7.
      // raw_nibble=0x0 -> (0x00)>>4 (signed) = 0.
      // This is the signed interpretation where 0..7 are positive, 8..15 are negative.
      int8_t val_for_sign_ext = static_cast<int8_t>(raw_nibble << 4);  // Place nibble in high part of a byte
      signed_nibble = val_for_sign_ext >> 4;                           // Arithmetic shift right to sign extend

      // Add bias (maps signed int4 to unsigned int4 [0, 15])
      uint8_t biased_nibble = static_cast<uint8_t>(signed_nibble + 8);

      // Determine destination index (dest_idx) for this src_idx (i)
      int dest_idx;
      if ((i % 2) == 0) {  // src_idx is even: 0, 2, 4, 6
        dest_idx = i / 2;
      } else {  // src_idx is odd: 1, 3, 5, 7
        dest_idx = (i - 1) / 2 + 4;
      }

      // Place the biased nibble (making sure it's masked to 4 bits) into the transformed_register
      transformed_register |= (static_cast<uint32_t>(biased_nibble & 0x0F) << (dest_idx * 4));
    }
    tensor[register_idx] = transformed_register;
  }
}

/**
 * @brief Launches the CUDA kernel for in-place bias addition and interleaving.
 *
 * This function selects the correct CUDA kernel based on the quantization type
 * and launches it on the provided stream.
 *
 * @param tensor The quantized tensor on the GPU to be modified in place.
 * @param num_elts The total number of elements in the tensor.
 * @param quant_type The quantization type (W8_A16, W4_A16, etc.).
 * @param stream The CUDA stream for the operation.
 */
void add_bias_and_interleave_quantized_tensor_inplace_cuda(
    int8_t* tensor,
    size_t num_elts,
    QuantType quant_type,
    cudaStream_t stream) {
  const int threads_per_block = 256;

  if (quant_type == QuantType::W8_A16) {
    // Each thread handles 4 elements (32 bits)
    const int num_registers = SafeInt<int32_t>(num_elts) / 4;
    const int num_blocks = (num_registers + threads_per_block - 1) / threads_per_block;

    add_bias_and_interleave_int8s_inplace_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<uint32_t*>(tensor),
        num_elts);
  } else if (quant_type == QuantType::W4_A16 || quant_type == QuantType::W4_AFP8) {
    // Each thread handles 8 elements (32 bits)
    const int num_registers = SafeInt<int32_t>(num_elts) / 8;
    const int num_blocks = (num_registers + threads_per_block - 1) / threads_per_block;

    add_bias_and_interleave_int4s_inplace_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<uint32_t*>(tensor),
        num_elts);
  } else {
    ORT_THROW("Invalid quantization type for interleaving.");
  }
}

void preprocess_weights_for_mixed_gemm_cuda(cudaStream_t stream,
                                            int arch,
                                            int8_t* preprocessed_quantized_weight,
                                            int8_t* row_major_quantized_weight,
                                            int32_t* d_permutation_map,
                                            std::vector<size_t> const& shape,
                                            QuantType quant_type) {
  LayoutDetails details = getLayoutDetailsForTransform(quant_type, arch);

  ORT_ENFORCE(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");

  size_t num_elts = 1;
  for (auto const& dim : shape) {
    num_elts *= dim;
  }

  int8_t* src_buf = row_major_quantized_weight;
  int8_t* dst_buf = preprocessed_quantized_weight;

  if (details.uses_imma_ldsm) {
    auto row_permutation = get_permutation_map(quant_type);
    cudaMemcpyAsync(d_permutation_map, row_permutation.data(), row_permutation.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    permute_B_rows_on_gpu(dst_buf, src_buf, d_permutation_map, static_cast<int>(row_permutation.size()), shape, quant_type, stream);
    std::swap(src_buf, dst_buf);
  }

  if (details.layoutB == LayoutDetails::Layout::COLUMN_MAJOR) {
    subbyte_transpose_cuda(dst_buf, src_buf, shape, quant_type, stream);
    std::swap(src_buf, dst_buf);
  }

  if (details.columns_interleaved > 1 && arch != 90) {
    interleave_column_major_tensor_cuda(
        dst_buf,
        src_buf,
        shape,
        quant_type,
        details,
        stream);

    std::swap(src_buf, dst_buf);
  }

  add_bias_and_interleave_quantized_tensor_inplace_cuda(
      src_buf,
      num_elts,
      quant_type,
      stream);

  if (preprocessed_quantized_weight != src_buf) {
    const size_t num_bytes = num_elts * static_cast<size_t>(get_weight_quant_bits(quant_type)) / static_cast<size_t>(8);
    CUDA_CALL_THROW(cudaMemcpyAsync(preprocessed_quantized_weight, src_buf, num_bytes, cudaMemcpyDeviceToDevice, stream));
  }

  // Synchronize the stream to ensure the permutation is complete before row_permutation memory is relased.
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
}

}  // namespace weight_only
}  // namespace kernels
}  // namespace onnxruntime::llm
