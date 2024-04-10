// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/sparse/block_mask.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Expand block mask from (num_heads, max_blocks, max_blocks) to (num_heads, max_blocks * splits, max_blocks * splits),
__global__ void ExpandMask(int* expanded_mask, const int* mask, int max_blocks, int splits, bool causal) {
  if (threadIdx.x >= max_blocks * splits) {
    return;
  }

  int expanded_col = threadIdx.x;
  int expanded_row = blockIdx.x;
  int h = blockIdx.y;

  // Get mask value from the original mask.
  const int row = expanded_row / splits;
  const int col = expanded_col / splits;
  int value = mask[row * max_blocks + col];

  // Apply causal constraint.
  if (causal && expanded_col > expanded_row) {
    value = 0;
  }

  int offset = h * (max_blocks * splits) * (max_blocks * splits) + expanded_row * (max_blocks * splits) + expanded_col;
  expanded_mask[offset] = value;
}

void ExpandBlockMask(cudaStream_t stream, int* expanded_mask, const int* mask,
                     int num_heads, int max_blocks, int splits, bool causal, int max_threads_per_block) {
  // Each block handle one row. For example, max_blocks=64, splits=2, then each block handle 128 elements.
  int num_cols = max_blocks * splits;
  int threads_per_block = (num_cols + 31) / 32 * 32;

  // Each thread handle one row. The kernel assumes that all rows can be handled in one block.
  if (threads_per_block > max_threads_per_block) {
    ORT_THROW("Threads per block is too large: max_blocks=", max_blocks, ", splits=", splits,
              ", max_threads_per_block=", max_threads_per_block);
  }

  const int num_rows = max_blocks * splits;
  dim3 gridSize(num_rows, num_heads, 1);
  ExpandMask<<<gridSize, threads_per_block, 0, stream>>>(expanded_mask, mask, max_blocks, splits, causal);
}

__global__ void MaskToCSR(const int* mask, int* csr_row_indices, int* csr_col_indices, int num_rows, int num_cols) {
  int row = threadIdx.x;
  if (row >= num_rows) {
    return;
  }

  // Update input and output data pointers to the start of current head
  int head = blockIdx.x;
  mask += head * num_rows * num_cols;
  csr_row_indices += head * (num_rows + 1);
  csr_col_indices += head * num_rows * num_cols;

  int count = 0;
  for (int col = 0; col < num_cols; col++) {
    if (mask[row * num_cols + col] == 1) {
      count++;
    }
  }

  extern __shared__ int non_zero_counts[];
  non_zero_counts[threadIdx.x] = count;
  __syncthreads();

  // The first thread will calculate the accumulated partial sum of non-zero counts.
  if (row == 0) {
    for (int i = 1; i < row; i++) {
      non_zero_counts[i] += non_zero_counts[i - 1];
    }
  }
  __syncthreads();

  // The starting index of current row in csr_col_indices
  int offset = (row == 0) ? 0 : non_zero_counts[row - 1];

  // Output row indices.
  csr_row_indices[row] = offset;
  if (row == 0) {
    // The first thread output the last element.
    csr_row_indices[num_rows] = non_zero_counts[num_rows - 1];
  }

  for (int col = 0; col < num_cols; col++) {
    if (mask[row * num_cols + col] == 1) {
      csr_col_indices[offset] = col;
      offset++;
    }
  }

  // Note that the remaining buffer in csr_col_indices are not filled with dummy value, but it's fine.
  // The last element of csr_row_indices is the total number of non-zero elements.
}

void ConvertMaskToCSR(cudaStream_t stream,
                      const int* mask,       // mask with shape (num_rows, num_cols)
                      int num_heads,         // number of heads
                      int num_rows,          // number of rows of block_mask
                      int num_cols,          // number of cols of block_mask
                      int* csr_row_indices,  // output CSR row indices
                      int* csr_col_indices,  // output CSR col indices
                      int max_threads_per_block) {
  int threads_per_block = (num_rows + 31) / 32 * 32;

  // Each thread handle one row. The kernel assumes that all rows of one head can be handled in one block.
  if (threads_per_block > max_threads_per_block) {
    ORT_THROW("num_rows is too large: num_rows=", num_rows, ", max_threads_per_block=", max_threads_per_block);
  }

  MaskToCSR<<<num_heads, threads_per_block, threads_per_block * sizeof(int), stream>>>(
      mask, csr_row_indices, csr_col_indices, num_rows, num_cols);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
