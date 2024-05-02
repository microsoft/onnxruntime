// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/sparse/block_mask.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

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
    for (int i = 1; i < num_rows; i++) {
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
                      const int* mask,       // input mask with shape (num_layout, num_rows, num_cols)
                      int num_layout,        // number of layouts
                      int num_rows,          // number of rows
                      int num_cols,          // number of columns
                      int* csr_row_indices,  // output CSR row indices
                      int* csr_col_indices,  // output CSR column indices
                      int max_threads_per_block) {
  int threads_per_block = (num_rows + 31) / 32 * 32;

  // Each thread handle one row. The kernel assumes that all rows of one head can be handled in one block.
  if (threads_per_block > max_threads_per_block) {
    ORT_THROW("num_rows is too large: num_rows=", num_rows, ", max_threads_per_block=", max_threads_per_block);
  }

  MaskToCSR<<<num_layout, threads_per_block, threads_per_block * sizeof(int), stream>>>(
      mask, csr_row_indices, csr_col_indices, num_rows, num_cols);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
