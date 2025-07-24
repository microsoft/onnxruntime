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

  extern __shared__ int shared_row_indices[];
  shared_row_indices[row + 1] = count;
  __syncthreads();

  // The first thread will calculate the accumulated partial sum of non-zero counts.
  // The result is csr_row_indices stored in shared memory.
  if (row == 0) {
    shared_row_indices[0] = 0;
    for (int i = 1; i < num_rows; i++) {
      shared_row_indices[i + 1] += shared_row_indices[i];
    }

    // The first thread outputs the last element.
    csr_row_indices[num_rows] = shared_row_indices[num_rows];
  }
  __syncthreads();

  // The starting index of current row in csr_col_indices
  int offset = shared_row_indices[row];

  // Output row indices.
  csr_row_indices[row] = offset;

  for (int col = 0; col < num_cols; col++) {
    if (mask[row * num_cols + col] == 1) {
      csr_col_indices[offset] = col;
      offset++;
    }
  }

  // Note that the remaining buffer in csr_col_indices are not filled with dummy value, but it's fine.
  // The last element of csr_row_indices is the total number of non-zero elements.
}

__global__ void MaskToCSR_Large(const int* mask,
                                int* csr_row_indices,
                                int* csr_col_indices,
                                int num_rows,
                                int num_cols,
                                int rows_per_thread  // Each thread handles multiple rows
) {
  extern __shared__ int shared_row_indices[];

  // Update input and output data pointers to the start of current head
  int head = blockIdx.x;
  mask += head * num_rows * num_cols;
  csr_row_indices += head * (num_rows + 1);
  csr_col_indices += head * num_rows * num_cols;

  int tid = threadIdx.x;
  for (int row = tid * rows_per_thread; row < num_rows && row < (tid + 1) * rows_per_thread; row++) {
    int count = 0;
    for (int col = 0; col < num_cols; col++) {
      if (mask[row * num_cols + col] == 1) {
        count++;
      }
    }
    shared_row_indices[row + 1] = count;
  }

  __syncthreads();

  // The first thread will calculate the accumulated partial sum of non-zero counts.
  if (tid == 0) {
    shared_row_indices[0] = 0;
    for (int i = 1; i < num_rows; i++) {
      shared_row_indices[i + 1] += shared_row_indices[i];
    }

    csr_row_indices[num_rows] = shared_row_indices[num_rows];
  }

  __syncthreads();

  for (int row = tid * rows_per_thread; row < num_rows && row < (tid + 1) * rows_per_thread; row++) {
    int offset = shared_row_indices[row];
    csr_row_indices[row] = offset;

    for (int col = 0; col < num_cols; col++) {
      if (mask[row * num_cols + col] == 1) {
        csr_col_indices[offset] = col;
        offset++;
      }
    }
  }
}

void ConvertMaskToCSR(cudaStream_t stream,
                      const int* mask,       // input mask with shape (num_layout, num_rows, num_cols)
                      int num_layout,        // number of layouts
                      int num_rows,          // number of rows
                      int num_cols,          // number of columns
                      int* csr_row_indices,  // output CSR row indices
                      int* csr_col_indices,  // output CSR column indices
                      int max_threads_per_block) {
  if (num_rows <= max_threads_per_block) {
    // Each thread handle one row.
    MaskToCSR<<<num_layout, num_rows, (num_rows + 1) * sizeof(int), stream>>>(
        mask, csr_row_indices, csr_col_indices, num_rows, num_cols);
  } else {
    // Each thread will handle multiple rows when number of rows > max_threads_per_block.
    // For example 128K length with sparse block size 64 will have 2048 rows. Each thread will handle 2 rows.
    int rows_per_thread = (num_rows + max_threads_per_block - 1) / max_threads_per_block;
    MaskToCSR_Large<<<num_layout, max_threads_per_block, (num_rows + 1) * sizeof(int), stream>>>(
        mask, csr_row_indices, csr_col_indices, num_rows, num_cols, rows_per_thread);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
