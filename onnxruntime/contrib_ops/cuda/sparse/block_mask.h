// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Expand block mask from (num_layout, max_blocks, max_blocks)
//                     to (num_layout, max_blocks * row_splits, max_blocks * col_splits),
// and apply causal constraint if specified. The mask is stored in row-major order.
// For example,
//   mask = [[[1, 0],
//            [0, 1]]]
//   row_splits = 2,  col_splits = 2, causal = true
//   expanded_mask = [[[1, 0, 0, 0],
//                     [1, 1, 0, 0],
//                     [0, 0, 1, 0],
//                     [0, 0, 1, 1]]]
void ExpandBlockMask(cudaStream_t stream,
                     int* expanded_mask,  // output shape (num_layout, max_blocks * row_splits, max_blocks * col_splits)
                     const int* mask,     // input shape (num_layout, max_blocks, max_blocks)
                     int num_layout,
                     int max_blocks,
                     int row_splits,
                     int col_splits,
                     bool causal,
                     int max_threads_per_block);

// Convert mask to compressed sparse row (CSR) format ( https://en.wikipedia.org/wiki/Sparse_matrix)
// For example, num_layout=1, num_rows=4 and num_cols=4, and the mask is like
//      1, 0, 0, 0
//      1, 1, 0, 0
//      0, 1, 1, 0
//      0, 1, 1, 1
// The CSR format is like:
//  csr_col_indices:
//      0,  0, 1,  1, 2,  1, 2, 3,  0*, 0*, 0*, 0*, 0*, 0*, 0*, 0* (* is padding)
//  csr_row_indices:
//      0, 1, 3, 5, 8
void ConvertMaskToCSR(cudaStream_t stream,
                      const int* mask,       // input mask with shape (num_layout, num_rows, num_cols)
                      int num_layout,        // number of layout
                      int num_rows,          // number of rows of block_mask
                      int num_cols,          // number of cols of block_mask
                      int* csr_row_indices,  // output CSR row indices with shape (num_layout, num_rows + 1).
                      int* csr_col_indices,  // output CSR col indices with shape (num_layout, num_rows * num_cols).
                      int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
