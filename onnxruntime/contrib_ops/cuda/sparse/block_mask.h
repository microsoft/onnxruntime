// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Expand block mask from (num_heads, max_blocks, max_blocks) to (num_heads, max_blocks * splits, max_blocks * splits),
// and apply causal constraint if specified. The mask is stored in row-major order.
// For example,
//   mask = [[[1, 0],
//            [0, 1]]]
//   splits = 2
//   causal = true
//   expanded_mask = [[[1, 0, 0, 0],
//                     [1, 1, 0, 0],
//                     [0, 0, 1, 0],
//                     [0, 0, 1, 1]]]
void ExpandBlockMask(cudaStream_t stream,
                     int* expanded_mask,  // output shape (num_heads, max_blocks * splits, max_blocks * splits)
                     const int* mask,     // input shape (num_heads, max_blocks, max_blocks)
                     int num_heads,
                     int max_blocks,
                     int splits,
                     bool causal,
                     int max_threads_per_block);

// Convert mask to compressed sparse row (CSR) format ( https://en.wikipedia.org/wiki/Sparse_matrix)
void ConvertMaskToCSR(cudaStream_t stream,
                      const int* mask,       // input mask with shape (num_heads, num_rows, num_cols)
                      int num_heads,         // number of heads
                      int num_rows,          // number of rows of block_mask
                      int num_cols,          // number of cols of block_mask
                      int* csr_row_indices,  // output CSR row indices with shape (num_heads, num_rows + 1).
                      int* csr_col_indices,  // output CSR col indices with shape (num_heads, num_rows * num_cols).
                      int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
