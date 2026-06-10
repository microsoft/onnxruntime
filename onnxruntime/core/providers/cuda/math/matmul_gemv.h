// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

// Decode-oriented GEMV fast path for MatMul: Y[1, N] = alpha * A[1, K] * B[K, N].
//
// At decode (M == 1) a plain fp16/bf16 MatMul with a constant weight is a memory
// bound GEMV. cuBLAS dispatches a split-K path (a `dot_kernel` followed by a
// separate `reduce_1Block_kernel`) -- two launches for one tiny GEMV. A single
// custom kernel that assigns one thread block per output column, splitting K
// across the block and reducing once, is ~2x faster on small N.
//
// The kernel reads the weight in a transposed [N, K] layout (row `n` contiguous)
// so consecutive threads access consecutive addresses (coalesced). The caller
// produces that layout once via TransposeForGemv during PrePack.

// Transpose a row-major [rows, cols] matrix into [cols, rows] (row-major).
template <typename T>
void TransposeForGemv(cudaStream_t stream, T* dst, const T* src, int rows, int cols);

// Compute Y[n] = alpha * sum_k A[k] * Bt[n, k] for n in [0, N), Bt is [N, K].
// Accumulation is done in fp32 to match cuBLAS's fp32-compute fp16 GEMM.
template <typename T>
void MatMulGemvM1(cudaStream_t stream, T* y, const T* a, const T* b_transposed,
                  int n, int k, float alpha);

}  // namespace cuda
}  // namespace onnxruntime
