// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

// Returns true when the shape/layout is eligible for LaunchSmallNGemv.
bool CanUseSmallNGemv(int64_t m, int64_t n, int64_t k, const void* a, const void* b, const void* c);

// True when LaunchSmallNGemv takes the vectorized kernel for these operands.
bool SmallNGemvUsesVectorizedKernel(int n, int k, const void* a, const void* b);

// Number of K-slices the launch will use, and the derived scratch sizes the
// caller has to provide to LaunchSmallNGemv.
int SmallNGemvSplitK(int n, int k);
size_t SmallNGemvWorkspaceElements(int m, int n, int k);
size_t SmallNGemvCounterElements(int n);

// C[m, n] = sum_k A[m, k] * B[k, n], all row-major T (half or nv_bfloat16), no transpose, fp32
// accumulation. M up to 64 is processed in ordered 8-row chunks so every row keeps the same
// reduction geometry.
// Targets decode-time projections whose N is far too small to keep a cuBLAS
// tile kernel busy (router 2048x256, linear-attention gates 2048x32,
// shared-expert gate 2048x1). The K axis is split across `SmallNGemvSplitK`
// blocks so the read of B spreads over many SMs; the last block to finish a
// column tile reduces the fp32 partials in slice order, so the result is
// deterministic. Even N with K % 8 == 0 and 16-byte aligned A takes a vectorized
// kernel (two columns and eight K rows per lane); other shapes take a scalar one.
//
// `workspace` must hold SmallNGemvWorkspaceElements() floats. `counter` must
// hold SmallNGemvCounterElements() unsigned ints; the launcher clears it before
// the first chunk. Both buffers are exclusive to the launch.
template <typename T>
Status LaunchSmallNGemv(cudaStream_t stream, const T* a, const T* b, T* c,
                        int m, int n, int k, float* workspace, unsigned int* counter);

}  // namespace cuda
}  // namespace onnxruntime
