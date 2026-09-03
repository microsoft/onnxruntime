// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

// Returns true when the shape/layout is eligible for LaunchSmallNGemv.
bool CanUseSmallNGemv(int64_t m, int64_t n, int64_t k, const void* a, const void* b, const void* c);

// Number of K-slices the launch will use, and the derived scratch sizes the
// caller has to provide to LaunchSmallNGemv.
int SmallNGemvSplitK(int n, int k);
size_t SmallNGemvWorkspaceElements(int m, int n, int k);
size_t SmallNGemvCounterElements(int n);

// C[m, n] = sum_k A[m, k] * B[k, n], all row-major half, no transpose. M up to 64 is
// processed in ordered 8-row chunks so every row keeps the same reduction geometry.
// Targets decode-time projections whose N is far too small to keep a cuBLAS
// tile kernel busy (router 2048x256, linear-attention gates 2048x32,
// shared-expert gate 2048x1). The K axis is split across `SmallNGemvSplitK`
// blocks so the read of B spreads over many SMs; the last block to finish a
// column tile reduces the fp32 partials in slice order, so the result is
// deterministic.
//
// `workspace` must hold SmallNGemvWorkspaceElements() floats. `counter` must
// hold SmallNGemvCounterElements() unsigned ints; the launcher clears it before
// each 8-row chunk. Both buffers are exclusive to the launch.
Status LaunchSmallNGemv(cudaStream_t stream, const half* a, const half* b, half* c,
                        int m, int n, int k, float* workspace, unsigned int* counter);

}  // namespace cuda
}  // namespace onnxruntime
