// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace onnxruntime::cuda_plugin {

struct GpuGreedyResult {
  int64_t token;
  uint32_t has_nan;
  uint32_t reserved;
};
static_assert(sizeof(GpuGreedyResult) == 16);

constexpr size_t kGreedyThreads = 256;
constexpr size_t kGreedyMaxBlocks = 256;
constexpr size_t kGreedyWorkspaceBytes = sizeof(GpuGreedyResult) * (kGreedyMaxBlocks + 1);

#if !defined(__CUDACC__)
// These signatures match the kernel symbols registered by the separate NVCC translation unit.
void GreedyPartialsFloat(const void* input, size_t count, size_t row_begin, size_t row_end, void* workspace);
void GreedyPartialsHalf(const void* input, size_t count, size_t row_begin, size_t row_end, void* workspace);
void GreedyFinish(const void* workspace, unsigned blocks, GpuGreedyResult* output);
#endif

cudaError_t LaunchGreedy(cudaStream_t stream, const void* input, bool half,
                         size_t count, size_t row_begin, size_t row_end,
                         void* workspace, const GpuGreedyResult*& result);

}  // namespace onnxruntime::cuda_plugin
