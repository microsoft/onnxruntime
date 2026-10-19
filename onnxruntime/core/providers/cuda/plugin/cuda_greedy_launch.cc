// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_greedy.h"

#include <algorithm>

namespace onnxruntime::cuda_plugin {

cudaError_t LaunchGreedy(cudaStream_t stream, const void* input, bool half,
                         size_t count, size_t row_begin, size_t row_end,
                         void* workspace, const GpuGreedyResult*& result) {
  auto blocks = static_cast<unsigned>(std::min(kGreedyMaxBlocks, 1 + (count - 1) / kGreedyThreads));
  auto* output = reinterpret_cast<GpuGreedyResult*>(
      static_cast<char*>(workspace) + sizeof(GpuGreedyResult) * kGreedyMaxBlocks);
  const void* kernel = half ? reinterpret_cast<const void*>(&GreedyPartialsHalf)
                            : reinterpret_cast<const void*>(&GreedyPartialsFloat);
  void* arguments[]{&input, &count, &row_begin, &row_end, &workspace};
  auto status = cudaLaunchKernel(kernel, dim3(blocks), dim3(kGreedyThreads), arguments, 0, stream);
  if (status != cudaSuccess) return status;
  void* finish_arguments[]{&workspace, &blocks, &output};
  status = cudaLaunchKernel(reinterpret_cast<const void*>(&GreedyFinish),
                            dim3(1), dim3(kGreedyThreads), finish_arguments, 0, stream);
  result = output;
  return status;
}

}  // namespace onnxruntime::cuda_plugin
