// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_greedy.h"

#include <cuda_fp16.h>
#include <math_constants.h>
#include <cub/block/block_reduce.cuh>

namespace onnxruntime::cuda_plugin {
namespace {

struct Partial {
  float value;
  uint32_t has_nan;
  int64_t token;
};
static_assert(sizeof(Partial) == sizeof(GpuGreedyResult));

__device__ Partial Identity() {
  return {-CUDART_INF_F, 0, 0};
}

struct Merge {
  __device__ Partial operator()(const Partial& a, const Partial& b) const {
    const bool take_b = b.value > a.value || (b.value == a.value && b.token < a.token);
    Partial result = take_b ? b : a;
    result.has_nan = a.has_nan | b.has_nan;
    return result;
  }
};

template <typename T>
__device__ void ScanPartials(const T* input, size_t count, size_t row_begin, size_t row_end,
                             Partial* partials) {
  Partial local = Identity();
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    const float value = static_cast<float>(input[index]);
    const bool nan = isnan(value);
    local.has_nan |= nan;
    if (index >= row_begin && index < row_end && !nan) {
      local = Merge{}(local, {value, 0, static_cast<int64_t>(index - row_begin)});
    }
  }
  using Reduce = cub::BlockReduce<Partial, kGreedyThreads>;
  __shared__ typename Reduce::TempStorage temporary;
  const Partial combined = Reduce(temporary).Reduce(local, Merge{});
  if (threadIdx.x == 0) partials[blockIdx.x] = combined;
}

}  // namespace

__global__ void GreedyPartialsFloat(const void* input, size_t count, size_t row_begin,
                                    size_t row_end, void* workspace) {
  ScanPartials(static_cast<const float*>(input), count, row_begin, row_end, static_cast<Partial*>(workspace));
}

__global__ void GreedyPartialsHalf(const void* input, size_t count, size_t row_begin,
                                   size_t row_end, void* workspace) {
  ScanPartials(static_cast<const __half*>(input), count, row_begin, row_end, static_cast<Partial*>(workspace));
}

__global__ void GreedyFinish(const void* workspace, unsigned blocks, GpuGreedyResult* output) {
  const auto* partials = static_cast<const Partial*>(workspace);
  const Partial local = threadIdx.x < blocks ? partials[threadIdx.x] : Identity();
  using Reduce = cub::BlockReduce<Partial, kGreedyThreads>;
  __shared__ typename Reduce::TempStorage temporary;
  const Partial combined = Reduce(temporary).Reduce(local, Merge{});
  if (threadIdx.x == 0) {
    *output = {combined.token, combined.has_nan, 0};
  }
}

}  // namespace onnxruntime::cuda_plugin
