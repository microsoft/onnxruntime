// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/math/isfinite.cuh"

namespace onnxruntime {
namespace cuda {

template <typename TSrc>
__global__ void _IsFinite(const TSrc* input, bool* output, CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output[id] = IsFiniteScalar(input[id]);
}

template <typename TSrc>
void IsFinite(cudaStream_t stream, const TSrc* input, bool* output, size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _IsFinite<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input, output, N);
}

#define SPECIALIZE_ISFINITE_IMPL(T) \
  template void IsFinite(cudaStream_t stream, const T* input, bool* output, size_t count);

SPECIALIZE_ISFINITE_IMPL(half)
SPECIALIZE_ISFINITE_IMPL(float)
SPECIALIZE_ISFINITE_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime