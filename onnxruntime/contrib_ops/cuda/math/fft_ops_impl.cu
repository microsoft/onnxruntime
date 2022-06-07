/*All contributions by Facebook :
Copyright(c) 2016 Facebook Inc.
==============================================================================*/
/* Modifications Copyright (c) Microsoft. */

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

#include "fft_ops_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
__global__ void _Normalize(
    T* data,
    const int64_t N,
    const int scale) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N)

  int index = static_cast<int>(id);
  data[index] = data[index] / static_cast<T>(scale);
}

template <typename T>
void PostProcess(cudaStream_t stream, const std::vector<int64_t>& signal_dims, int64_t N, T* output_data) {
  int64_t scale = std::accumulate(signal_dims.begin(), signal_dims.end(), 1ll, std::multiplies<int64_t>());
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _Normalize<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(output_data, N, static_cast<int>(scale));
}

#define SPECIALIZED_IMPL(T) \
  template void PostProcess<T>(cudaStream_t stream, const std::vector<int64_t>& signal_dims, int64_t N, T* output_data);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
