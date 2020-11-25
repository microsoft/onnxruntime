#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "mixed_precision_scale.h"

namespace onnxruntime {
namespace cuda {

template <typename SrcT, typename DstT>
__global__ void _MixedPrecisionScale(
    const SrcT* input_data,
    const float* scale_data,
    DstT* output_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = static_cast<DstT>(*scale_data * static_cast<float>(input_data[id]));
}

template <typename SrcT, typename DstT>
void Impl_MixedPrecisionScale(
    const SrcT* input_data,
    const float* scale_data,
    DstT* output_data,
    size_t count){
  int blocksPerGrid = CeilDiv(count, GridDim::maxThreadsPerBlock);
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _MixedPrecisionScale<SrcT, DstT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      input_data,
      scale_data,
      output_data,
      N);
}

#define SPECIALIZE_MIXEDPRECISIONSCALE_IMPL(SrcT, DstT) \
template void Impl_MixedPrecisionScale<SrcT, DstT>(     \
    const SrcT* input_data,                             \
    const float* scale_data,                            \
    DstT* output_data,                                  \
    size_t count);

SPECIALIZE_MIXEDPRECISIONSCALE_IMPL(half, half)
SPECIALIZE_MIXEDPRECISIONSCALE_IMPL(half, float)
SPECIALIZE_MIXEDPRECISIONSCALE_IMPL(float, half)
SPECIALIZE_MIXEDPRECISIONSCALE_IMPL(float, float)

}  // namespace cuda
}  // namespace onnxruntime
