#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename TSrc>
__global__ void _IsFinite(const TSrc* input, bool* output, CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output[id] = isfinite(input[id]);
}

template<>
__global__ void _IsFinite(const half* input, bool* output, CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  output[id] = !__hisinf(input[id]) && !__hisnan(input[id]);
#else
  output[id] = isfinite(float(input[id]));
#endif
}

template <typename TSrc>
void IsFinite(const TSrc* input, bool* output, size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _IsFinite<<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(input, output, N);
}

#define SPECIALIZE_ISFINITE_IMPL(T) \
template void IsFinite(const T* input, bool* output, size_t count);

SPECIALIZE_ISFINITE_IMPL(half)
SPECIALIZE_ISFINITE_IMPL(float)
SPECIALIZE_ISFINITE_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime