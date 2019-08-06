#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/gather_nd_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _GatherNDKernel(
    const size_t N,  //The number of copies
    const T* input_data,
    T* output_data,
    const size_t nums_of_elements,
    const int64_t* element_offsets) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N * nums_of_elements)
  uint64_t element_offset = element_offsets[i / nums_of_elements];
  output_data[i] = input_data[element_offset + i % nums_of_elements];
};

template <typename T>
__global__ void _GatherNDGradKernel(
    const size_t N,  //The number of copies
    const T* input_data,
    T* output_data,
    const size_t nums_of_elements,
    const int64_t* element_offsets) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N * nums_of_elements);
  uint64_t element_offset = element_offsets[i / nums_of_elements];
  size_t j = i % nums_of_elements;
  atomicAdd(output_data + element_offset + j, input_data[i]);
};

template <typename T>
void GatherNDImpl(
    const size_t N,  //The number of copies
    const void* input_data,
    void* output_data,
    const size_t nums_of_elements,
    const int64_t* element_offset) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _GatherNDKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      N, static_cast<const T*>(input_data), static_cast<T*>(output_data), nums_of_elements, element_offset);
}

template <typename T>
void GatherNDGradImpl(
    const size_t N,  //The number of copies
    const void* input_data,
    void* output_data,
    const size_t nums_of_elements,
    const int64_t* element_offset) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _GatherNDGradKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      N, static_cast<const T*>(input_data), static_cast<T*>(output_data), nums_of_elements, element_offset);
}

#define SPECIALIZED_IMPL(T) \
template void GatherNDImpl<T>(const size_t N, const void* input_data, void* output_data, const size_t nums_of_elements, const int64_t* element_offset);
#define SPECIALIZED_GRAD_IMPL(T) \
template void GatherNDGradImpl<T>(const size_t N, const void* input_data, void* output_data, const size_t nums_of_elements, const int64_t* element_offset);

SPECIALIZED_IMPL(float)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
SPECIALIZED_IMPL(half)
#endif
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
SPECIALIZED_IMPL(double)
#endif

SPECIALIZED_GRAD_IMPL(float)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
SPECIALIZED_GRAD_IMPL(half)
#endif
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
SPECIALIZED_GRAD_IMPL(double)
#endif
}  // namespace cuda
}  // namespace onnxruntime
