#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/gather_nd_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _GatherNDKernel(
    const size_t N,  //The number of copies
    const T* input_data,
    T* output_data,
    const size_t nums_of_elements,
    const int64_t* element_offsets) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N * nums_of_elements, NumElementsPerThread)

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N * nums_of_elements) {
      uint64_t element_offset = element_offsets[id / nums_of_elements];
      output_data[id] = input_data[element_offset + id % nums_of_elements];
      id += NumThreadsPerBlock;
    }
  }
};

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _GatherNDGradKernel(
    const size_t N,  //The number of copies
    const T* input_data,
    T* output_data,
    const size_t nums_of_elements,
    const int64_t* element_offsets) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N * nums_of_elements, NumElementsPerThread)

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N * nums_of_elements) {
      uint64_t element_offset = element_offsets[id / nums_of_elements];
      size_t j = id % nums_of_elements;
      atomicAdd(output_data + element_offset + j, input_data[id]);
      id += NumThreadsPerBlock;
    }
  }
};

template <typename T>
void GatherNDImpl(
    const size_t N,  //The number of copies
    const void* input_data,
    void* output_data,
    const size_t nums_of_elements,
    const int64_t* element_offset) {
  int blocksPerGrid = static_cast<int>(CeilDiv(N * nums_of_elements, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  _GatherNDKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
    <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      N, static_cast<const T*>(input_data), static_cast<T*>(output_data), nums_of_elements, element_offset);
}

template <typename T>
void GatherNDGradImpl(
    const size_t N,  //The number of copies
    const void* input_data,
    void* output_data,
    const size_t nums_of_elements,
    const int64_t* element_offset) {
  int blocksPerGrid = static_cast<int>(CeilDiv(N * nums_of_elements, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  _GatherNDGradKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
    <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
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
