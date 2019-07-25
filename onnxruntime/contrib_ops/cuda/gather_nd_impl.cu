#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/gather_nd_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

template <typename Tind>
__global__ void _GatherNDKernel(
    const int64_t* element_index_counts,
    const Tind* indice,  //The address of the indices
    const int64_t last_indice_dimension,
    const size_t N,  //The number of copies
    void* input_data,
    void* output_data,
    const size_t nums_of_elements,
    const size_t element_bytes,
    const int64_t axis_) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N);
  //Compute the element_offset
  uint64_t element_offset = 0;
  int64_t reminder = i;
  for (uint64_t j = 0; j < axis_; ++j) {
    uint64_t index = reminder / element_index_counts[last_indice_dimension + j];
    element_offset += index * element_index_counts[j];
    reminder -= (index * element_index_counts[last_indice_dimension + j]);
  }
  for (uint64_t j = axis_; j < last_indice_dimension; ++j) {
    uint64_t index = *(indice + i * (last_indice_dimension - axis_) + (j - axis_));
    element_offset += index * element_index_counts[j];
  }

  //TODO: Add error check, is there a better way to copy data inide device function?
  int8_t* tgt_addr = ((int8_t*)output_data) + i * nums_of_elements * element_bytes;
  int8_t* src_addr = ((int8_t*)input_data) + element_offset * element_bytes;
  memcpy(tgt_addr, src_addr, nums_of_elements * element_bytes);
};

template <typename T, typename Tind>
__global__ void _GatherNDGradKernel(
    const int64_t* element_index_counts,
    const Tind* indice,  //The address of the indices
    const int64_t last_indice_dimension,
    const size_t N,  //The number of copies
    T* input_data,
    T* output_data,
    const size_t nums_of_elements,
    const int64_t axis_) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(i, N);
  //Compute the element_offset
  uint64_t element_offset = 0;
  int64_t reminder = i;
  for (uint64_t j = 0; j < axis_; ++j) {
    uint64_t index = reminder / element_index_counts[last_indice_dimension + j];
    element_offset += index * element_index_counts[j];
    reminder -= (index * element_index_counts[last_indice_dimension + j]);
  }
  for (uint64_t j = axis_; j < last_indice_dimension; ++j) {
    uint64_t index = *(indice + i * (last_indice_dimension - axis_) + (j - axis_));
    element_offset += index * element_index_counts[j];
  }
  //TODO: This potentially can be done in nums_to_copy threads, needs to store element_offset somewhere
  for (size_t j = 0; j < nums_of_elements; ++j) {
    atomicAdd(output_data + element_offset + j, input_data[i * nums_of_elements + j]);
  }
};

template <typename Tind>
void GatherNDImpl(
    const int64_t* element_index_counts,
    const Tind* indice,  //The address of the indices
    const int64_t last_indice_dimension,
    const size_t N,  //The number of copies
    void* input_data,
    void* output_data,
    const size_t nums_of_elements,
    const size_t element_bytes,
    const int64_t axis_) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _GatherNDKernel<Tind><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      element_index_counts, indice, last_indice_dimension, N, input_data, output_data, nums_of_elements, element_bytes, axis_);
}

template <typename T, typename Tind>
void GatherNDGradImpl(
    const int64_t* element_index_counts,
    const Tind* indice,  //The address of the indices
    const int64_t last_indice_dimension,
    const size_t N,  //The number of copies
    void* input_data,
    void* output_data,
    const size_t nums_of_elements,
    const int64_t axis_) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _GatherNDGradKernel<T, Tind><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      element_index_counts, indice, last_indice_dimension, N, static_cast<T*>(input_data), static_cast<T*>(output_data), nums_of_elements, axis_);
}

#define SPECIALIZED_IMPL(Tind)                                                                                                                                     \
  template void GatherNDImpl<Tind>(const int64_t* element_index_counts, const Tind* indice, const int64_t last_indice_dimension, const size_t N, void* input_data, \
                                   void* output_data, const size_t nums_of_elements, const size_t element_bytes, const int64_t axis_);
#define SPECIALIZED_GRAD_IMPL(T, Tind)                                                                                                                                    \
  template void GatherNDGradImpl<T, Tind>(const int64_t* element_index_counts, const Tind* indice, const int64_t last_indice_dimension, const size_t N, void* input_data, \
                                          void* output_data, const size_t nums_of_elements, const int64_t axis_);

SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_GRAD_IMPL(float, int64_t)
SPECIALIZED_GRAD_IMPL(float, int32_t)
//TODO: Enable following when the GPU architecture supports AtomicAdd with following data types
//SPECIALIZED_GRAD_IMPL(half, int64_t)
//SPECIALIZED_GRAD_IMPL(half, int32_t)
//SPECIALIZED_GRAD_IMPL(double, int64_t)
//SPECIALIZED_GRAD_IMPL(double, int32_t)
}  // namespace cuda
}  // namespace onnxruntime