// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/gather_grad_impl.h"

#include <iterator>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {
namespace cuda {

template <typename TInputIterator, typename TOutputIterator>
__global__ void CopyKernel(
    TInputIterator input,
    size_t length,
    TOutputIterator output) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(idx, length);
  output[idx] = input[idx];
}

template <typename T, typename Tin>
__global__ void GatherGradKernel_ThreadPerUniqueIndex(
    const Tin* input,
    const Tin* indices,
    const T* grad_output,
    T* grad_weight,
    int64_t numel,
    int64_t input_numel,
    int64_t param_itrs,
    int64_t stride) {
  int idx = blockIdx.x * 4 + threadIdx.y;

  const int SZ = 4;
  if (idx < numel && (idx == 0 || input[idx] != input[idx - 1])) {
    do {
      for (int itr = 0; itr < param_itrs; ++itr) {
        const int start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        const int weight_row = itr * input_numel + ((int)input[idx]) * stride;  //the offset of the input
        const int grad_row = (itr * numel + ((int)indices[idx])) * stride;      //the offset of the gradient

        float gradient[SZ];
        float weight[SZ];

#pragma unroll
        for (int ii = 0; ii < SZ; ii++) {
          int feature_dim = start_feature + ii * GPU_WARP_SIZE;
          if (feature_dim < stride) {
            gradient[ii] = static_cast<float>(grad_output[grad_row + feature_dim]);
            weight[ii] = static_cast<float>(grad_weight[weight_row + feature_dim]);
          }
        }

#pragma unroll
        for (int ii = 0; ii < SZ; ii++) {
          weight[ii] += gradient[ii];
        }

#pragma unroll
        for (int ii = 0; ii < SZ; ii++) {
          int feature_dim = start_feature + ii * GPU_WARP_SIZE;
          if (feature_dim < stride) {
            grad_weight[weight_row + feature_dim] = static_cast<T>(weight[ii]);
          }
        }
      }
      idx++;
    } while (idx < numel && input[idx] == input[idx - 1]);
  }
}

template <typename T, typename Tin>
void GatherGradImpl_ThreadPerUniqueIndex(
    const CudaScratchBufferAllocator& allocator,
    const T* grad_data,
    const Tin* indices_data,
    const int64_t num_indices,
    const int64_t num_weights,
    const int64_t stride,
    T* output_data,
    const int64_t num_inputs,  //The number of input elements starting from the gathering dimension
    const int64_t param_itrs   //The size of dimensions of the data before gathering dimension
) {
  // allocate intermediate buffers
  auto original_indices = allocator.GetScratchBuffer<Tin>(num_indices);

  // initialize original_indices with [0, num_indices)
  {
    const auto blocks_per_grid = CeilDiv(num_indices, GridDim::maxThreadsPerBlock);
    cub::CountingInputIterator<Tin> counting_input(Tin{});
    CopyKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock>>>(
        counting_input, num_indices, original_indices.get());
  }

  auto indices_data_sorted = allocator.GetScratchBuffer<Tin>(num_indices);
  auto original_indices_sorted = allocator.GetScratchBuffer<Tin>(num_indices);

  // sort indices and original indices
  size_t sort_temp_storage_size_bytes = 0;
  CUDA_CALL_THROW(cub::DeviceRadixSort::SortPairs(
      nullptr, sort_temp_storage_size_bytes,
      indices_data, indices_data_sorted.get(),
      original_indices.get(), original_indices_sorted.get(),
      num_indices));

  auto sort_temp_storage = allocator.GetScratchBuffer<void>(sort_temp_storage_size_bytes);

  CUDA_CALL_THROW(cub::DeviceRadixSort::SortPairs(
      sort_temp_storage.get(), sort_temp_storage_size_bytes,
      indices_data, indices_data_sorted.get(),
      original_indices.get(), original_indices_sorted.get(),
      num_indices));

  dim3 block(GPU_WARP_SIZE, 4);
  dim3 grid(CeilDiv(num_indices, 4), CeilDiv(stride, 128));

  GatherGradKernel_ThreadPerUniqueIndex<<<grid, block>>>(
      indices_data_sorted.get(),
      original_indices_sorted.get(),
      grad_data,
      output_data,
      num_indices,
      num_inputs,
      param_itrs,
      stride);
}

// adapted from https://github.com/NVIDIA/thrust/blob/cccd45ef3b5ec2351a4fc551211fc58fdcefa9fd/examples/strided_range.cu

using IndexType = int64_t;
using IndexCountingIterator = thrust::counting_iterator<IndexType>;
template <typename IndexGeneratorFn>
using IndexTransformIterator = thrust::transform_iterator<
    IndexGeneratorFn, IndexCountingIterator>;

template <typename IndexGeneratorFn>
IndexTransformIterator<IndexGeneratorFn>
MakeIndexTransformIterator(IndexGeneratorFn generator) {
  return IndexTransformIterator<IndexGeneratorFn>{IndexCountingIterator{0}, generator};
}

template <typename ElementIterator, typename IndexGeneratorFn>
class GeneratedPermutation {
 public:
  using PermutationIterator = thrust::permutation_iterator<
      ElementIterator, IndexTransformIterator<IndexGeneratorFn>>;

  GeneratedPermutation(ElementIterator base, IndexGeneratorFn generator)
      : base_{base}, generator_{generator} {
  }

  PermutationIterator begin() {
    return PermutationIterator{
        base_, MakeIndexTransformIterator(generator_)};
  }

 private:
  const ElementIterator base_;
  const IndexGeneratorFn generator_;
};

template <typename ElementIterator, typename IndexGeneratorFn>
GeneratedPermutation<ElementIterator, IndexGeneratorFn> MakeGeneratedPermutation(
    ElementIterator base, IndexGeneratorFn generator) {
  return GeneratedPermutation<ElementIterator, IndexGeneratorFn>(base, generator);
}

template <typename T>
struct Sum {
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

template <typename T>
__global__ void PrintDataKernel(const T* data, size_t length) {
  if (threadIdx.x == 0) {
    printf("PrintData:\n");
    for (size_t i = 0; i < length; ++i) {
      printf("%f ", static_cast<float>(data[i]));
    }
    printf("\n");
  }
}

template <typename T, typename TIndex>
void GatherGradImpl_FancyIterator(
    const CudaScratchBufferAllocator& allocator,
    const T* dY_data,
    const TIndex* dX_indices,
    const int64_t num_indices,
    const int64_t /*num_weights*/,
    const int64_t num_gathered_per_index,
    T* dX_data,
    const int64_t dX_batch_size,  //The number of input elements starting from the gathering dimension
    const int64_t num_batches     //The size of dimensions of the data before gathering dimension
) {
  const auto total_num_dY_indices = num_indices;
  const auto total_num_dX_indices = dX_batch_size / num_gathered_per_index;

  // allocate intermediate buffers
  auto dY_indices = allocator.GetScratchBuffer<TIndex>(num_indices);

  // initialize dY_indices with [0, num_indices)
  {
    const auto blocks_per_grid = CeilDiv(num_indices, GridDim::maxThreadsPerBlock);
    cub::CountingInputIterator<TIndex> counting_input(TIndex{0});
    CopyKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock>>>(
        counting_input, num_indices, dY_indices.get());
  }

  // sort index lists together by dX_indices
  auto dX_indices_sorted = allocator.GetScratchBuffer<TIndex>(num_indices);
  auto dY_indices_sorted = allocator.GetScratchBuffer<TIndex>(num_indices);
  {
    size_t temp_storage_size_bytes = 0;
    CUDA_CALL_THROW(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_size_bytes,
        dX_indices, dX_indices_sorted.get(),
        dY_indices.get(), dY_indices_sorted.get(),
        num_indices));

    auto temp_storage = allocator.GetScratchBuffer<void>(temp_storage_size_bytes);
    CUDA_CALL_THROW(cub::DeviceRadixSort::SortPairs(
        temp_storage.get(), temp_storage_size_bytes,
        dX_indices, dX_indices_sorted.get(),
        dY_indices.get(), dY_indices_sorted.get(),
        num_indices));
  }

  // PrintDataKernel<<<1, 1>>>(dX_indices_sorted.get(), num_indices);
  // PrintDataKernel<<<1, 1>>>(dY_indices_sorted.get(), num_indices);

  // get stats about continuous indices
  auto dX_indices_unique_values = allocator.GetScratchBuffer<TIndex>(num_indices);
  auto dX_indices_counts = allocator.GetScratchBuffer<int32_t>(num_indices);
  auto num_unique_dX_indices = allocator.GetScratchBuffer<int32_t>(1);
  {
    size_t temp_storage_size_bytes = 0;
    CUDA_CALL_THROW(cub::DeviceRunLengthEncode::Encode(
        nullptr, temp_storage_size_bytes,
        dX_indices_sorted.get(),
        dX_indices_unique_values.get(),
        dX_indices_counts.get(),
        num_unique_dX_indices.get(),
        num_indices));

    auto temp_storage = allocator.GetScratchBuffer<void>(temp_storage_size_bytes);
    CUDA_CALL_THROW(cub::DeviceRunLengthEncode::Encode(
        temp_storage.get(), temp_storage_size_bytes,
        dX_indices_sorted.get(),
        dX_indices_unique_values.get(),
        dX_indices_counts.get(),
        num_unique_dX_indices.get(),
        num_indices));
  }

  // idea 1:
  // split into similarly sized segments
  // - one segment sums a fixed number of items
  // combine segment sums into output

  // idea 2:
  // use fancy iterators and cub::DeviceSegmentedReduce

  // TODO - this causes sync with CPU!
  int32_t host_num_unique_dX_indices;
  CUDA_CALL_THROW(cudaMemcpy(
      &host_num_unique_dX_indices, num_unique_dX_indices.get(), sizeof(int32_t), cudaMemcpyDeviceToHost));

  // PrintDataKernel<<<1, 1>>>(dX_indices_unique_values.get(), host_num_unique_dX_indices);
  // PrintDataKernel<<<1, 1>>>(dX_indices_counts.get(), host_num_unique_dX_indices);

  auto dX_indices_offsets = allocator.GetScratchBuffer<int32_t>(host_num_unique_dX_indices);
  {
    size_t temp_storage_size_bytes = 0;
    CUDA_CALL_THROW(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_size_bytes,
        dX_indices_counts.get(),
        dX_indices_offsets.get(),
        host_num_unique_dX_indices));

    auto temp_storage = allocator.GetScratchBuffer<void>(temp_storage_size_bytes);

    CUDA_CALL_THROW(cub::DeviceScan::ExclusiveSum(
        temp_storage.get(), temp_storage_size_bytes,
        dX_indices_counts.get(),
        dX_indices_offsets.get(),
        host_num_unique_dX_indices));
  }

  // PrintDataKernel<<<1, 1>>>(dX_indices_offsets.get(), host_num_unique_dX_indices);

  auto gather_from_dY = MakeGeneratedPermutation(
      dY_data,
      [total_num_dY_indices,
       num_gathered_per_index,
       dY_indices_sorted_raw = dY_indices_sorted.get()] __device__(IndexType id) {
        // what we want to do...
        // for i in num_batches
        //   for j in num_gathered_per_index
        //     for k in total_num_dY_indices
        //       read dY_data[total_num_dY_indices * num_gathered_per_index * i +
        //                    num_gathered_per_index * dY_indices_sorted[k] +
        //                    j]

        const auto i = id / (total_num_dY_indices * num_gathered_per_index);
        id %= (total_num_dY_indices * num_gathered_per_index);
        const auto j = id / total_num_dY_indices;
        const auto k = id % total_num_dY_indices;

        const auto dY_index = total_num_dY_indices * num_gathered_per_index * i +
                              num_gathered_per_index * dY_indices_sorted_raw[k] +
                              j;

        // printf(
        //     "gather_from_dY: i = %d, j = %d, k = %d, dY_index = %d\n",
        //     static_cast<int>(i), static_cast<int>(j), static_cast<int>(k), static_cast<int>(dY_index));

        return dY_index;
      });

  auto scatter_to_dX = MakeGeneratedPermutation(
      dX_data,
      [total_num_dX_indices,
       num_gathered_per_index,
       num_unique_dX_indices_raw = num_unique_dX_indices.get(),
       dX_indices_unique_values_raw = dX_indices_unique_values.get()] __device__(IndexType id) {
        // what we want to do...
        // for i in num_batches
        //   for j in num_gathered_per_index
        //     for k in num_unique_dX_indices
        //       write dX_data[total_num_dX_indices * num_gathered_per_index * i +
        //                     num_gathered_per_index * dX_indices_unique_values[k] +
        //                     j]

        const auto i = id / (*num_unique_dX_indices_raw * num_gathered_per_index);
        id %= (*num_unique_dX_indices_raw * num_gathered_per_index);
        const auto j = id / *num_unique_dX_indices_raw;
        const auto k = id % *num_unique_dX_indices_raw;

        const auto dX_index = total_num_dX_indices * num_gathered_per_index * i +
                              num_gathered_per_index * dX_indices_unique_values_raw[k] +
                              j;

        // printf(
        //     "scatter_to_dX: i = %d, j = %d, k = %d, dX_index = %d\n",
        //     static_cast<int>(i), static_cast<int>(j), static_cast<int>(k), static_cast<int>(dX_index));

        return dX_index;
      });

  auto segment_offsets = MakeIndexTransformIterator(
      [num_indices,
       num_gathered_per_index,
       num_unique_dX_indices_raw = num_unique_dX_indices.get(),
       dX_indices_offsets_raw = dX_indices_offsets.get()] __device__(IndexType id) {
        // what we want to do...
        // for i in num_batches
        //   for j in num_gathered_per_index
        //     for k in num_unique_dX_indices
        //       return num_gathered_per_index * num_indices * i +
        //              num_indices * j +
        //              dX_indices_offsets[k]

        const auto i = id / (*num_unique_dX_indices_raw * num_gathered_per_index);
        id %= (*num_unique_dX_indices_raw * num_gathered_per_index);
        const auto j = id / *num_unique_dX_indices_raw;
        const auto k = id % *num_unique_dX_indices_raw;

        const auto segment_offset = num_gathered_per_index * num_indices * i +
                                    num_indices * j +
                                    dX_indices_offsets_raw[k];

        // printf(
        //     "segment_offsets: i = %d, j = %d, k = %d, segment_offset = %d\n",
        //     static_cast<int>(i), static_cast<int>(j), static_cast<int>(k), static_cast<int>(segment_offset));

        return segment_offset;
      });

  const auto num_segments = num_batches * num_gathered_per_index * host_num_unique_dX_indices;

  {
    const auto reduction_op = Sum<T>{};
    const auto initial_value = T{0};

    size_t temp_storage_size_bytes = 0;
    CUDA_CALL_THROW(cub::DeviceSegmentedReduce::Reduce(
        nullptr, temp_storage_size_bytes,
        gather_from_dY.begin(),
        scatter_to_dX.begin(),
        num_segments,
        segment_offsets,
        segment_offsets + 1,
        reduction_op,
        initial_value));

    auto temp_storage = allocator.GetScratchBuffer<void>(temp_storage_size_bytes);
    CUDA_CALL_THROW(cub::DeviceSegmentedReduce::Reduce(
        temp_storage.get(), temp_storage_size_bytes,
        gather_from_dY.begin(),
        scatter_to_dX.begin(),
        num_segments,
        segment_offsets,
        segment_offsets + 1,
        reduction_op,
        initial_value));
  }
}

template <typename T, typename Tin>
void GatherGradImpl(
    const CudaScratchBufferAllocator& allocator,
    const T* grad_data,
    const Tin* indices_data,
    const int64_t num_indices,
    const int64_t num_weights,
    const int64_t stride,
    T* output_data,
    const int64_t num_inputs,  //The number of input elements starting from the gathering dimension
    const int64_t param_itrs,  //The size of dimensions of the data before gathering dimension
    GatherGradImplementation impl) {
  switch (impl) {
    case GatherGradImplementation::ThreadPerIndex:
      GatherGradImpl_ThreadPerUniqueIndex(allocator, grad_data, indices_data, num_indices, num_weights, stride, output_data, num_inputs, param_itrs);
      break;

    case GatherGradImplementation::FancyIterator:
      GatherGradImpl_FancyIterator(allocator, grad_data, indices_data, num_indices, num_weights, stride, output_data, num_inputs, param_itrs);
      break;

    default:
      ORT_NOT_IMPLEMENTED("Unsupported GatherGradImplementation: ", static_cast<int>(impl));
  }
}

#define SPECIALIZED_GRAD_IMPL2(T)                  \
  template void GatherGradImpl<T, int64_t>(        \
      const CudaScratchBufferAllocator& allocator, \
      const T* grad_data,                          \
      const int64_t* indices_data,                 \
      const int64_t num_indices,                   \
      const int64_t num_weights,                   \
      const int64_t stride,                        \
      T* output_data,                              \
      const int64_t num_inputs,                    \
      const int64_t param_itrs,                    \
      GatherGradImplementation impl);              \
  template void GatherGradImpl<T, int32_t>(        \
      const CudaScratchBufferAllocator& allocator, \
      const T* grad_data,                          \
      const int32_t* indices_data,                 \
      const int64_t num_indices,                   \
      const int64_t num_weights,                   \
      const int64_t stride,                        \
      T* output_data,                              \
      const int64_t num_inputs,                    \
      const int64_t param_itrs,                    \
      GatherGradImplementation impl);

SPECIALIZED_GRAD_IMPL2(float)
SPECIALIZED_GRAD_IMPL2(half)

}  // namespace cuda
}  // namespace onnxruntime
