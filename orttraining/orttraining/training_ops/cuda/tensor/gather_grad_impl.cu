// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/gather_grad_impl.h"

#include <iterator>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
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
    for (size_t i = 0; i < length; ++i) {
      printf("%f ", static_cast<float>(data[i]));
    }
    printf("\n");
  }
}

template <typename T>
void PrintData(const char* description, const T* data, size_t length) {
  printf("%s\n", description);
  PrintDataKernel<T><<<1, 1>>>(data, length);
  cudaDeviceSynchronize();
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

// adapted from here: https://github.com/pytorch/pytorch/blob/b186831c08e0e4e447eedb8a5cfab582995d37f9/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu
constexpr int kMaxPartialSegmentSize = 10;

template <typename TIndex>
__global__ void krn_partials_per_segment(
    TIndex* ret, const TIndex* segment_offsets,
    TIndex num_of_segments, TIndex numel) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < num_of_segments) {
    const TIndex idx_start = segment_offsets[id];
    const TIndex idx_end = (id == num_of_segments - 1) ? numel : segment_offsets[id + 1];
    const TIndex size = idx_end - idx_start;
    ret[id] = CeilDiv(size, kMaxPartialSegmentSize);
  }
}

template <typename TIndex>
__global__ void krn_partial_segment_offset(
    TIndex* ret,
    const TIndex* partials_per_segment,
    const TIndex* partials_per_segment_offset,
    const TIndex* segment_offsets,
    TIndex num_of_segments) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < num_of_segments) {
    TIndex idx = partials_per_segment_offset[id];
    const TIndex num_partials = partials_per_segment[id];
    const TIndex segment_offset = segment_offsets[id];
    for (TIndex i = 0; i < num_partials; ++i) {
      ret[idx++] = segment_offset + i * kMaxPartialSegmentSize;
    }
  }
}

template <typename T>
struct AccumulationType;
template <>
struct AccumulationType<half> { using type = float; };
template <>
struct AccumulationType<float> { using type = float; };

template <typename T>
using AccumulationType_t = typename AccumulationType<T>::type;

template <typename T, typename TIndex>
__global__ void compute_grad_weight(
    const TIndex* dY_indices_sorted,
    const T* dY_data,
    TIndex num_indices,
    TIndex num_gathered_per_index,
    const TIndex* partial_segment_offsets,
    TIndex num_partial_segments,
    AccumulationType_t<T>* partial_segment_sums,
    const TIndex num_gathered_per_index_warp_size_multiple) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int partial_segment_id = id / num_gathered_per_index_warp_size_multiple;
  const int gathered_id = id % num_gathered_per_index_warp_size_multiple;
  const int batch_id = blockIdx.y;

  if (gathered_id >= num_gathered_per_index) {
    return;
  }
  if (partial_segment_id >= num_partial_segments) {
    return;
  }

  const TIndex idx_begin = partial_segment_offsets[partial_segment_id];
  const TIndex idx_end =
      (partial_segment_id == num_partial_segments - 1) ? num_indices : partial_segment_offsets[partial_segment_id + 1];

  AccumulationType_t<T> partial_segment_sum = 0;
  for (TIndex idx = idx_begin; idx < idx_end; ++idx) {
    const TIndex target_row = dY_indices_sorted[idx];
    partial_segment_sum += static_cast<AccumulationType_t<T>>(
        dY_data[batch_id * num_indices * num_gathered_per_index +
                target_row * num_gathered_per_index +
                gathered_id]);
  }

  // printf("compute_partial_sums: batch %d, partial_segment %d, gathered %d, partial_sum %f",
  //        static_cast<int>(batch_id), static_cast<int>(partial_segment_id), static_cast<int>(gathered_id),
  //        static_cast<float>(partial_segment_sum));
  partial_segment_sums[batch_id * num_partial_segments * num_gathered_per_index +
                       partial_segment_id * num_gathered_per_index +
                       gathered_id] =
      partial_segment_sum;
}

// This kernel assumes that all input tensors are contiguous.
template <typename T, typename TIndex>
__global__ void sum_and_scatter(
    const TIndex* dX_indices_sorted, T* dX_data, TIndex num_gathered_per_index,
    const TIndex* segment_offsets, TIndex num_segments,
    const AccumulationType_t<T>* partial_segment_sums,
    const TIndex* per_segment_partial_segment_offsets, TIndex num_partial_segments,
    const TIndex num_gathered_per_index_warp_size_multiple,
    const TIndex dX_batch_size) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment_id = gid / num_gathered_per_index_warp_size_multiple;
  const int gathered_id = gid % num_gathered_per_index_warp_size_multiple;
  const int batch_id = blockIdx.y;

  if (gathered_id >= num_gathered_per_index) {
    return;
  }
  if (segment_id >= num_segments) {
    return;
  }

  const TIndex idx_begin = per_segment_partial_segment_offsets[segment_id];
  const TIndex idx_end =
      (segment_id == num_segments - 1) ? num_partial_segments : per_segment_partial_segment_offsets[segment_id + 1];

  AccumulationType_t<T> segment_sum = 0;
  for (TIndex idx = idx_begin; idx < idx_end; ++idx) {
    segment_sum +=
        partial_segment_sums[batch_id * num_partial_segments * num_gathered_per_index +
                             idx * num_gathered_per_index +
                             gathered_id];
  }

  const int64_t target_row = dX_indices_sorted[segment_offsets[segment_id]];
  dX_data[batch_id * dX_batch_size +
          target_row * num_gathered_per_index +
          gathered_id] =
      segment_sum;
}

template <typename T, typename TIndex>
void GatherGradImpl_PartialSums(
    const CudaScratchBufferAllocator& allocator,
    const T* dY_data,
    const TIndex* dX_indices,
    const TIndex num_indices,
    const int64_t /*num_weights*/,
    const TIndex num_gathered_per_index,
    T* dX_data,
    const TIndex dX_batch_size,
    const TIndex num_batches) {
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
  // PrintData("dX_indices_sorted", dX_indices_sorted.get(), num_indices);
  // PrintData("dY_indices_sorted", dY_indices_sorted.get(), num_indices);

  // a segment is a group of dX and dY indices. each continuous run of indices
  // with the same value in dX_indices_sorted forms a segment.
  // for example, given:
  //   dX_indices_sorted = [1, 1, 2, 2, 2, 3]
  //   dY_indices_sorted = [1, 4, 0, 3, 5, 2]
  // the segments will be:  '--'  '-----'  '

  // get number of segments and segment offsets
  TIndex host_num_segments = 0;
  auto segment_offsets = allocator.GetScratchBuffer<TIndex>(num_indices);
  {
    auto num_segments = allocator.GetScratchBuffer<TIndex>(1);
    auto dX_indices_sorted_unique = allocator.GetScratchBuffer<TIndex>(num_indices);
    size_t temp_storage_size_bytes = 0;
    CUDA_CALL_THROW(cub::DeviceReduce::ReduceByKey(
        nullptr, temp_storage_size_bytes,
        dX_indices_sorted.get(),
        dX_indices_sorted_unique.get(),
        thrust::make_counting_iterator<TIndex>(0),
        segment_offsets.get(),
        num_segments.get(),
        cub::Min{},
        num_indices));

    auto temp_storage = allocator.GetScratchBuffer<void>(temp_storage_size_bytes);
    CUDA_CALL_THROW(cub::DeviceReduce::ReduceByKey(
        temp_storage.get(), temp_storage_size_bytes,
        dX_indices_sorted.get(),
        dX_indices_sorted_unique.get(),
        thrust::make_counting_iterator(0),
        segment_offsets.get(),
        num_segments.get(),
        cub::Min{},
        num_indices));

    // CPU/GPU sync!
    CUDA_CALL_THROW(cudaMemcpy(
        &host_num_segments, num_segments.get(), sizeof(TIndex), cudaMemcpyDeviceToHost));
  }
  // PrintData("segment_offsets", segment_offsets.get(), host_num_segments);

  // each segment is split into partial segments of at most
  // kMaxPartialSegmentSize index pairs.

  // compute the number of partial segments per segment
  auto per_segment_partial_segment_counts = allocator.GetScratchBuffer<TIndex>(host_num_segments);
  {
    const auto blocks_per_grid = CeilDiv(num_indices, GridDim::maxThreadsPerBlock);
    krn_partials_per_segment<<<blocks_per_grid, GridDim::maxThreadsPerBlock>>>(
        per_segment_partial_segment_counts.get(),
        segment_offsets.get(), host_num_segments, num_indices);
  }
  // PrintData("per_segment_partial_segment_counts", per_segment_partial_segment_counts.get(), host_num_segments);

  // compute offsets (in partial segments) per segment
  auto per_segment_partial_segment_offsets = allocator.GetScratchBuffer<TIndex>(host_num_segments);
  {
    size_t temp_storage_size_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_size_bytes,
        per_segment_partial_segment_counts.get(),
        per_segment_partial_segment_offsets.get(),
        host_num_segments);

    auto temp_storage = allocator.GetScratchBuffer<void>(temp_storage_size_bytes);
    cub::DeviceScan::ExclusiveSum(
        temp_storage.get(), temp_storage_size_bytes,
        per_segment_partial_segment_counts.get(),
        per_segment_partial_segment_offsets.get(),
        host_num_segments);
  }
  // PrintData("per_segment_partial_segment_offsets", per_segment_partial_segment_offsets.get(), host_num_segments);

  TIndex host_num_partial_segments = 0;
  {
    // CPU/GPU sync!
    TIndex last_segment_partial_segment_offset, last_segment_num_partial_segments;
    CUDA_CALL_THROW(cudaMemcpy(
        &last_segment_partial_segment_offset,
        &per_segment_partial_segment_offsets.get()[host_num_segments - 1],
        sizeof(TIndex), cudaMemcpyDeviceToHost));
    CUDA_CALL_THROW(cudaMemcpy(
        &last_segment_num_partial_segments,
        &per_segment_partial_segment_counts.get()[host_num_segments - 1],
        sizeof(TIndex), cudaMemcpyDeviceToHost));
    host_num_partial_segments =
        last_segment_partial_segment_offset + last_segment_num_partial_segments;
  }

  auto partial_segment_offsets = allocator.GetScratchBuffer<TIndex>(host_num_partial_segments);
  {
    const auto blocks_per_grid = CeilDiv(host_num_segments, GridDim::maxThreadsPerBlock);
    krn_partial_segment_offset<<<blocks_per_grid, GridDim::maxThreadsPerBlock>>>(
        partial_segment_offsets.get(),
        per_segment_partial_segment_counts.get(),
        per_segment_partial_segment_offsets.get(),
        segment_offsets.get(),
        host_num_segments);
  }
  // PrintData("partial_segment_offsets", partial_segment_offsets.get(), host_num_partial_segments);

  {
    auto partial_segment_sums = allocator.GetScratchBuffer<AccumulationType_t<T>>(
        num_batches * host_num_partial_segments * num_gathered_per_index);
    const auto num_gathered_per_index_warp_size_multiple =
        CeilDiv(num_gathered_per_index, GPU_WARP_SIZE) * GPU_WARP_SIZE;
    const auto threads_per_block =
        std::min<TIndex>(num_gathered_per_index_warp_size_multiple, GridDim::maxThreadsPerBlock);
    {
      const dim3 blocks_per_grid(
          CeilDiv(host_num_partial_segments * num_gathered_per_index_warp_size_multiple, threads_per_block),
          num_batches);
      compute_grad_weight<<<blocks_per_grid, threads_per_block>>>(
          dY_indices_sorted.get(),
          dY_data,
          num_indices,
          num_gathered_per_index,
          partial_segment_offsets.get(),
          host_num_partial_segments,
          partial_segment_sums.get(),
          num_gathered_per_index_warp_size_multiple);
    }
    // PrintData("partial_segment_sums", partial_segment_sums.get(),
    //           num_batches * host_num_partial_segments * num_gathered_per_index);

    {
      const dim3 blocks_per_grid(
          CeilDiv(host_num_segments * num_gathered_per_index_warp_size_multiple, threads_per_block),
          num_batches);
      sum_and_scatter<<<blocks_per_grid, threads_per_block>>>(
          dX_indices_sorted.get(),
          dX_data,
          num_gathered_per_index,
          segment_offsets.get(),
          host_num_segments,
          partial_segment_sums.get(),
          per_segment_partial_segment_offsets.get(),
          host_num_partial_segments,
          num_gathered_per_index_warp_size_multiple,
          dX_batch_size);
    }
  }
}

template <typename T, typename TIndex>
void GatherGradImpl(
    const CudaScratchBufferAllocator& allocator,
    const T* grad_data,
    const TIndex* indices_data,
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

    case GatherGradImplementation::PartialSums:
      GatherGradImpl_PartialSums(allocator, grad_data, indices_data, static_cast<TIndex>(num_indices), static_cast<TIndex>(num_weights), static_cast<TIndex>(stride), output_data, static_cast<TIndex>(num_inputs), static_cast<TIndex>(param_itrs));
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
