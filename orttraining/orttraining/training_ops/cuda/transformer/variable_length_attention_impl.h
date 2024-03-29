// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
// #include <stdint.h>
// #include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/framework/stream_handles.h"
// #include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

// o matching function for call to ‘LaunchGroupTranspose(cudaStream_t,
// size_t&,
//  const int32_t*,
// int&, int&, onnxruntime::cuda::TArray<long int>&, onnxruntime::cuda::TArray<long int>&, int64_t&,
//  std::vector<long unsigned int>&,
//  const float*&, std::unique_ptr<float, std::function<void(float*)> >::pointer, int64_t)’

template <typename T>
Status LaunchGroupTranspose(
    cudaStream_t stream,
    size_t element_size,
    const int64_t* cum_seq_length,
    int variant_axis_in_output,
    int variant_axis_on_output,
    const TArray<int64_t> input_shape,
    const TArray<int64_t> output_shape,
    int64_t factor_for_fixed_dims,
    const TArray<int> reverse_perms,
    const T* input_data,
    T* output_data,
    size_t N);

// template <typename T>
// Status LaunchGroupTranspose(cudaStream_t stream,
//                             size_t element_size,
//                             const int64_t* cum_seq_length,
//                             int variant_axis_in_output,
//                             // int variant_axis_on_output,
//                             const TArray<int64_t> input_shape,
//                             // const TArray<int64_t> output_shape,
//                             int64_t factor_for_fixed_dims,
//                             const TArray<int> reverse_perms,
//                             const T* input_data,
//                             const TArray<fast_divmod> output_strides,
//                             T* output_data,
//                             size_t N);

// template <typename T>
// void UnfusedScaledDotProductAttentionVariableSeqlenImpl(
//     cudaStream_t stream,
//     const cudaDeviceProp& prop,
//     const CudaScratchBufferAllocator& allocator,
//     const T* dY_data,
//     const TIndex* dX_indices,
//     const GatheredIndexIndex_t num_gathered_indices,
//     const int64_t gather_dimension_size,
//     const int64_t num_gathered_per_index,
//     const int64_t num_batches,
//     T* dX_data);

template <typename T, typename TOut, bool is_log_softmax>
Status SoftMaxVarLengthComputeHelper(
    Stream* stream,
    const T* X,
    const int64_t max_seq_lengh,
    const int64_t* cu_seqlens,
    const int64_t seqlen_count,
    const int64_t head_count,
    TOut* Y);

}  // namespace cuda
}  // namespace onnxruntime
