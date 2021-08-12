// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

bool CanDoTranspose3D(int32_t rank, const std::vector<int64_t>& input_dims, const std::vector<size_t>& permutations);
Status Transpose3DImpl(cudaStream_t stream, size_t element_size, const TArray<int64_t>& input_shape, const TArray<int64_t>& input_strides, const void* input_data,
                       void* output_data, int64_t N);

bool CanDoTranspose4DParallelizeMultipleElementsPerThreadInInnermostDim(const cudaDeviceProp& prop,
                                                                        size_t element_size,
                                                                        int32_t rank,
                                                                        const std::vector<int64_t>& input_dims,
                                                                        const std::vector<size_t>& permutations);
Status Transpose4DParallelizeMultipleElementsPerThreadInInnermostDim(cudaStream_t stream, size_t element_size, const TArray<int64_t>& input_shape,
                                                                     const TArray<int64_t>& input_strides, const void* input_data,
                                                                     const TArray<int64_t>& output_strides, void* output_data, int N);

bool CanDoTranspose4DParallelizeOneElementPerThread(const cudaDeviceProp& prop,
                                                    size_t element_size,
                                                    int32_t rank,
                                                    const std::vector<int64_t>& input_dims,
                                                    const std::vector<size_t>& permutations);
Status Transpose4DParallelizeOneElementPerThread(cudaStream_t stream, size_t element_size, const TArray<int64_t>& input_shape,
                                                 const TArray<int64_t>& input_strides, const void* input_data,
                                                 const TArray<int64_t>& output_strides, void* output_data, int N);

Status TransposeImpl(cudaStream_t stream, size_t element_size, int32_t shape_rank, const TArray<int64_t>& input_strides,
                     const void* input_data, const TArray<fast_divmod>& fdm_output_strides, void* output_data, int N);
}  // namespace cuda
}  // namespace onnxruntime
