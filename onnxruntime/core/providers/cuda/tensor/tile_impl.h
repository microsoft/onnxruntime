// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void TileImpl(
    cudaStream_t stream,
    const size_t shape_rank,
    const TArray<fast_divmod>& input_shape,
    const TArray<int64_t>& input_strides,
    const T* input_data,
    const TArray<fast_divmod>& fdm_output_strides,
    T* output_data,
    const size_t N);

template <typename T>
void TileMemcpyImpl(
    cudaStream_t stream,
    const T* input_data,
    const size_t num_input_elements,
    T* output_data,
    const size_t num_output_elements);

template <typename T>
void TileBatchedMemcpyImpl(
    cudaStream_t stream,
    const T* input_data,
    const size_t num_of_elements_per_input_batch,
    const size_t num_input_batch_count,
    const fast_divmod& num_of_elements_per_output_batch,
    T* output_data,
    const size_t num_output_elements);

}  // namespace cuda
}  // namespace onnxruntime
