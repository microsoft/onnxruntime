// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/tensor/scatter_nd_kind.h"
#include "core/providers/cuda/tensor/scatter_nd_common.h"

namespace onnxruntime {
namespace cuda {

Status ScatterNDImpl(
    cudaStream_t stream,
    void* output_data,
    const size_t element_size,
    const size_t num_indices,
    const void* indices_data,
    const int64_t last_index_dimension,
    const ElementCountsAndInputDimsSpanOrGpu& element_counts_and_input_dims,
    const void* updates_data,
    const size_t num_updates_elements,
    size_t index_element_size = sizeof(int64_t),
    bool strict_indices = false);

Status ScatterNDImplReduction(
    cudaStream_t stream,
    void* output_data,
    const int32_t element_type,
    const size_t num_indices,
    const void* indices_data,
    const int64_t last_index_dimension,
    const ElementCountsAndInputDimsSpanOrGpu& element_counts_and_input_dims,
    const void* updates_data,
    const size_t num_updates_elements,
    ScatterNDReduction reduction,
    size_t index_element_size = sizeof(int64_t),
    bool strict_indices = false);

Status ScatterMeanImpl(cudaStream_t stream, int32_t element_type, const void* input, const void* updates,
                       void* output, const int64_t* heads, const int64_t* next, int64_t size, int64_t slice_size);

Status ScatterNDMeanImpl(cudaStream_t stream, const void* indices, size_t index_element_size,
                         int64_t num_indices, int64_t index_depth,
                         const ElementCountsAndInputDimsSpanOrGpu& dimensions,
                         int64_t* heads, int64_t* next, int64_t slice_size, bool deterministic);

}  // namespace cuda
}  // namespace onnxruntime
