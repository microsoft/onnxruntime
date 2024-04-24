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
    const int64_t* indices_data,
    const int64_t last_index_dimension,
    const ElementCountsAndInputDimsSpanOrGpu& element_counts_and_input_dims,
    const void* updates_data,
    const size_t num_updates_elements);

Status ScatterNDImplReduction(
    cudaStream_t stream,
    void* output_data,
    const int32_t element_type,
    const size_t num_indices,
    const int64_t* indices_data,
    const int64_t last_index_dimension,
    const ElementCountsAndInputDimsSpanOrGpu& element_counts_and_input_dims,
    const void* updates_data,
    const size_t num_updates_elements,
    ScatterNDReduction reduction);

}  // namespace cuda
}  // namespace onnxruntime
