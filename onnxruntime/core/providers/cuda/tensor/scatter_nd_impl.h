// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

Status ScatterNDImpl(
    cudaStream_t stream,
    void* output_data,
    const size_t element_size,
    const size_t num_indices,
    const int64_t* indices_data,
    const int64_t last_index_dimension,
    const int64_t* element_counts_and_input_dims,
    const void* updates_data,
    const size_t num_updates_elements);

void cudaRandomUniform(cudaStream_t stream, void* buffer, const int size);

void SubnormalFlush(cudaStream_t stream,
                    void* output,           // output tensor
                    const int hidden_size,  // hidden size (that is head_size * num_heads)
                    int batch_size,         // batch size
                    int sequence_length,    // sequence length
                    int flush_all_to_zero = 0);
}  // namespace cuda
}  // namespace onnxruntime
