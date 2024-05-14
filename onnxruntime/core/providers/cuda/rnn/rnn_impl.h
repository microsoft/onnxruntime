// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void ReverseBySequence(cudaStream_t stream,
                       const int32_t max_seq_length,
                       const int32_t* seq_lengths,
                       const int32_t batch_size,
                       const int32_t input_or_hidden_size,
                       const T* data,
                       T* reversed_data,
                       const size_t N);

template <typename T>
void ReorderBidirectionalDataInSequence(cudaStream_t stream,
                                        const int32_t seq_length,
                                        const int32_t batch_size,
                                        const int32_t hidden_size,
                                        const T* data,
                                        T* reordered_data,
                                        const size_t N);

template <typename T>
void MaskZeroSequences(cudaStream_t stream,
                       const int32_t hidden_size,
                       T* y_output_data,
                       T* y_h_output_data,
                       T* y_c_output_data,
                       const int32_t* zeor_seq_index_cache_async_buffer,
                       const size_t N);
}  // namespace cuda
}  // namespace onnxruntime
