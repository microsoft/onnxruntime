// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template<typename T>
void ReverseBySequence(const int32_t seq_length,
                       const int32_t batch_size,
                       const int32_t input_or_hidden_size,
                       const T* data,
                       T* reversed_data,
                       const size_t N);

template <typename T>
void ReorderBidirectionalDataInSequence(const int32_t seq_length,
                                        const int32_t batch_size,
                                        const int32_t hidden_size,
                                        const T* data,
                                        T* reordered_data,
                                        const size_t N);

template <typename T>
void RnnMaskImpl(const int32_t num_directions,
                 const int32_t seq_length,
                 const int32_t batch_size,
                 const int32_t hidden_size,
                 const int32_t* sequence_lens,
                 T* y_output_data,
                 T* y_h_output_data,
                 const size_t N);

}  // namespace cuda
}  // namespace onnxruntime
