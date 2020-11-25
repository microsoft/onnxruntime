// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
cudaError_t ReverseSequenceCudaImpl(
    const T* x_data,
    const int64_t* seq_len_data,
    T* y_data,
    const int batch_size,
    const int max_seq_len,
    const int element_size,
    const bool time_major);

}  // namespace cuda
}  // namespace onnxruntime
