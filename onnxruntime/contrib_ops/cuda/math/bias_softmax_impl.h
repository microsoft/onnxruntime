// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status BiasSoftmaxImpl(cudaStream_t stream, cudnnHandle_t cudnn_handle, T* output_data, const T* input_data,
                       const T* bias_data, int element_count, int batch_count, bool is_inner_broadcast,
                       int bias_broadcast_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
