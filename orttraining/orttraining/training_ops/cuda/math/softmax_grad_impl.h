// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {
template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_backward(cudaStream_t stream, output_t* grad_input, const input_t* grad, const input_t* output,
                               int softmax_elements, int softmax_elements_stride, int batch_count);
}
}  // namespace onnxruntime
