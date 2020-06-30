// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
bool LaunchMaskIndexKernel(cudaStream_t stream,   // stream
                           const T* mask,         // input mask, NULL when no mask input.
                           int* mask_index,       // output mask index
                           int batch_size,        // batch size
                           int sequence_length);  // sequence length

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
