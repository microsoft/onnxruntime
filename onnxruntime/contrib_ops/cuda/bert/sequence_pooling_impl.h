// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

bool LaunchSequencePoolingKernel(
    void* output,
    const void* input,
    const int batch_size,
    const int sequence_length_for_split,
    const int hidden_size,
    const int num_sequences,
    const size_t element_size
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
