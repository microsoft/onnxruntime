// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Size of global Index scratch in bytes.
size_t GetGlobalScratchSize(int batch_size, int sequence_length);

// Find the global attention indices and number of global attention tokens
void BuildGlobalIndex(
    cudaStream_t stream,
    const int* global_attention,
    int batch_size,
    int sequence_length,
    int* global_index,
    int* batch_global_num,
    void* scratch,
    size_t scratch_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
